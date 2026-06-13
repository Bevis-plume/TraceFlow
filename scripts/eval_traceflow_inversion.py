"""
scripts/eval_traceflow_inversion.py
=====================================
TraceFlow inversion evaluation harness.

Runs gradient-matching inversion attacks (latent-space and/or pixel-space)
against a trained TraceFlow checkpoint and reports image-quality and
watermark-detection metrics.

Usage
-----
    python -m scripts.eval_traceflow_inversion \\
      --config  configs/defaults/exp04_traceflow.yml \\
      --checkpoint checkpoints/flow_transformer/smoke/latest.pt \\
      --attack latent --steps 5 --batch-size 1

    python -m scripts.eval_traceflow_inversion \\
      --config  configs/defaults/exp04_traceflow.yml \\
      --checkpoint checkpoints/flow_transformer/smoke/latest.pt \\
      --attack pixel  --steps 5 --batch-size 1

Threat model
------------
* Attacker has access to model/AE/watermark weights but NOT ``secret_key``.
* The evaluation script uses ``secret_key`` only for:
  1. Defender-side forensic decode of latent attack results.
  2. Latent-detector verification (re-encode path).
* ``secret_key`` is NEVER written to output JSON.

Output (under ``--output-dir``)
---------------------------------
images/
    original_grid.png
    latent_<attacker>_raw_nokey_grid.png             (latent attack, raw no-key decode)
    latent_<attacker>_raw_defender_grid.png          (latent attack, raw defender-key decode)
    latent_<attacker>_post_watermark_defender_grid.png  (defender decode, adapter re-applied)
    pixel_<attacker>_raw_recon_grid.png              (pixel attack, raw reconstruction)
    pixel_<attacker>_post_watermark_recon_grid.png   (pixel reconstruction, adapter re-applied)
metrics.json

Scientific-validity notes
-------------------------
* Target and dummy gradients share a fixed ``(t, eps, bits)`` realisation
  (``AttackBatchState``) so the gradient-matching objective is noise-free.
* Watermark detection is performed on the *raw* attack output (``raw_*`` fields)
  WITHOUT applying the decoder_adapter — applying it first would re-stamp the
  watermark and produce a circular metric.  ``post_watermark_*`` fields are an
  explicit-adapter sanity check only.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_pil_image


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _force_math_sdp_for_higher_order_grads(device: torch.device) -> None:
    """Use the SDPA backend that supports inversion's higher-order gradients.

    The inversion objective backpropagates through gradients. CUDA's flash and
    memory-efficient SDPA kernels are fast for first-order training, but PyTorch
    does not implement the derivative needed for this second backward pass.
    Evaluation is correctness-first, so force the math backend here only.
    """
    if device.type != "cuda":
        return
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[eval] SDPA backend: math only for higher-order gradients")
    except Exception as exc:  # pragma: no cover - defensive for older PyTorch
        print(f"[eval] warning: could not force math SDPA backend: {exc}")


def _free_cuda(device: torch.device) -> None:
    """Release cached CUDA blocks between expensive eval stages."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Image-quality metrics (operate on [-1, 1] tensors)
# ---------------------------------------------------------------------------

def _to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((_to_01(a) - _to_01(b)) ** 2).item()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    m = _mse(a, b)
    if m == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / m)


def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Simplified SSIM for batched RGB images in [-1, 1]."""
    x = _to_01(a)
    y = _to_01(b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = x.mean(dim=(2, 3), keepdim=True)
    mu_y = y.mean(dim=(2, 3), keepdim=True)
    sig_x  = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_y  = ((y - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2)
    return (num / den).mean().item()


def _quality_pair(a: torch.Tensor, b: torch.Tensor, prefix: str) -> Dict[str, Any]:
    try:
        from src.utils.quality_metrics import pair_quality
        return pair_quality(a, b, prefix=prefix)
    except Exception as exc:
        return {f"{prefix}_metric_warning": str(exc)}


# ---------------------------------------------------------------------------
# Grid saving
# ---------------------------------------------------------------------------

def _save_grid(images: torch.Tensor, path: Path, nrow: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(
        images.clamp(-1, 1), nrow=nrow, normalize=True, value_range=(-1, 1)
    )
    to_pil_image(grid).save(str(path))


ROBUSTNESS_TRANSFORMS = [
    "clean",
    "jpeg",
    "resize",
    "blur",
    "gaussian_noise",
    "crop_resize",
]


def _apply_robustness_transform(
    images: torch.Tensor,
    transform_name: str,
) -> torch.Tensor:
    """Apply a deterministic image transform to raw attack outputs.

    Input and output are batched image tensors in [-1, 1].
    """
    images_01 = _to_01(images)
    _, _, height, width = images_01.shape

    if transform_name == "clean":
        transformed = images_01
    elif transform_name == "jpeg":
        jpeg_images: List[torch.Tensor] = []
        for image in images_01:
            pil_img = to_pil_image(image.cpu())
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=55)
            buffer.seek(0)
            jpeg_pil = Image.open(buffer).convert("RGB")
            jpeg_tensor = TF.pil_to_tensor(jpeg_pil).float() / 255.0
            jpeg_images.append(jpeg_tensor)
        transformed = torch.stack(jpeg_images, dim=0)
    elif transform_name == "resize":
        down_h = max(16, height // 2)
        down_w = max(16, width // 2)
        transformed = TF.resize(
            images_01,
            [down_h, down_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        transformed = TF.resize(
            transformed,
            [height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    elif transform_name == "blur":
        transformed = TF.gaussian_blur(images_01, kernel_size=[5, 5], sigma=[1.2, 1.2])
    elif transform_name == "gaussian_noise":
        transformed = torch.clamp(images_01 + 0.05 * torch.randn_like(images_01), 0.0, 1.0)
    elif transform_name == "crop_resize":
        crop_h = max(16, int(height * 0.75))
        crop_w = max(16, int(width * 0.75))
        top = max(0, (height - crop_h) // 2)
        left = max(0, (width - crop_w) // 2)
        cropped = TF.crop(images_01, top=top, left=left, height=crop_h, width=crop_w)
        transformed = TF.resize(
            cropped,
            [height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    else:
        raise ValueError(f"Unknown robustness transform: {transform_name!r}")

    return transformed.mul(2.0).sub(1.0).clamp(-1.0, 1.0)


def _evaluate_transform_robustness(
    images: torch.Tensor,
    *,
    prefix: str,
    img_dir: Path,
    watermark_modules: Dict[str, Any],
    latent_transform: Any,
    autoencoder: Any,
    batch_bits: torch.Tensor,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluate detector accuracy after deterministic transforms on raw attack outputs."""
    robustness: Dict[str, Dict[str, float]] = {}
    for transform_name in ROBUSTNESS_TRANSFORMS:
        x_t = _apply_robustness_transform(images, transform_name)
        _save_grid(
            x_t.cpu(),
            img_dir / f"{prefix}_robustness_{transform_name}_grid.png",
            nrow=max(1, x_t.size(0)),
        )
        raw_metrics = _detect_raw(
            watermark_modules,
            latent_transform,
            autoencoder,
            x_t,
            batch_bits,
            device,
        )
        robustness[transform_name] = {
            "image_bit_acc": raw_metrics["image_bit_acc"],
            "latent_bit_acc": raw_metrics["latent_bit_acc"],
        }
    return robustness


def _sample_filename(dataset: Any, idx: int) -> Optional[str]:
    """Best-effort filename lookup for a dataset sample index."""
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list) and 0 <= idx < len(samples):
        sample = samples[idx]
        if isinstance(sample, (list, tuple)) and sample:
            first = sample[0]
            if isinstance(first, (str, Path)):
                return str(first)
        if isinstance(sample, (str, Path)):
            return str(sample)

    paths = getattr(dataset, "paths", None)
    if isinstance(paths, list) and 0 <= idx < len(paths):
        path = paths[idx]
        if isinstance(path, (str, Path)):
            return str(path)

    return None


def _load_source_images(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load the source image batch for inversion evaluation.

    ``random`` preserves the previous smoke/backward-compatible path.
    ``config`` builds the dataset described in the YAML config and selects a
    deterministic contiguous slice starting at ``sample_index``.
    """
    source_count = args.num_source_samples or args.batch_size
    if source_count <= 0:
        raise ValueError(f"num_source_samples must be positive, got {source_count}.")

    if args.data_source == "random":
        x_orig = torch.randn(source_count, 3, image_size, image_size, device=device)
        x_orig = x_orig.tanh()
        y_orig = torch.zeros(source_count, device=device, dtype=torch.long)
        source_info = {
            "data_source": "random",
            "sample_index": None,
            "num_source_samples": source_count,
            "dataset_name": "random",
            "dataset_root": None,
            "samples": [
                {"index": None, "filename": None}
                for _ in range(source_count)
            ],
            "labels": y_orig.detach().cpu().tolist(),
        }
        print(
            f"[eval] Using {source_count} random source image(s) | size={image_size}x{image_size}"
        )
        return x_orig, y_orig, source_info

    from src.data.image_datasets import build_dataset

    data_cfg = cfg["data"]
    dataset = build_dataset(
        name=data_cfg["name"],
        root=data_cfg.get("root", "./data"),
        image_size=image_size,
        download=data_cfg.get("download", False),
        smoke=False,
        smoke_samples=source_count,
    )

    start_idx = args.sample_index
    end_idx = start_idx + source_count
    if start_idx < 0:
        raise ValueError(f"sample_index must be >= 0, got {start_idx}.")
    if end_idx > len(dataset):
        raise ValueError(
            f"Requested dataset slice [{start_idx}:{end_idx}) exceeds dataset length {len(dataset)}."
        )

    images: List[torch.Tensor] = []
    labels: List[int] = []
    source_samples: List[Dict[str, Any]] = []
    for idx in range(start_idx, end_idx):
        sample = dataset[idx]
        image = sample[0] if isinstance(sample, (list, tuple)) else sample
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"Dataset sample {idx} did not return a tensor image; got {type(image)!r}."
            )
        images.append(image)
        label = sample[1] if isinstance(sample, (list, tuple)) and len(sample) > 1 else 0
        labels.append(int(label))
        source_samples.append(
            {
                "index": idx,
                "filename": _sample_filename(dataset, idx),
                "label": int(label),
            }
        )

    x_orig = torch.stack(images, dim=0).to(device)
    y_orig = torch.tensor(labels, device=device, dtype=torch.long)
    source_info = {
        "data_source": "config",
        "sample_index": start_idx,
        "num_source_samples": source_count,
        "dataset_name": data_cfg["name"],
        "dataset_root": data_cfg.get("root"),
        "samples": source_samples,
        "labels": labels,
    }
    print(
        f"[eval] Using config dataset samples [{start_idx}:{end_idx}) from "
        f"{data_cfg['name']} | size={image_size}x{image_size}"
    )
    return x_orig, y_orig, source_info


# ---------------------------------------------------------------------------
# Watermark detection helpers
# ---------------------------------------------------------------------------

def _detect_raw(
    watermark_modules: Dict[str, Any],
    latent_transform: Any,
    autoencoder: Any,
    x_img: torch.Tensor,
    batch_bits: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """RAW watermark detection on an image *as produced by the attack*.

    Crucially this does **not** apply the decoder_adapter — it measures whether
    the watermark actually survives in the attacker's reconstruction.  Applying
    the adapter first would re-stamp the watermark and yield a circular,
    scientifically invalid metric.

    Returns dict with ``image_bit_acc/ber`` and ``latent_bit_acc/ber``.
    """
    from src.watermarking.metrics import bit_accuracy, bit_error_rate

    extractor = watermark_modules["extractor"]
    latent_detector = watermark_modules["latent_detector"]
    with torch.no_grad():
        x_dev = x_img.to(device)
        bit_probs_img = extractor(x_dev)
        z_re_k = latent_transform(autoencoder.encode(x_dev))
        bit_probs_lat = latent_detector(z_re_k)
        return {
            "image_bit_acc":  bit_accuracy(bit_probs_img, batch_bits),
            "image_ber":      bit_error_rate(bit_probs_img, batch_bits),
            "latent_bit_acc": bit_accuracy(bit_probs_lat, batch_bits),
            "latent_ber":     bit_error_rate(bit_probs_lat, batch_bits),
        }


def _detect_post_watermark(
    watermark_modules: Dict[str, Any],
    latent_transform: Any,
    autoencoder: Any,
    x_img: torch.Tensor,
    batch_bits: torch.Tensor,
    device: torch.device,
) -> tuple:
    """Sanity-check detection AFTER explicitly applying the decoder_adapter.

    This is *not* the headline metric — it only confirms the watermark machinery
    works when the defender deliberately re-stamps the watermark.  Returns
    ``(metrics_dict, watermarked_image_cpu)``.
    """
    from src.watermarking.metrics import bit_accuracy, bit_error_rate

    extractor = watermark_modules["extractor"]
    latent_detector = watermark_modules["latent_detector"]
    decoder_adapter = watermark_modules["decoder_adapter"]
    alpha = watermark_modules["config"]["alpha"]
    with torch.no_grad():
        x_dev = x_img.to(device)
        residual = decoder_adapter(x_dev, batch_bits)
        x_w = torch.clamp(x_dev + alpha * residual, -1.0, 1.0)
        bit_probs_img = extractor(x_w)
        z_re_k = latent_transform(autoencoder.encode(x_w))
        bit_probs_lat = latent_detector(z_re_k)
        metrics = {
            "image_bit_acc":  bit_accuracy(bit_probs_img, batch_bits),
            "image_ber":      bit_error_rate(bit_probs_img, batch_bits),
            "latent_bit_acc": bit_accuracy(bit_probs_lat, batch_bits),
            "latent_ber":     bit_error_rate(bit_probs_lat, batch_bits),
        }
    return metrics, x_w.cpu().clamp(-1, 1)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:  # noqa: C901  (complex but sequential)
    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg["project"].get("device", "auto"))
    print(f"[eval] Device: {device}")
    _force_math_sdp_for_higher_order_grads(device)

    # ------------------------------------------------------------------
    # 2. Load checkpoint
    # ------------------------------------------------------------------
    print(f"[eval] Checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)

    m_cfg   = state.get("model_cfg", cfg["model"])
    ae_saved = state.get("ae_cfg", None)

    # ------------------------------------------------------------------
    # 3. Autoencoder
    # ------------------------------------------------------------------
    from src.models.autoencoder_backend import AutoencoderBackend

    if ae_saved is not None:
        autoencoder = AutoencoderBackend(
            backend=ae_saved.get("backend", "local"),
            pretrained_model_name_or_path=ae_saved.get("pretrained_model_name_or_path"),
            latent_channels=ae_saved["latent_channels"],
            image_size=ae_saved["image_size"],
            latent_size=ae_saved["latent_size"],
            scaling_factor=ae_saved.get("scaling_factor", 1.0),
            freeze=True,
            base_channels=ae_saved.get("base_channels", 64),
        ).to(device)
        ae_lc = ae_saved["latent_channels"]
        ae_ls = ae_saved["latent_size"]
        image_size = ae_saved["image_size"]
    else:
        ae_cfg = cfg["autoencoder"]
        autoencoder = AutoencoderBackend(
            backend=ae_cfg.get("backend", "local"),
            pretrained_model_name_or_path=ae_cfg.get("pretrained_model_name_or_path"),
            latent_channels=ae_cfg["latent_channels"],
            image_size=cfg["data"]["image_size"],
            latent_size=ae_cfg["latent_size"],
            scaling_factor=ae_cfg.get("scaling_factor", 1.0),
            freeze=True,
            base_channels=ae_cfg.get("base_channels", 64),
        ).to(device)
        ae_lc = ae_cfg["latent_channels"]
        ae_ls = ae_cfg["latent_size"]
        image_size = cfg["data"]["image_size"]

    autoencoder.eval()

    # ------------------------------------------------------------------
    # 4. Latent transform (defender has the key)
    # ------------------------------------------------------------------
    transform_meta = state.get("transform_meta", {"type": "identity"})
    transform_type = transform_meta.get("type", "identity")
    sec_cfg = cfg.get("security", {})

    from src.security.factory import build_latent_transform
    latent_transform = build_latent_transform(
        sec_cfg,
        latent_channels=ae_lc,
        latent_size=ae_ls,
    ).to(device)
    print(f"[eval] Latent transform: {transform_type}")

    # ------------------------------------------------------------------
    # 5. FlowTransformer
    # ------------------------------------------------------------------
    from src.models.flow_transformer import build_flow_transformer

    _has_resolved = all(k in m_cfg for k in ("hidden_size", "depth", "num_heads"))
    _preset = None if _has_resolved else m_cfg.get("preset")

    model = build_flow_transformer(
        preset=_preset,
        latent_channels=m_cfg["latent_channels"],
        latent_size=m_cfg["latent_size"],
        patch_size=m_cfg.get("patch_size", 2),
        hidden_size=m_cfg.get("hidden_size", 512),
        depth=m_cfg.get("depth", 12),
        num_heads=m_cfg.get("num_heads", 8),
        mlp_ratio=m_cfg.get("mlp_ratio", 4.0),
        dropout=0.0,
        class_conditional=m_cfg.get("class_conditional", False),
        num_classes=m_cfg.get("num_classes"),
    ).to(device)

    # Load weights (prefer EMA)
    if "ema_model" in state:
        ema_state = state["ema_model"]
        ms = model.state_dict()
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema_state:
                ms[name] = ema_state[name].to(device)
        model.load_state_dict(ms)
        print("[eval] Using EMA weights.")
    else:
        model.load_state_dict(state["model"])

    model.eval()

    # ------------------------------------------------------------------
    # 6. Watermark modules (from checkpoint state)
    # ------------------------------------------------------------------
    wm_state = state.get("watermark", None)
    watermark_modules: Optional[Dict[str, Any]] = None

    if wm_state is not None and wm_state.get("config", {}).get("enabled", False):
        from src.watermarking.image_watermark import ImageWatermarkDetector
        from src.watermarking.decoder_watermark import TraceDecoderAdapter
        from src.watermarking.latent_watermark import TraceLatentDetector
        from src.watermarking.message import generate_watermark_bits

        wm_cfg = wm_state["config"]
        wm_type = wm_state.get("type", wm_cfg.get("type", "traceflow"))
        if wm_type != "traceflow":
            raise ValueError(f"Unsupported watermark checkpoint type: {wm_type!r}")

        extractor = ImageWatermarkDetector(
            bit_length=wm_cfg["bit_length"],
            image_size=image_size,
            channels=3,
            hidden_dim=wm_cfg["extractor_hidden_dim"],
            base_channels=wm_cfg.get("detector_base_channels", 64),
            num_scales=wm_cfg.get("detector_num_scales", 4),
            num_blocks=wm_cfg.get("detector_num_blocks", 2),
            max_channels=wm_cfg.get("detector_max_channels", 768),
        ).to(device)
        extractor.load_state_dict(wm_state["extractor"])
        extractor.eval()

        decoder_adapter = TraceDecoderAdapter(
            bit_length=wm_cfg["bit_length"],
            channels=3,
            hidden_dim=wm_cfg["extractor_hidden_dim"],
            image_size=image_size,
            base_channels=wm_cfg.get("adapter_base_channels", 64),
            num_blocks=wm_cfg.get("adapter_num_blocks", 3),
            max_channels=wm_cfg.get("adapter_max_channels", 512),
        ).to(device)
        decoder_adapter.load_state_dict(wm_state["decoder_adapter"])
        decoder_adapter.eval()

        latent_detector = TraceLatentDetector(
            bit_length=wm_cfg["bit_length"],
            latent_channels=ae_lc,
            hidden_dim=wm_cfg.get("latent_detector_hidden_dim", 128),
            base_channels=wm_cfg.get("latent_detector_base_channels", 64),
            num_blocks=wm_cfg.get("latent_detector_num_blocks", 3),
            max_channels=wm_cfg.get("latent_detector_max_channels", 512),
        ).to(device)
        latent_detector.load_state_dict(wm_state["latent_detector"])
        latent_detector.eval()

        bits = generate_watermark_bits(wm_cfg["bit_length"], wm_cfg["seed"], device=device)

        watermark_modules = {
            "enabled": True,
            "type": wm_type,
            "bits": bits,
            "extractor": extractor,
            "decoder_adapter": decoder_adapter,
            "latent_detector": latent_detector,
            "config": wm_cfg,
        }
        print(f"[eval] Watermark: {wm_type} | bits={wm_cfg['bit_length']}")
    else:
        print("[eval] No watermark state found in checkpoint.")

    # ------------------------------------------------------------------
    # 7. Source images (random by default; config-backed for paper-valid eval)
    # ------------------------------------------------------------------
    x_orig, y_source, source_info = _load_source_images(args, cfg, image_size, device)
    B = x_orig.size(0)
    y_target = y_source if m_cfg.get("class_conditional", False) else None
    if y_target is not None:
        print(f"[eval] Class labels: {y_target.detach().cpu().tolist()}")

    # ------------------------------------------------------------------
    # 8. Compute target gradients (defender side, with key)
    # ------------------------------------------------------------------
    from src.attacks.traceflow_inversion import (
        compute_target_gradients,
        latent_inversion_attack,
        pixel_inversion_attack,
    )

    print("[eval] Computing target gradients …")
    target_grads, attack_state = compute_target_gradients(
        model=model,
        autoencoder=autoencoder,
        latent_transform=latent_transform,
        watermark_modules=watermark_modules,
        x=x_orig,
        y=y_target,
        objective="traceflow",
    )
    n_valid = sum(1 for g in target_grads if g is not None)
    print(f"[eval] Target gradients: {n_valid}/{len(target_grads)} non-None.")
    print(
        f"[eval] Fixed attack state: t.shape={tuple(attack_state.t.shape)} "
        f"eps.shape={tuple(attack_state.eps.shape)} "
        f"bits={'None' if attack_state.bits is None else tuple(attack_state.bits.shape)} "
        f"y={'None' if attack_state.y is None else tuple(attack_state.y.shape)}"
    )

    # Save original grid
    _save_grid(x_orig.cpu(), img_dir / "original_grid.png", nrow=max(1, B))
    print(f"[eval] Saved: {img_dir / 'original_grid.png'}")

    has_wm = (
        watermark_modules is not None
        and watermark_modules.get("type") == "traceflow"
    )
    batch_bits = (
        attack_state.bits.to(device) if attack_state.bits is not None else None
    )

    # Shared result container
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metrics: Dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "attack": args.attack,
        "attacker": args.attacker,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": B,
        "image_size": image_size,
        "transform_type": transform_type,
        "n_target_grads_valid": n_valid,
        "source_data": source_info,
        "attacker_runs": {},
    }

    z_k_shape = (B, ae_lc, ae_ls, ae_ls)
    x_shape   = (B, 3, image_size, image_size)

    run_latent = args.attack in ("latent", "both")
    run_pixel  = args.attack in ("pixel",  "both")
    snapshot_steps = _parse_snapshot_steps(args.snapshot_steps, args.steps)
    if snapshot_steps:
        print(f"[eval] Attack snapshots: {snapshot_steps}")

    if args.attacker == "both":
        attacker_modes = ["no_key", "oracle_key"]
    else:
        attacker_modes = [args.attacker]

    for attacker in attacker_modes:
        print(f"\n[eval] ########## attacker mode: {attacker} ##########")
        run_metrics: Dict[str, Any] = {}

        # --------------------------------------------------------------
        # 9a. Latent inversion attack
        # --------------------------------------------------------------
        if run_latent:
            print(f"[eval] === Latent inversion attack | steps={args.steps} lr={args.lr} ===")
            latent_t0 = time.time()
            latent_result = latent_inversion_attack(
                model=model,
                autoencoder=autoencoder,
                watermark_modules=watermark_modules,
                target_grads=target_grads,
                z_k_shape=z_k_shape,
                attack_state=attack_state,
                steps=args.steps,
                lr=args.lr,
                device=device,
                log_interval=max(1, args.steps // 5),
                attacker=attacker,
                latent_transform=latent_transform,
                snapshot_steps=snapshot_steps,
            )
            latent_wall_time_s = time.time() - latent_t0
            z_k_dummy = latent_result["z_k_dummy"]

            # Raw no-key decode (attacker treats dummy latent as plain AE latent)
            with torch.no_grad():
                x_nokey = autoencoder.decode(z_k_dummy).cpu().clamp(-1, 1)
            _save_grid(x_nokey, img_dir / f"latent_{attacker}_raw_nokey_grid.png", nrow=max(1, B))

            # Raw defender-key decode (defender inverts the key first)
            with torch.no_grad():
                z_recovered = latent_transform.invert(z_k_dummy.to(device))
                x_defender  = autoencoder.decode(z_recovered).cpu().clamp(-1, 1)
            _save_grid(x_defender, img_dir / f"latent_{attacker}_raw_defender_grid.png", nrow=max(1, B))

            latent_snapshot_paths: Dict[str, Dict[str, str]] = {}
            for snap_step, snap_z in latent_result.get("snapshots", {}).items():
                with torch.no_grad():
                    snap_z_dev = snap_z.to(device)
                    snap_nokey = autoencoder.decode(snap_z_dev).cpu().clamp(-1, 1)
                    snap_defender = autoencoder.decode(latent_transform.invert(snap_z_dev)).cpu().clamp(-1, 1)
                nokey_path = img_dir / "attack_progress" / f"latent_{attacker}_step{snap_step:04d}_nokey.png"
                defender_path = img_dir / "attack_progress" / f"latent_{attacker}_step{snap_step:04d}_defender.png"
                _save_grid(snap_nokey, nokey_path, nrow=max(1, B))
                _save_grid(snap_defender, defender_path, nrow=max(1, B))
                latent_snapshot_paths[str(snap_step)] = {
                    "no_key": str(nokey_path),
                    "defender": str(defender_path),
                }

            lat_metrics: Dict[str, Any] = {
                "final_gml":   latent_result["final_gml"],
                "gml_history": latent_result["gml_history"],
                "snapshot_paths": latent_snapshot_paths,
                "attack_wall_time_s": latent_wall_time_s,
                "avg_attack_step_time_s": latent_wall_time_s / max(args.steps, 1),
                "cuda_max_memory_allocated_MB": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device.type == "cuda" else None
                ),
                # No-key reconstruction quality vs original
                "no_key_mse":  _mse(x_nokey, x_orig.cpu()),
                "no_key_psnr": _psnr(x_nokey, x_orig.cpu()),
                "no_key_ssim": _ssim(x_nokey, x_orig.cpu()),
                # Defender-key reconstruction quality vs original
                "defender_mse":  _mse(x_defender, x_orig.cpu()),
                "defender_psnr": _psnr(x_defender, x_orig.cpu()),
                "defender_ssim": _ssim(x_defender, x_orig.cpu()),
                **_quality_pair(x_nokey, x_orig.cpu(), "no_key"),
                **_quality_pair(x_defender, x_orig.cpu(), "defender"),
            }

            if has_wm and batch_bits is not None:
                # RAW detection (NO decoder_adapter) — the headline forensic metric
                raw_nokey = _detect_raw(
                    watermark_modules, latent_transform, autoencoder,
                    x_nokey, batch_bits, device,
                )
                raw_defender = _detect_raw(
                    watermark_modules, latent_transform, autoencoder,
                    x_defender, batch_bits, device,
                )
                lat_metrics["raw_no_key_image_bit_acc"]   = raw_nokey["image_bit_acc"]
                lat_metrics["raw_no_key_image_ber"]       = raw_nokey["image_ber"]
                lat_metrics["raw_no_key_latent_bit_acc"]  = raw_nokey["latent_bit_acc"]
                lat_metrics["raw_no_key_latent_ber"]      = raw_nokey["latent_ber"]
                lat_metrics["raw_defender_image_bit_acc"] = raw_defender["image_bit_acc"]
                lat_metrics["raw_defender_image_ber"]     = raw_defender["image_ber"]
                lat_metrics["raw_defender_latent_bit_acc"]= raw_defender["latent_bit_acc"]
                lat_metrics["raw_defender_latent_ber"]    = raw_defender["latent_ber"]
                lat_metrics["robustness"] = _evaluate_transform_robustness(
                    x_nokey,
                    prefix=f"latent_{attacker}",
                    img_dir=img_dir,
                    watermark_modules=watermark_modules,
                    latent_transform=latent_transform,
                    autoencoder=autoencoder,
                    batch_bits=batch_bits,
                    device=device,
                )

                # POST-WATERMARK sanity check (adapter explicitly applied)
                post_metrics, x_wm_defender = _detect_post_watermark(
                    watermark_modules, latent_transform, autoencoder,
                    x_defender, batch_bits, device,
                )
                lat_metrics["post_watermark_defender_image_bit_acc"]  = post_metrics["image_bit_acc"]
                lat_metrics["post_watermark_defender_image_ber"]      = post_metrics["image_ber"]
                lat_metrics["post_watermark_defender_latent_bit_acc"] = post_metrics["latent_bit_acc"]
                lat_metrics["post_watermark_defender_latent_ber"]     = post_metrics["latent_ber"]
                _save_grid(
                    x_wm_defender,
                    img_dir / f"latent_{attacker}_post_watermark_defender_grid.png",
                    nrow=max(1, B),
                )

                print(
                    f"[eval] latent[{attacker}] final_gml={latent_result['final_gml']:.6f}\n"
                    f"  RAW   no_key  img={raw_nokey['image_bit_acc']:.4f} lat={raw_nokey['latent_bit_acc']:.4f}\n"
                    f"  RAW   defndr  img={raw_defender['image_bit_acc']:.4f} lat={raw_defender['latent_bit_acc']:.4f}\n"
                    f"  ROB   jpeg    img={lat_metrics['robustness']['jpeg']['image_bit_acc']:.4f} "
                    f"lat={lat_metrics['robustness']['jpeg']['latent_bit_acc']:.4f}\n"
                    f"  POST  defndr  img={post_metrics['image_bit_acc']:.4f} lat={post_metrics['latent_bit_acc']:.4f}"
                )
            else:
                print(f"[eval] latent[{attacker}] final_gml={latent_result['final_gml']:.6f} (no watermark)")

            run_metrics["latent_attack"] = lat_metrics
            del latent_result, z_k_dummy
            _free_cuda(device)

        # --------------------------------------------------------------
        # 9b. Pixel inversion attack
        # --------------------------------------------------------------
        if run_pixel:
            print(f"[eval] === Pixel inversion attack | steps={args.steps} lr={args.lr} ===")
            pixel_t0 = time.time()
            pixel_result = pixel_inversion_attack(
                model=model,
                autoencoder=autoencoder,
                watermark_modules=watermark_modules,
                target_grads=target_grads,
                x_shape=x_shape,
                attack_state=attack_state,
                steps=args.steps,
                lr=args.lr,
                device=device,
                log_interval=max(1, args.steps // 5),
                attacker=attacker,
                latent_transform=latent_transform,
                snapshot_steps=snapshot_steps,
            )
            pixel_wall_time_s = time.time() - pixel_t0
            x_recon = pixel_result["x_dummy"].cpu().clamp(-1, 1)
            _save_grid(x_recon, img_dir / f"pixel_{attacker}_raw_recon_grid.png", nrow=max(1, B))

            pixel_snapshot_paths: Dict[str, str] = {}
            for snap_step, snap_x in pixel_result.get("snapshots", {}).items():
                snap_path = img_dir / "attack_progress" / f"pixel_{attacker}_step{snap_step:04d}.png"
                _save_grid(snap_x.clamp(-1, 1), snap_path, nrow=max(1, B))
                pixel_snapshot_paths[str(snap_step)] = str(snap_path)

            px_metrics: Dict[str, Any] = {
                "final_gml":   pixel_result["final_gml"],
                "gml_history": pixel_result["gml_history"],
                "snapshot_paths": pixel_snapshot_paths,
                "attack_wall_time_s": pixel_wall_time_s,
                "avg_attack_step_time_s": pixel_wall_time_s / max(args.steps, 1),
                "cuda_max_memory_allocated_MB": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device.type == "cuda" else None
                ),
                "mse":  _mse(x_recon, x_orig.cpu()),
                "psnr": _psnr(x_recon, x_orig.cpu()),
                "ssim": _ssim(x_recon, x_orig.cpu()),
                **_quality_pair(x_recon, x_orig.cpu(), "pixel"),
            }

            if has_wm and batch_bits is not None:
                # RAW detection (NO decoder_adapter)
                raw_pixel = _detect_raw(
                    watermark_modules, latent_transform, autoencoder,
                    x_recon, batch_bits, device,
                )
                px_metrics["raw_pixel_image_bit_acc"]   = raw_pixel["image_bit_acc"]
                px_metrics["raw_pixel_image_ber"]       = raw_pixel["image_ber"]
                px_metrics["raw_pixel_latent_bit_acc"]  = raw_pixel["latent_bit_acc"]
                px_metrics["raw_pixel_latent_ber"]      = raw_pixel["latent_ber"]
                px_metrics["robustness"] = _evaluate_transform_robustness(
                    x_recon,
                    prefix=f"pixel_{attacker}",
                    img_dir=img_dir,
                    watermark_modules=watermark_modules,
                    latent_transform=latent_transform,
                    autoencoder=autoencoder,
                    batch_bits=batch_bits,
                    device=device,
                )

                # POST-WATERMARK sanity check
                post_px, x_wm_px = _detect_post_watermark(
                    watermark_modules, latent_transform, autoencoder,
                    x_recon, batch_bits, device,
                )
                px_metrics["post_watermark_pixel_image_bit_acc"]   = post_px["image_bit_acc"]
                px_metrics["post_watermark_pixel_image_ber"]       = post_px["image_ber"]
                px_metrics["post_watermark_pixel_latent_bit_acc"]  = post_px["latent_bit_acc"]
                px_metrics["post_watermark_pixel_latent_ber"]      = post_px["latent_ber"]
                _save_grid(
                    x_wm_px,
                    img_dir / f"pixel_{attacker}_post_watermark_recon_grid.png",
                    nrow=max(1, B),
                )

                print(
                    f"[eval] pixel[{attacker}] final_gml={pixel_result['final_gml']:.6f}\n"
                    f"  RAW   pixel   img={raw_pixel['image_bit_acc']:.4f} lat={raw_pixel['latent_bit_acc']:.4f}\n"
                    f"  ROB   jpeg    img={px_metrics['robustness']['jpeg']['image_bit_acc']:.4f} "
                    f"lat={px_metrics['robustness']['jpeg']['latent_bit_acc']:.4f}\n"
                    f"  POST  pixel   img={post_px['image_bit_acc']:.4f} lat={post_px['latent_bit_acc']:.4f}"
                )
            else:
                print(f"[eval] pixel[{attacker}] final_gml={pixel_result['final_gml']:.6f} (no watermark)")

            run_metrics["pixel_attack"] = px_metrics
            del pixel_result
            _free_cuda(device)

        metrics["attacker_runs"][attacker] = run_metrics
        _free_cuda(device)

    # ------------------------------------------------------------------
    # 10. Save metrics JSON (no secret_key in output)
    # ------------------------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[eval] Metrics saved: {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_snapshot_steps(spec: str, total_steps: int) -> List[int]:
    if not spec:
        return []
    if spec.strip().lower() in {"none", "off", "false", "0"}:
        return []
    values: List[int] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        step = int(raw)
        if 0 < step <= total_steps and step not in values:
            values.append(step)
    if total_steps > 0 and total_steps not in values:
        values.append(total_steps)
    return sorted(values)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TraceFlow inversion attack evaluation harness."
    )
    p.add_argument("--config",      required=True,  help="Config YAML path.")
    p.add_argument("--checkpoint",  required=True,  help="Checkpoint .pt path.")
    p.add_argument(
        "--attack",
        default="latent",
        choices=["latent", "pixel", "both"],
        help="Which attack(s) to run (default: latent).",
    )
    p.add_argument(
        "--attacker",
        default="no_key",
        choices=["no_key", "oracle_key", "both"],
        help=(
            "Attacker knowledge for the dummy objective: 'no_key' (identity "
            "transform, realistic attacker), 'oracle_key' (real key, upper-bound "
            "oracle), or 'both' (default: no_key)."
        ),
    )
    p.add_argument("--steps",      type=int,   default=300,   help="Optimisation steps.")
    p.add_argument("--lr",         type=float, default=0.01,  help="Adam learning rate.")
    p.add_argument(
        "--snapshot-steps",
        default="10,20,30,40,50,60,70,80,90,100",
        help=(
            "Comma-separated attack steps to save as progress grids. "
            "Steps larger than --steps are ignored; the final step is always added. "
            "Use 'none' to disable."
        ),
    )
    p.add_argument("--batch-size", type=int,   default=1,
                   dest="batch_size", help="Number of images to attack (default 1).")
    p.add_argument(
        "--data-source",
        default="random",
        choices=["random", "config"],
        help=(
            "Source-image mode: 'random' preserves smoke/backward-compatible synthetic "
            "inputs; 'config' loads deterministic samples from the YAML dataset config."
        ),
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Starting dataset index for deterministic config-backed source selection.",
    )
    p.add_argument(
        "--num-source-samples",
        type=int,
        default=None,
        dest="num_source_samples",
        help="Number of source samples to load. Defaults to --batch-size.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/flow_transformer/inversion_eval",
        dest="output_dir",
        help="Directory for saved grids and metrics JSON.",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
