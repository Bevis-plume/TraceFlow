"""
scripts/train_flow_transformer.py
===================================
Training script for the TraceFlow latent rectified flow transformer.

Usage
-----
    python -m scripts.train_flow_transformer --config configs/flow_transformer.yml
    python -m scripts.train_flow_transformer --config configs/flow_transformer.yml --smoke
    python -m scripts.train_flow_transformer --config configs/flow_transformer.yml \
        --resume checkpoints/flow_transformer/latest.pt
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
import yaml


print = functools.partial(print, flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_str)
    # Gracefully fall back when the requested accelerator is unavailable
    # (e.g. a cuda config run on a CPU/MPS-only dev box or smoke test).
    if device.type == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("[train] CUDA unavailable, falling back to mps")
            return torch.device("mps")
        print("[train] CUDA unavailable, falling back to cpu")
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("[train] MPS unavailable, falling back to cpu")
        return torch.device("cpu")
    return device


def _get_autocast_ctx(mixed_precision: str, device: torch.device):
    """Return appropriate autocast context manager.

    - CUDA: supports bf16 and fp16.
    - MPS:  autocast is supported but bf16 may be unstable; fall back to none.
    - CPU:  no autocast.
    """
    if mixed_precision == "none" or device.type in ("cpu", "mps"):
        return contextlib.nullcontext()
    if device.type == "cuda":
        dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _build_scaler(mixed_precision: str, device: torch.device):
    """Build GradScaler only for fp16 on CUDA."""
    if mixed_precision == "fp16" and device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


def _ramp(step: int, start: int, length: int) -> float:
    if length <= 0:
        return 1.0 if step + 1 >= start else 0.0
    return min(1.0, max(0.0, float(step + 1 - start) / float(length)))


def _watermark_phase_weights(wm_cfg: Dict[str, Any], step: int) -> Dict[str, float]:
    """Staged TraceFlow loss schedule.

    0-main: ramp core watermark losses without hurting flow learning.
    robust: ramp augmentation losses after the core signal appears.
    polish: ramp clean-negative/perceptual/frequency terms late in training.
    """
    main_steps = int(wm_cfg.get("schedule_main_steps", wm_cfg.get("schedule_warmup_steps", 0)) or 0)
    robust_start = int(wm_cfg.get("schedule_robust_start", main_steps) or 0)
    robust_steps = int(wm_cfg.get("schedule_robust_steps", 30000) or 0)
    polish_start = int(wm_cfg.get("schedule_polish_start", 30000) or 0)
    polish_steps = int(wm_cfg.get("schedule_polish_steps", 5000) or 0)
    main = _ramp(step, 0, main_steps) if main_steps > 0 else 1.0
    robust = _ramp(step, robust_start, robust_steps)
    polish = _ramp(step, polish_start, polish_steps)
    return {"main": main, "robust": robust, "polish": polish}


def _carrier_schedule_scale(wm_cfg: Dict[str, Any], step: int) -> float:
    """Slowly enable direct image-domain carriers.

    The carrier path is useful as a stable spread-spectrum scaffold, but if it
    is fully enabled immediately it can solve the image detector task much
    faster than the re-encoded latent detector.  Ramping it keeps the paper's
    core re-encoder/latent trace objective in the driver's seat.
    """
    if not bool(wm_cfg.get("carrier_schedule_enabled", False)):
        return 1.0
    start = int(wm_cfg.get("carrier_schedule_start", 0) or 0)
    steps = int(wm_cfg.get("carrier_schedule_steps", 1) or 1)
    min_scale = float(wm_cfg.get("carrier_schedule_min_scale", 0.0))
    if step + 1 < start:
        return min_scale
    progress = min(1.0, max(0.0, float(step + 1 - start) / float(max(steps, 1))))
    return min_scale + (1.0 - min_scale) * progress


def _clean_negative_loss(logits: torch.Tensor) -> torch.Tensor:
    """Penalize confident bit predictions on clean, unwatermarked samples."""
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, 0.5))

def _high_frequency(x: torch.Tensor) -> torch.Tensor:
    return x - F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)


def _frequency_loss(residual: torch.Tensor) -> torch.Tensor:
    """Encourage residual energy to live in high-frequency texture."""
    high = _high_frequency(residual)
    low = residual - high
    return low.pow(2).mean() / (high.pow(2).mean().detach() + 1e-6)


def _perceptual_invisibility_loss(x_w: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Lightweight perceptual proxy used during training.

    LPIPS is intentionally reserved for evaluation because running it every
    training step is expensive.  This proxy combines low-resolution structure
    and high-frequency differences, which is stable under bf16 training.
    """
    x_ref = x_ref.detach()
    scales = [1.0, 0.5, 0.25]
    losses = []
    for scale in scales:
        if scale < 1.0:
            size = (max(8, int(x_w.shape[-2] * scale)), max(8, int(x_w.shape[-1] * scale)))
            a = F.interpolate(x_w, size=size, mode="bilinear", align_corners=False)
            b = F.interpolate(x_ref, size=size, mode="bilinear", align_corners=False)
        else:
            a, b = x_w, x_ref
        losses.append(F.l1_loss(a, b))
    losses.append(F.l1_loss(_high_frequency(x_w), _high_frequency(x_ref)))
    return torch.stack(losses).mean()


def _chunked_bce_logits_loss(
    module: nn.Module,
    x: torch.Tensor,
    bits: torch.Tensor,
    loss_fn: nn.Module,
    chunk_size: int,
) -> torch.Tensor:
    """Compute BCE(logits(x), bits) in chunks to cap activation memory."""
    batch = x.size(0)
    chunk_size = max(1, int(chunk_size or batch))
    if chunk_size >= batch:
        return loss_fn(module.logits(x), bits)
    losses = []
    weights = []
    for start in range(0, batch, chunk_size):
        end = min(start + chunk_size, batch)
        losses.append(loss_fn(module.logits(x[start:end]), bits[start:end]))
        weights.append(end - start)
    total = sum(loss * weight for loss, weight in zip(losses, weights))
    return total / float(sum(weights))


def _chunked_clean_negative_loss(
    module: nn.Module,
    x: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute clean-negative loss in chunks to cap detector activations."""
    batch = x.size(0)
    chunk_size = max(1, int(chunk_size or batch))
    if chunk_size >= batch:
        return _clean_negative_loss(module.logits(x))
    losses = []
    weights = []
    for start in range(0, batch, chunk_size):
        end = min(start + chunk_size, batch)
        losses.append(_clean_negative_loss(module.logits(x[start:end])))
        weights.append(end - start)
    total = sum(loss * weight for loss, weight in zip(losses, weights))
    return total / float(sum(weights))


def _save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 4,
) -> None:
    """Save a batch of [-1, 1] images as a PNG grid."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(images.clamp(-1, 1), nrow=nrow, normalize=True, value_range=(-1, 1))
    to_pil_image(grid).save(path)


def _log_metrics(log_path: str, record: Dict[str, Any]) -> None:
    """Append a metrics record to a JSONL log file."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _save_latent_trajectory_3d(trajectory: torch.Tensor, path: str) -> None:
    """Project an Euler latent trajectory to 3D with PCA and save as JSON."""
    # trajectory: (T, B, C, H, W), usually small because it is saved only at sample intervals.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    traj = trajectory.float().flatten(2)  # (T, B, D)
    T_steps, B, D = traj.shape
    flat = traj.reshape(T_steps * B, D)
    flat = flat - flat.mean(dim=0, keepdim=True)
    try:
        _u, s, vh = torch.linalg.svd(flat, full_matrices=False)
        k = min(3, vh.shape[0])
        coords = flat @ vh[:k].T
        if k < 3:
            coords = torch.cat([coords, torch.zeros(coords.shape[0], 3 - k)], dim=1)
        denom = torch.clamp((s ** 2).sum(), min=1e-12)
        explained = ((s[:k] ** 2) / denom).tolist()
    except RuntimeError:
        coords = flat[:, :3]
        if coords.shape[1] < 3:
            coords = torch.cat([coords, torch.zeros(coords.shape[0], 3 - coords.shape[1])], dim=1)
        explained = []
    coords = coords.reshape(T_steps, B, 3)
    points = []
    for t_idx in range(T_steps):
        t_value = 1.0 - (t_idx / max(T_steps - 1, 1))
        for b_idx in range(B):
            x, y, z = coords[t_idx, b_idx].tolist()
            points.append({
                "sample": b_idx,
                "step_index": t_idx,
                "t": t_value,
                "pc1": x,
                "pc2": y,
                "pc3": z,
            })
    payload = {
        "description": "PCA projection of reverse Euler latent trajectory from t=1 noise to t=0 sample.",
        "shape": list(trajectory.shape),
        "explained_variance_ratio": explained,
        "points": points,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "rms": float(x.pow(2).mean().sqrt().item()),
        "finite_fraction": float(torch.isfinite(x).float().mean().item()),
    }


def _write_generated_latent_stats(
    path: Path | str,
    *,
    z_protected: torch.Tensor,
    z_native: torch.Tensor,
    images: torch.Tensor,
    labels: Optional[torch.Tensor],
) -> None:
    payload: Dict[str, Any] = {
        "protected_latent": _tensor_stats(z_protected),
        "native_latent": _tensor_stats(z_native),
        "decoded_images": _tensor_stats(images),
    }
    if labels is not None:
        payload["class_labels"] = [int(v) for v in labels.detach().cpu().view(-1).tolist()]
    _write_json(path, payload)


@torch.no_grad()
def _write_denoise_probe(
    path: Path | str,
    *,
    model: nn.Module,
    autoencoder,
    latent_transform: nn.Module,
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    t_value: float = 0.5,
) -> None:
    """Save original / noisy-latent decode / one-step denoise decode.

    If this probe looks good while random samples look bad, the model has
    learned local denoising but the prior/sampler path is still wrong. If this
    is bad too, the flow itself has not learned the training distribution.
    """
    from src.generation.rectified_flow import interpolate

    B = x.size(0)
    z_real_k = latent_transform(autoencoder.encode(x))
    eps = torch.randn_like(z_real_k)
    t = torch.full((B,), float(t_value), device=x.device, dtype=torch.float32)
    z_t_k = interpolate(z_real_k, eps, t)
    v_pred = model(z_t_k, t, y)
    z_hat_k = z_t_k - t.view(B, 1, 1, 1) * v_pred
    x_noisy = autoencoder.decode(latent_transform.invert(z_t_k)).clamp(-1.0, 1.0)
    x_hat = autoencoder.decode(latent_transform.invert(z_hat_k)).clamp(-1.0, 1.0)
    grid = torch.cat([x.detach().cpu(), x_noisy.cpu(), x_hat.cpu()], dim=0)
    _save_image_grid(grid, str(path), nrow=B)


def _cosine_warmup_lr(optimizer, step: int, warmup_steps: int, base_lr: float) -> None:
    """Apply linear warmup to optimizer learning rate."""
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(warmup_steps, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    smoke = args.smoke

    # Apply smoke overrides
    if smoke:
        sc = cfg.get("smoke", {})
        cfg["data"]["image_size"] = sc.get("image_size", 64)
        cfg["data"]["name"] = "random"
        cfg["training"]["batch_size"] = sc.get("batch_size", 2)
        cfg["autoencoder"]["backend"] = "local"
        cfg["autoencoder"]["pretrained_model_name_or_path"] = None
        cfg["autoencoder"]["scaling_factor"] = 1.0
        cfg["autoencoder"]["latent_size"] = sc.get("latent_size", 8)
        cfg["autoencoder"]["latent_channels"] = sc.get("latent_channels", 4)
        cfg["model"]["latent_size"] = sc.get("latent_size", 8)
        cfg["model"]["latent_channels"] = sc.get("latent_channels", 4)
        cfg["model"]["preset"] = sc.get("model_preset", "DiT-XS")
        cfg["sampling"]["steps"] = sc.get("steps", 2)
        cfg["training"]["num_steps"] = 3
        cfg["training"]["log_interval"] = 1
        cfg["training"]["sample_interval"] = 1
        cfg["training"]["save_interval"] = 3
        cfg["training"]["warmup_steps"] = 0
        cfg["training"]["grad_accum_steps"] = 1
        cfg["training"]["mixed_precision"] = "none"
        print("[smoke] Smoke mode enabled — using random data, small model, 3 steps.")

    # ------------------------------------------------------------------
    # 2. Setup
    # ------------------------------------------------------------------
    from src.utils.seed import seed_everything
    seed_everything(cfg["project"].get("seed", 42))

    device = _resolve_device(cfg["project"].get("device", "auto"))
    print(f"[train] Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(cfg["training"].get("matmul_precision", "high"))
        print(
            "[train] CUDA performance: cudnn.benchmark=True "
            f"matmul_precision={cfg['training'].get('matmul_precision', 'high')} "
            "tf32=True"
        )

    # Determine run name and isolated per-run directories.
    # An explicit --run-name always wins.  In smoke mode the default is "smoke"
    # so that a bare ``--smoke`` run is idempotent (no timestamp sprawl).
    if args.run_name:
        run_name = args.run_name
    elif smoke:
        run_name = "smoke"
    else:
        run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    base_output = Path(cfg["project"].get("output_dir", "outputs/flow_transformer"))
    base_ckpt   = Path(cfg["training"]["checkpoint_dir"])
    output_dir     = base_output / run_name
    checkpoint_dir = base_ckpt   / run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = str(output_dir / "train_log.jsonl")
    mixed_precision = cfg["training"].get("mixed_precision", "none")

    # Persist the fully-resolved config for this run (useful for debugging / reproducibility).
    resolved_cfg_path = output_dir / "resolved_config.yml"
    with open(resolved_cfg_path, "w") as _f:
        yaml.dump(cfg, _f, default_flow_style=False, sort_keys=False)

    print(f"[train] Run name:    {run_name}")
    print(f"[train] Outputs:     {output_dir}")
    print(f"[train] Checkpoints: {checkpoint_dir}")

    # ------------------------------------------------------------------
    # 3. Dataset
    # ------------------------------------------------------------------
    from src.data.image_datasets import build_dataset, build_dataloader

    dataset = build_dataset(
        name=cfg["data"]["name"],
        root=cfg["data"].get("root", "./data"),
        image_size=cfg["data"]["image_size"],
        download=cfg["data"].get("download", False),
        smoke=smoke,
        smoke_samples=max(cfg["training"]["batch_size"] * 4, 16),
    )

    loader = build_dataloader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=0 if smoke else cfg["data"].get("num_workers", 4),
        pin_memory=False if smoke else cfg["data"].get("pin_memory", False),
        drop_last=True,
        persistent_workers=False if smoke else cfg["data"].get("persistent_workers", False),
        prefetch_factor=None if smoke else cfg["data"].get("prefetch_factor"),
    )
    print(f"[train] Dataset: {cfg['data']['name']} | {len(dataset)} samples | "
          f"batch_size={cfg['training']['batch_size']}")

    # Fixed real examples for denoise probes. Keeping them on CPU avoids
    # pinning GPU memory for the whole run; they are moved to device only when
    # a sample interval fires.
    _diag_batch = next(iter(loader))
    diagnostic_x = _diag_batch[0][:min(4, cfg["training"]["batch_size"])].detach().cpu()
    diagnostic_y = None
    if m_cfg := cfg.get("model", {}):
        if m_cfg.get("class_conditional", False) and len(_diag_batch) > 1:
            diagnostic_y = _diag_batch[1][:diagnostic_x.size(0)].detach().cpu().long()

    # ------------------------------------------------------------------
    # 4. Autoencoder
    # ------------------------------------------------------------------
    from src.models.autoencoder_backend import AutoencoderBackend

    ae_cfg = cfg["autoencoder"]
    autoencoder = AutoencoderBackend(
        backend=ae_cfg.get("backend", "local"),
        pretrained_model_name_or_path=ae_cfg.get("pretrained_model_name_or_path"),
        latent_channels=ae_cfg["latent_channels"],
        image_size=cfg["data"]["image_size"],
        latent_size=ae_cfg["latent_size"],
        scaling_factor=ae_cfg.get("scaling_factor", 1.0),
        freeze=ae_cfg.get("freeze", True),
        base_channels=ae_cfg.get("base_channels", 64),
        checkpoint_path=None if smoke else ae_cfg.get("checkpoint_path"),
        require_latent_stats=False if smoke else bool(ae_cfg.get("require_latent_stats", False)),
    ).to(device)
    _ae_ckpt = None if smoke else ae_cfg.get("checkpoint_path")
    if ae_cfg.get("backend", "local") == "local" and not _ae_ckpt and not smoke:
        print(
            "[train] WARNING: local autoencoder backend with no "
            "autoencoder.checkpoint_path. Training the flow model against a "
            "randomly initialised AE will produce poor samples. Pretrain it with "
            "`python -m scripts.traceflow train-autoencoder` first."
        )
    print(f"[train] Autoencoder backend: {ae_cfg.get('backend', 'local')} | "
          f"latent {ae_cfg['latent_channels']}x{ae_cfg['latent_size']}x{ae_cfg['latent_size']}"
          + (f" | loaded checkpoint {_ae_ckpt}" if _ae_ckpt else ""))
    if hasattr(autoencoder, "latent_stats_enabled") and autoencoder.latent_stats_enabled():
        stats = autoencoder.latent_stats_metadata()
        avg_std = sum(stats.get("std", [1.0])) / max(len(stats.get("std", [])), 1)
        print(f"[train] AE latent normalization: enabled ({stats.get('type', 'unknown')}, avg_std={avg_std:.4f})")
    elif ae_cfg.get("backend", "local") == "local" and not smoke:
        print("[train] WARNING: AE latent normalization is disabled; generation quality may collapse.")

    # Sanity-check: one encode/decode pass
    with torch.no_grad():
        _dummy_x = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"], device=device)
        _dummy_z = autoencoder.encode(_dummy_x)
        _dummy_recon = autoencoder.decode(_dummy_z)
        print(f"[train] AE sanity: input {tuple(_dummy_x.shape)} -> latent {tuple(_dummy_z.shape)} "
              f"-> recon {tuple(_dummy_recon.shape)}")
        _recon_img_path = str(output_dir / "ae_recon_sanity.png")
        _save_image_grid(torch.cat([_dummy_x.cpu(), _dummy_recon.cpu()], dim=0), _recon_img_path, nrow=2)
        del _dummy_x, _dummy_z, _dummy_recon

    # ------------------------------------------------------------------
    # 5. Latent transform (factory-driven: identity | keyed)
    # ------------------------------------------------------------------
    sec_cfg = cfg.get("security", {})

    from src.security.factory import build_latent_transform
    latent_transform = build_latent_transform(
        sec_cfg,
        latent_channels=ae_cfg["latent_channels"],
        latent_size=ae_cfg["latent_size"],
    ).to(device)

    lt_kind = sec_cfg.get("latent_transform", {}).get("type", "identity")
    print(f"[train] Latent transform: {lt_kind}")

    # ------------------------------------------------------------------
    # 6. FlowTransformer
    # ------------------------------------------------------------------
    from src.models.flow_transformer import build_flow_transformer

    m_cfg = cfg["model"]
    # When the config carries fully-resolved architecture dims, they are
    # authoritative: ignore any `preset` so it does not silently override the
    # explicit hidden_size/depth/num_heads (a preset like DiT-S would otherwise
    # rebuild a different model than the config specifies, and a different one
    # than the eval/sample loaders reconstruct — causing checkpoint mismatches).
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
        dropout=m_cfg.get("dropout", 0.0),
        class_conditional=m_cfg.get("class_conditional", False),
        num_classes=m_cfg.get("num_classes"),
        time_scale=m_cfg.get("time_scale", 1.0),
    ).to(device)
    print(f"[train] FlowTransformer arch: hidden={m_cfg.get('hidden_size', 512)} "
          f"depth={m_cfg.get('depth', 12)} heads={m_cfg.get('num_heads', 8)} "
          f"patch={m_cfg.get('patch_size', 2)} time_scale={m_cfg.get('time_scale', 1.0)} "
          f"(preset={'ignored (resolved dims)' if _has_resolved else _preset})")

    if cfg["training"].get("torch_compile", False) and device.type == "cuda":
        print("[train] torch.compile: enabled")
        model = torch.compile(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] FlowTransformer params: {num_params / 1e6:.2f}M")

    # ------------------------------------------------------------------
    # 6b. TraceFlow watermark modules (disabled | final TraceFlow)
    # ------------------------------------------------------------------
    from src.watermarking.factory import build_watermark_modules
    from src.watermarking.message import expand_bits, generate_random_batch_bits
    from src.watermarking.metrics import bit_accuracy, bit_error_rate, image_delta_mse

    watermark = build_watermark_modules(cfg, image_size=cfg["data"]["image_size"], device=device)
    if watermark is not None:
        wm_cfg = watermark["config"]
        wm_type = watermark["type"]
        decoder_adapter = watermark["decoder_adapter"]
        extractor = watermark["extractor"]
        latent_detector = watermark["latent_detector"]
        wm_bits = watermark["bits"]
        bce_loss = nn.BCEWithLogitsLoss()
        print(
            f"[train] Watermark: ENABLED | type={wm_type} bits={wm_cfg['bit_length']} "
            f"alpha={wm_cfg['alpha']} lambda_wm_img={wm_cfg['lambda_wm_img']} "
            f"lambda_wm_latent={wm_cfg['lambda_wm_latent']} "
            f"lambda_img={wm_cfg['lambda_img']} lambda_cycle={wm_cfg['lambda_cycle']} "
            f"lambda_residual={wm_cfg['lambda_residual']}"
        )
    else:
        wm_type = None
        decoder_adapter = None
        latent_detector = None
        extractor = None
        wm_bits = None
        print("[train] Watermark: disabled")

    # EMA
    from src.utils.checkpoint import EMAModel, save_checkpoint
    ema = EMAModel(model, decay=cfg["training"].get("ema_decay", 0.9999))

    # ------------------------------------------------------------------
    # 7. Optimizer
    # ------------------------------------------------------------------
    # FlowTransformer params plus final TraceFlow detector/adapter params.
    # Autoencoder stays frozen.
    opt_params = list(model.parameters())
    if watermark is not None:
        opt_params += list(extractor.parameters())
        opt_params += list(decoder_adapter.parameters())
        opt_params += list(latent_detector.parameters())

    adamw_kwargs = {
        "lr": cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"].get("weight_decay", 0.01),
    }
    if device.type == "cuda" and cfg["training"].get("fused_optimizer", True):
        adamw_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(opt_params, **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(opt_params, **adamw_kwargs)
    scaler = _build_scaler(mixed_precision, device)

    # ------------------------------------------------------------------
    # 8. Resume checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])

        if watermark is not None:
            wm_state = state.get("watermark")
            if not wm_state:
                raise RuntimeError(
                    "Resume checkpoint does not contain TraceFlow watermark module states. "
                    "Refusing to continue because this would randomly reinitialize the "
                    "decoder adapter / image detector / latent detector."
                )
            missing, unexpected = extractor.load_state_dict(wm_state["extractor"], strict=False)
            allowed_missing = [
                k for k in missing
                if k.startswith("carrier_norm.") or k.startswith("carrier_head.")
            ]
            disallowed_missing = [k for k in missing if k not in allowed_missing]
            if disallowed_missing or unexpected:
                raise RuntimeError(
                    "Watermark image detector checkpoint mismatch: "
                    f"missing={disallowed_missing}, unexpected={unexpected}"
                )
            if allowed_missing:
                print(
                    "[train] Initialized new image detector carrier parameters "
                    f"not present in checkpoint: {allowed_missing}"
                )
            missing, unexpected = decoder_adapter.load_state_dict(wm_state["decoder_adapter"], strict=False)
            allowed_missing = [k for k in missing if k.startswith("carrier.")]
            disallowed_missing = [k for k in missing if not k.startswith("carrier.")]
            if disallowed_missing or unexpected:
                raise RuntimeError(
                    "Watermark decoder adapter checkpoint mismatch: "
                    f"missing={disallowed_missing}, unexpected={unexpected}"
                )
            if allowed_missing:
                print(
                    "[train] Initialized new decoder adapter carrier parameters "
                    f"not present in checkpoint: {allowed_missing}"
                )
            missing, unexpected = latent_detector.load_state_dict(wm_state["latent_detector"], strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Watermark latent detector checkpoint mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            print("[train] Restored TraceFlow watermark modules from checkpoint")

        if optimizer is not None and "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except ValueError as exc:
                print(f"[train] Warning: optimizer state was not restored: {exc}")
        if "ema_model" in state:
            ema.load_state_dict(state["ema_model"])
        start_step = int(state.get("step", 0))
        # Restore EMA warmup counter so a resumed run keeps the converged decay
        # instead of re-warming up from scratch.
        ema.num_updates = start_step
        print(f"[train] Resumed from {args.resume} at step {start_step} (model/EMA/watermark restored)")

    # ------------------------------------------------------------------
    # 8b. Warm-start (--init-from): initialise the flow model + EMA from a
    # previously trained checkpoint (e.g. the baseline generator) without
    # restoring optimizer state or step count. This protects image quality
    # when training the heavier full-TraceFlow watermark path from a good
    # generator rather than from scratch. Watermark modules stay fresh.
    # ------------------------------------------------------------------
    elif getattr(args, "init_from", None):
        init_state = torch.load(args.init_from, map_location=device, weights_only=True)
        if "model" not in init_state:
            raise RuntimeError(f"--init-from checkpoint has no 'model' state: {args.init_from}")
        missing, unexpected = model.load_state_dict(init_state["model"], strict=False)
        if unexpected:
            raise RuntimeError(
                f"--init-from checkpoint is incompatible with the current model "
                f"(unexpected keys: {list(unexpected)[:6]}). Ensure the generator "
                "and full-TraceFlow models share the same architecture/config."
            )
        if missing:
            print(f"[train] init-from: {len(missing)} model params kept at fresh init "
                  f"(e.g. {list(missing)[:3]})")
        if "ema_model" in init_state:
            try:
                ema.load_state_dict(init_state["ema_model"])
                # Preserve the source EMA update count when warm-starting from
                # a mature generator. Otherwise EMA warmup restarts at zero and
                # the early full-TraceFlow pilot can over-track fresh updates.
                ema.num_updates = int(init_state.get("step", 0) or 0)
            except Exception as exc:  # noqa: BLE001 - EMA shape mismatch is non-fatal
                print(f"[train] init-from: EMA not restored ({exc}); reinitialising EMA from model.")
                ema = EMAModel(model, decay=cfg["training"].get("ema_decay", 0.9999))
        print(f"[train] Warm-started flow model + EMA from {args.init_from} "
              "(optimizer/step fresh; watermark modules fresh)")

    # ------------------------------------------------------------------
    # 9. Training loop
    # ------------------------------------------------------------------
    from src.generation.rectified_flow import flow_loss_with_state

    num_steps = cfg["training"]["num_steps"]
    grad_accum = cfg["training"]["grad_accum_steps"]
    log_interval = cfg["training"]["log_interval"]
    sample_interval = cfg["training"]["sample_interval"]
    save_interval = cfg["training"]["save_interval"]
    warmup_steps = cfg["training"].get("warmup_steps", 0)
    clean_fp_interval = int(cfg["training"].get("clean_fp_interval", 500))
    flow_cfg = cfg.get("flow_matching", {}) or {}
    probe_t_values = [float(v) for v in flow_cfg.get("denoise_probe_t_values", [0.5])]
    sampling_cfg = cfg.get("sampling", {}) or {}
    guidance_scale = float(sampling_cfg.get("guidance_scale", 1.0))
    print(f"[train] Flow matching config: {flow_cfg if flow_cfg else 'velocity-only uniform-t'}")
    print(f"[train] Sampling CFG guidance_scale={guidance_scale:.3f}")

    model.train()
    autoencoder.eval()
    if watermark is not None:
        extractor.train()
        decoder_adapter.train()
        latent_detector.train()

    step = start_step
    accum_loss = 0.0
    accum_count = 0
    # Running watermark stats for logging (reset each log interval).
    accum_flow = 0.0
    accum_flow_velocity = 0.0
    accum_flow_x0 = 0.0
    accum_flow_t_mean = 0.0
    accum_flow_high_t_fraction = 0.0
    accum_wm = 0.0
    accum_img = 0.0
    accum_residual = 0.0
    accum_bitacc = 0.0
    accum_ber = 0.0
    # traceflow-specific accumulators
    accum_wm_latent = 0.0
    accum_wm_robust = 0.0
    accum_clean_negative = 0.0
    accum_perceptual = 0.0
    accum_frequency = 0.0
    accum_schedule_main = 0.0
    accum_schedule_robust = 0.0
    accum_schedule_polish = 0.0
    accum_carrier_scale = 0.0
    accum_cycle = 0.0
    accum_bitacc_latent = 0.0
    accum_ber_latent = 0.0
    # traceflow clean false-positive accumulators
    accum_clean_bitacc_img = 0.0
    accum_clean_bitacc_latent = 0.0
    accum_clean_count = 0
    t0 = time.time()

    data_iter = iter(loader)

    print(f"[train] Starting training — steps {start_step+1} to {num_steps}")

    while step < num_steps:
        # True gradient accumulation: each sub-step fetches a fresh batch from the
        # dataloader so gradients are computed on different data when grad_accum > 1.
        # With grad_accum=1 (smoke default) this is a single standard forward/backward.
        optimizer.zero_grad()
        step_loss = 0.0
        for _sub in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x, _labels = batch
            x = x.to(device, non_blocking=True)
            y = None
            if m_cfg.get("class_conditional", False):
                y = _labels.to(device, non_blocking=True).long()

            with _get_autocast_ctx(mixed_precision, device):
                with torch.no_grad():
                    z = autoencoder.encode(x)
                    z_k = latent_transform(z)

                wm_l = None
                img_l = None
                residual_l = None
                wm_latent_l = None
                wm_robust_l = None
                clean_negative_l = None
                perceptual_l = None
                frequency_l = None
                cycle_l = None
                bit_probs = None
                bit_probs_latent = None
                clean_bitacc_img_step = None
                clean_bitacc_lat_step = None

                if watermark is not None:
                    flow_state = flow_loss_with_state(model, z_k, y=y, flow_cfg=flow_cfg)
                    flow_l = flow_state["loss"]
                    z_hat_k = flow_state["z_hat"]

                    if wm_cfg.get("message_mode", "random_per_sample") == "random_per_sample":
                        bits = generate_random_batch_bits(
                            wm_cfg["bit_length"], x.size(0), device=device
                        )
                    else:
                        bits = expand_bits(wm_bits, x.size(0)).to(device)

                    z_hat = latent_transform.invert(z_hat_k)
                    x_hat = autoencoder.decode_with_grad(z_hat)

                    should_check_clean_fp = (
                        clean_fp_interval > 0
                        and (step + 1) % clean_fp_interval == 0
                        and _sub == grad_accum - 1
                    )
                    if should_check_clean_fp:
                        _owner_bits_b = expand_bits(wm_bits, x.size(0)).to(device)
                        with torch.no_grad():
                            _clean_img_probs = extractor(x_hat.detach())
                            clean_bitacc_img_step = bit_accuracy(_clean_img_probs, _owner_bits_b)
                            _z_clean_k = latent_transform(autoencoder.encode(x_hat.detach()))
                            _clean_lat_probs = latent_detector(_z_clean_k)
                            clean_bitacc_lat_step = bit_accuracy(_clean_lat_probs, _owner_bits_b)

                    # During the image-watermark bootstrap phase, keep the
                    # stochastic per-sample bit objective from pushing the flow
                    # backbone through the VAE decoder.  The adapter/detector
                    # still receive full gradients and learn a readable image
                    # carrier; after this phase the complete TraceFlow loss is
                    # restored end-to-end.
                    x_hat_for_wm = x_hat.detach() if step < int(wm_cfg.get("image_warmup_detach_until", 0)) else x_hat
                    carrier_scale = _carrier_schedule_scale(wm_cfg, step)
                    if hasattr(decoder_adapter, "carrier_strength"):
                        decoder_adapter.carrier_strength = float(wm_cfg.get("adapter_carrier_strength", 1.0)) * carrier_scale
                    if hasattr(extractor, "carrier_weight"):
                        extractor.carrier_weight = float(wm_cfg.get("detector_carrier_weight", 1.0)) * carrier_scale
                    residual = decoder_adapter(x_hat_for_wm, bits)
                    x_w = torch.clamp(x_hat_for_wm + wm_cfg["alpha"] * residual, -1.0, 1.0)
                    residual_l = residual.pow(2).mean()

                    bit_logits = extractor.logits(x_w)
                    wm_l = bce_loss(bit_logits, bits)
                    bit_probs = torch.sigmoid(bit_logits)
                    img_l = F.mse_loss(x_w, x_hat.detach())

                    z_re = autoencoder.encode_with_grad(x_w)
                    z_re_k = latent_transform(z_re)
                    bit_logits_latent = latent_detector.logits(z_re_k)
                    wm_latent_l = bce_loss(bit_logits_latent, bits)
                    bit_probs_latent = torch.sigmoid(bit_logits_latent)
                    cycle_l = F.mse_loss(z_re_k, z_hat_k.detach())

                    phase = _watermark_phase_weights(wm_cfg, step)

                    wm_robust_l = torch.zeros((), device=device, dtype=flow_l.dtype)
                    robust_weight_active = (
                        phase["robust"] > 0.0
                        and wm_cfg.get("robustness_enabled", False)
                        and wm_cfg.get("lambda_wm_robust", 0.0) > 0
                    )
                    if robust_weight_active:
                        from src.watermarking.augment import deterministic_robust_views
                        robust_losses = []
                        max_views = int(wm_cfg.get("robust_max_views", 2))
                        robust_batch_size = int(wm_cfg.get("robust_batch_size", 0) or x_w.size(0))
                        robust_batch_size = max(1, min(robust_batch_size, x_w.size(0)))
                        robust_chunk_size = int(wm_cfg.get("robust_chunk_size", 0) or robust_batch_size)
                        robust_input = x_w[:robust_batch_size]
                        robust_bits = bits[:robust_batch_size]
                        if bool(wm_cfg.get("robust_detach_input", True)):
                            robust_input = robust_input.detach()
                        robust_latent_enabled = bool(wm_cfg.get("robust_latent_enabled", False))
                        for x_view in deterministic_robust_views(robust_input)[1:1 + max(0, max_views)]:
                            robust_losses.append(
                                _chunked_bce_logits_loss(
                                    extractor,
                                    x_view,
                                    robust_bits,
                                    bce_loss,
                                    robust_chunk_size,
                                )
                            )
                            if robust_latent_enabled:
                                z_view = autoencoder.encode_with_grad(x_view)
                                z_view_k = latent_transform(z_view)
                                robust_losses.append(bce_loss(latent_detector.logits(z_view_k), robust_bits))
                        if robust_losses:
                            wm_robust_l = torch.stack(robust_losses).mean()

                    clean_negative_l = torch.zeros((), device=device, dtype=flow_l.dtype)
                    if phase["polish"] > 0.0 and wm_cfg.get("lambda_clean_negative", 0.0) > 0:
                        clean_negative_batch_size = int(wm_cfg.get("clean_negative_batch_size", 0) or x_hat.size(0))
                        clean_negative_batch_size = max(1, min(clean_negative_batch_size, x_hat.size(0)))
                        clean_negative_chunk_size = int(
                            wm_cfg.get("clean_negative_chunk_size", 0) or clean_negative_batch_size
                        )
                        clean_x = x_hat[:clean_negative_batch_size].detach()
                        clean_img_negative_l = _chunked_clean_negative_loss(
                            extractor,
                            clean_x,
                            clean_negative_chunk_size,
                        )
                        with torch.no_grad():
                            clean_z_k = latent_transform(autoencoder.encode(clean_x))
                        clean_lat_logits = latent_detector.logits(clean_z_k.detach())
                        clean_negative_l = 0.5 * (
                            clean_img_negative_l
                            + _clean_negative_loss(clean_lat_logits)
                        )

                    perceptual_l = torch.zeros((), device=device, dtype=flow_l.dtype)
                    if phase["polish"] > 0.0 and wm_cfg.get("lambda_perceptual", 0.0) > 0:
                        perceptual_l = _perceptual_invisibility_loss(x_w, x_hat)

                    frequency_l = torch.zeros((), device=device, dtype=flow_l.dtype)
                    if phase["polish"] > 0.0 and wm_cfg.get("lambda_frequency", 0.0) > 0:
                        frequency_l = _frequency_loss(residual)

                    sub_loss = (
                        flow_l
                        + phase["main"] * wm_cfg["lambda_wm_img"] * wm_l
                        + phase["main"] * wm_cfg["lambda_wm_latent"] * wm_latent_l
                        + phase["robust"] * wm_cfg.get("lambda_wm_robust", 0.0) * wm_robust_l
                        + phase["polish"] * wm_cfg.get("lambda_clean_negative", 0.0) * clean_negative_l
                        + wm_cfg["lambda_img"] * img_l
                        + wm_cfg["lambda_cycle"] * cycle_l
                        + wm_cfg["lambda_residual"] * residual_l
                        + phase["polish"] * wm_cfg.get("lambda_perceptual", 0.0) * perceptual_l
                        + phase["polish"] * wm_cfg.get("lambda_frequency", 0.0) * frequency_l
                    )
                else:
                    flow_state = flow_loss_with_state(model, z_k, y=y, flow_cfg=flow_cfg)
                    flow_l = flow_state["loss"]
                    sub_loss = flow_l

                sub_loss = sub_loss / grad_accum  # scale for correct gradient sum

            if scaler is not None:
                scaler.scale(sub_loss).backward()
            else:
                sub_loss.backward()

            step_loss += sub_loss.item() * grad_accum  # un-scale back for logging

            # Accumulate component metrics for logging.
            accum_flow += flow_l.item()
            accum_flow_velocity += float(flow_state.get("loss_velocity", flow_l).detach().item())
            accum_flow_x0 += float(flow_state.get("loss_x0", torch.zeros((), device=device)).detach().item())
            accum_flow_t_mean += float(flow_state.get("t_mean", torch.zeros((), device=device)).detach().item())
            accum_flow_high_t_fraction += float(flow_state.get("high_t_fraction", torch.zeros((), device=device)).detach().item())
            if watermark is not None:
                accum_wm += wm_l.item()
                accum_img += img_l.item()
                accum_residual += residual_l.item()
                accum_wm_robust += wm_robust_l.item()
                accum_clean_negative += clean_negative_l.item()
                accum_perceptual += perceptual_l.item()
                accum_frequency += frequency_l.item()
                accum_schedule_main += phase["main"]
                accum_schedule_robust += phase["robust"]
                accum_schedule_polish += phase["polish"]
                accum_carrier_scale += carrier_scale
                with torch.no_grad():
                    accum_bitacc += bit_accuracy(bit_probs, bits)
                    accum_ber += bit_error_rate(bit_probs, bits)
                    accum_bitacc_latent += bit_accuracy(bit_probs_latent, bits)
                    accum_ber_latent += bit_error_rate(bit_probs_latent, bits)
                accum_wm_latent += wm_latent_l.item()
                accum_cycle += cycle_l.item()
                if clean_bitacc_img_step is not None:
                    accum_clean_bitacc_img += clean_bitacc_img_step
                    accum_clean_bitacc_latent += clean_bitacc_lat_step
                    accum_clean_count += 1

        accum_loss += step_loss / grad_accum  # average loss for this optimizer step
        accum_count += 1

        # LR warmup
        if warmup_steps > 0:
            _cosine_warmup_lr(optimizer, step, warmup_steps, cfg["training"]["learning_rate"])

        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(opt_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(opt_params, 1.0)
            optimizer.step()

        ema.update(model)
        step += 1

        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = accum_loss / max(accum_count, 1)
            n_sub = max(accum_count, 1) * grad_accum
            record = {
                "step": step,
                "loss": avg_loss,
                "elapsed_s": elapsed,
                "learning_rate": optimizer.param_groups[0].get("lr"),
            }
            record["loss_flow"] = accum_flow / n_sub
            record["loss_flow_velocity"] = accum_flow_velocity / n_sub
            record["loss_flow_x0"] = accum_flow_x0 / n_sub
            record["flow_t_mean"] = accum_flow_t_mean / n_sub
            record["flow_high_t_fraction"] = accum_flow_high_t_fraction / n_sub
            if watermark is not None:
                record["loss_wm_img"] = accum_wm / n_sub
                record["loss_wm_latent"] = accum_wm_latent / n_sub
                record["loss_wm_robust"] = accum_wm_robust / n_sub
                record["loss_clean_negative"] = accum_clean_negative / n_sub
                record["loss_perceptual"] = accum_perceptual / n_sub
                record["loss_frequency"] = accum_frequency / n_sub
                record["wm_schedule_main"] = accum_schedule_main / n_sub
                record["wm_schedule_robust"] = accum_schedule_robust / n_sub
                record["wm_schedule_polish"] = accum_schedule_polish / n_sub
                record["wm_schedule"] = record["wm_schedule_main"]
                record["wm_carrier_scale"] = accum_carrier_scale / n_sub
                record["loss_img"] = accum_img / n_sub
                record["loss_cycle"] = accum_cycle / n_sub
                record["loss_residual"] = accum_residual / n_sub
                record["bit_acc_img"] = accum_bitacc / n_sub
                record["ber_img"] = accum_ber / n_sub
                record["bit_acc_latent"] = accum_bitacc_latent / n_sub
                record["ber_latent"] = accum_ber_latent / n_sub
                if accum_clean_count > 0:
                    record["clean_false_positive_img"] = accum_clean_bitacc_img / accum_clean_count
                    record["clean_false_positive_latent"] = accum_clean_bitacc_latent / accum_clean_count
            _log_metrics(log_path, record)
            if watermark is not None:
                print(
                    f"[train] step={step:6d}  loss={avg_loss:.4f}  "
                    f"flow={record['loss_flow']:.4f}  "
                    f"v={record['loss_flow_velocity']:.4f}  x0={record['loss_flow_x0']:.4f}  "
                    f"t={record['flow_t_mean']:.3f}  high_t={record['flow_high_t_fraction']:.3f}  "
                    f"wm_img={record['loss_wm_img']:.4f}  wm_lat={record['loss_wm_latent']:.4f}  "
                    f"carrier={record['wm_carrier_scale']:.3f}  "
                    f"wm_rob={record['loss_wm_robust']:.4f}  clean_neg={record['loss_clean_negative']:.4f}  "
                    f"perc={record['loss_perceptual']:.4f}  freq={record['loss_frequency']:.4f}  "
                    f"img={record['loss_img']:.4f}  cycle={record['loss_cycle']:.4f}  "
                    f"res={record['loss_residual']:.4f}  "
                    f"acc_img={record['bit_acc_img']:.3f}  acc_lat={record['bit_acc_latent']:.3f}  "
                    f"clean_fp_img={record.get('clean_false_positive_img', float('nan')):.3f}  "
                    f"clean_fp_lat={record.get('clean_false_positive_latent', float('nan')):.3f}  "
                    f"elapsed={elapsed:.1f}s"
                )
            else:
                print(
                    f"[train] step={step:6d}  loss={avg_loss:.4f}  "
                    f"flow={record['loss_flow']:.4f}  "
                    f"v={record['loss_flow_velocity']:.4f}  x0={record['loss_flow_x0']:.4f}  "
                    f"t={record['flow_t_mean']:.3f}  high_t={record['flow_high_t_fraction']:.3f}  "
                    f"elapsed={elapsed:.1f}s"
                )
            accum_loss = 0.0
            accum_count = 0
            accum_flow = 0.0
            accum_flow_velocity = 0.0
            accum_flow_x0 = 0.0
            accum_flow_t_mean = 0.0
            accum_flow_high_t_fraction = 0.0
            accum_wm = 0.0
            accum_img = 0.0
            accum_residual = 0.0
            accum_bitacc = 0.0
            accum_ber = 0.0
            accum_wm_latent = 0.0
            accum_wm_robust = 0.0
            accum_clean_negative = 0.0
            accum_perceptual = 0.0
            accum_frequency = 0.0
            accum_schedule_main = 0.0
            accum_schedule_robust = 0.0
            accum_schedule_polish = 0.0
            accum_carrier_scale = 0.0
            accum_cycle = 0.0
            accum_bitacc_latent = 0.0
            accum_ber_latent = 0.0
            accum_clean_bitacc_img = 0.0
            accum_clean_bitacc_latent = 0.0
            accum_clean_count = 0

        # Sample grid
        if step % sample_interval == 0:
            model.eval()
            with ema.average_parameters(model), torch.no_grad():
                from src.generation.rectified_flow import sample_euler_trajectory
                latent_shape = (
                    min(4, cfg["training"]["batch_size"]),
                    ae_cfg["latent_channels"],
                    ae_cfg["latent_size"],
                    ae_cfg["latent_size"],
                )
                y_sample = None
                sample_num_classes = None
                if m_cfg.get("class_conditional", False):
                    sample_num_classes = int(m_cfg.get("num_classes") or 1000)
                    y_sample = torch.randint(sample_num_classes, (latent_shape[0],), device=device)
                z0_k, z_traj_k = sample_euler_trajectory(
                    model,
                    latent_shape,
                    cfg["sampling"]["steps"],
                    device,
                    y=y_sample,
                    guidance_scale=guidance_scale,
                    num_classes=sample_num_classes,
                )
                # Invert key transform before decoding so samples look meaningful.
                z0 = latent_transform.invert(z0_k)
                images = autoencoder.decode(z0)
                diag_x = diagnostic_x.to(device, non_blocking=True)
                diag_y = diagnostic_y.to(device, non_blocking=True) if diagnostic_y is not None else None
            sample_path = str(output_dir / f"samples_step{step:06d}.png")
            _save_image_grid(images.cpu(), sample_path, nrow=2)
            stats_path = output_dir / f"generated_latent_stats_step{step:06d}.json"
            _write_generated_latent_stats(
                stats_path,
                z_protected=z0_k,
                z_native=z0,
                images=images,
                labels=y_sample,
            )
            probe_paths = []
            with ema.average_parameters(model), torch.no_grad():
                for probe_t in probe_t_values:
                    t_tag = int(round(float(probe_t) * 100))
                    probe_path = output_dir / f"denoise_probe_t{t_tag:03d}_step{step:06d}.png"
                    _write_denoise_probe(
                        probe_path,
                        model=model,
                        autoencoder=autoencoder,
                        latent_transform=latent_transform,
                        x=diag_x,
                        y=diag_y,
                        t_value=float(probe_t),
                    )
                    probe_paths.append(probe_path)
                if probe_t_values:
                    legacy_probe_path = output_dir / f"denoise_probe_step{step:06d}.png"
                    _write_denoise_probe(
                        legacy_probe_path,
                        model=model,
                        autoencoder=autoencoder,
                        latent_transform=latent_transform,
                        x=diag_x,
                        y=diag_y,
                        t_value=float(probe_t_values[0]),
                    )
                    probe_paths.append(legacy_probe_path)
            traj_path = str(output_dir / f"latent_trajectory_step{step:06d}.json")
            _save_latent_trajectory_3d(z_traj_k, traj_path)
            print(f"[train] Saved sample grid: {sample_path}")
            print(f"[train] Saved generated latent stats: {stats_path}")
            for probe_path in probe_paths:
                print(f"[train] Saved denoise probe: {probe_path}")
            print(f"[train] Saved latent trajectory: {traj_path}")

            if watermark is not None:
                extractor.eval()
                decoder_adapter.eval()
                latent_detector.eval()
                with torch.no_grad():
                    wm_batch_bits = expand_bits(wm_bits, images.size(0)).to(device)
                    residual = decoder_adapter(images, wm_batch_bits)
                    images_w = torch.clamp(
                        images + wm_cfg["alpha"] * residual, -1.0, 1.0
                    )
                wm_sample_path = str(output_dir / f"samples_step{step:06d}_watermarked.png")
                _save_image_grid(images_w.cpu(), wm_sample_path, nrow=2)
                print(f"[train] Saved watermarked sample grid: {wm_sample_path}")
                extractor.train()
                decoder_adapter.train()
                latent_detector.train()
            model.train()

        # Checkpoint
        if step % save_interval == 0 or step == num_steps:
            ckpt_path = str(checkpoint_dir / f"step_{step:06d}.pt")
            latest_path = str(checkpoint_dir / "latest.pt")

            ema_state = ema.state_dict()

            # model.config stores the fully resolved architecture (set in FlowTransformer.__init__).
            model_cfg_snapshot = dict(model.config)
            model_cfg_snapshot["preset"] = m_cfg.get("preset")  # keep preset name for reference

            ae_cfg_snapshot = {
                "backend": ae_cfg.get("backend", "local"),
                "kind": ae_cfg.get("kind", "deterministic"),
                "pretrained_model_name_or_path": ae_cfg.get("pretrained_model_name_or_path"),
                "checkpoint_path": ae_cfg.get("checkpoint_path"),
                "latent_channels": ae_cfg["latent_channels"],
                "image_size": cfg["data"]["image_size"],
                "latent_size": ae_cfg["latent_size"],
                "scaling_factor": ae_cfg.get("scaling_factor", 1.0),
                "base_channels": ae_cfg.get("base_channels", 64),
                "require_latent_stats": bool(ae_cfg.get("require_latent_stats", False)),
                "require_prior_diagnostics": bool(ae_cfg.get("require_prior_diagnostics", False)),
                "latent_stats": autoencoder.latent_stats_metadata() if hasattr(autoencoder, "latent_stats_metadata") else {},
            }
            training_cfg_snapshot = {
                "run_name": run_name,
                "batch_size": cfg["training"]["batch_size"],
                "grad_accum_steps": grad_accum,
                "num_steps": num_steps,
                "learning_rate": cfg["training"]["learning_rate"],
                "mixed_precision": mixed_precision,
                "ema_decay": cfg["training"].get("ema_decay", 0.9999),
            }
            extra = {
                "ema_model": ema_state,
                "model_cfg": model_cfg_snapshot,
                "ae_cfg": ae_cfg_snapshot,
                "training_cfg": training_cfg_snapshot,
                "transform_meta": {
                    # Snapshot of transform config for informational use.
                    # secret_key is intentionally NOT saved here.
                    "type": sec_cfg.get("latent_transform", {}).get("type", "identity"),
                    "block_size": sec_cfg.get("latent_transform", {}).get("block_size", 16),
                    "block_layout": sec_cfg.get("latent_transform", {}).get("block_layout", "flat"),
                    "bias_scale": sec_cfg.get("latent_transform", {}).get("bias_scale", 0.1),
                    "latent_channels": ae_cfg["latent_channels"],
                    "latent_size": ae_cfg["latent_size"],
                },
                "run_name": run_name,
            }

            # Watermark checkpoint fields (no secret/private key is stored).
            if watermark is not None:
                wm_meta = dict(watermark["config"])
                wm_extra: Dict[str, Any] = {
                    "type": wm_type,
                    "config": wm_meta,
                    "extractor": extractor.state_dict(),
                    "decoder_adapter": decoder_adapter.state_dict(),
                    "latent_detector": latent_detector.state_dict(),
                }
                if watermark["config"]["save_bits"]:
                    wm_extra["bits"] = wm_bits.detach().cpu()
                extra["watermark"] = wm_extra

            save_checkpoint(ckpt_path, model, optimizer, step, extra=extra)
            save_checkpoint(latest_path, model, optimizer, step, extra=extra)
            print(f"[train] Saved checkpoint: {ckpt_path}")

    print(f"[train] Training complete. Steps: {num_steps}")
    print(f"[train] Latest checkpoint: {checkpoint_dir / 'latest.pt'}")
    print(f"[train] Sample outputs:    {output_dir}")

    # ------------------------------------------------------------------
    # 10. Write smoke / run report
    # ------------------------------------------------------------------
    total_train_wall_time_s = time.time() - t0
    cuda_max_memory_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )

    report = {
        "status": "ok",
        "smoke": smoke,
        "run_name": run_name,
        "steps_completed": num_steps,
        "device": str(device),
        "latent_shape": [ae_cfg["latent_channels"], ae_cfg["latent_size"], ae_cfg["latent_size"]],
        "model_params_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3),
        "total_train_wall_time_s": total_train_wall_time_s,
        "avg_step_time_s": total_train_wall_time_s / max(num_steps - start_step, 1),
        "cuda_max_memory_allocated_MB": cuda_max_memory_allocated_mb,
        "checkpoint": str(checkpoint_dir / "latest.pt"),
        "output_dir": str(output_dir),
        "resolved_config": str(resolved_cfg_path),
        "eval_weights": "ema",
        "autoencoder_kind": ae_cfg.get("kind", "deterministic"),
        "autoencoder_latent_stats": autoencoder.latent_stats_metadata() if hasattr(autoencoder, "latent_stats_metadata") else {},
    }

    # Final TraceFlow watermark evaluation.
    if watermark is not None:
        model.eval()
        extractor.eval()
        decoder_adapter.eval()
        latent_detector.eval()
        with ema.average_parameters(model), torch.no_grad():
            from src.generation.rectified_flow import sample_euler
            latent_shape = (
                min(4, cfg["training"]["batch_size"]),
                ae_cfg["latent_channels"],
                ae_cfg["latent_size"],
                ae_cfg["latent_size"],
            )
            y_sample = None
            sample_num_classes = None
            if m_cfg.get("class_conditional", False):
                sample_num_classes = int(m_cfg.get("num_classes") or 1000)
                y_sample = torch.randint(sample_num_classes, (latent_shape[0],), device=device)
            z0_k = sample_euler(
                model,
                latent_shape,
                cfg["sampling"]["steps"],
                device,
                y=y_sample,
                guidance_scale=guidance_scale,
                num_classes=sample_num_classes,
            )
            z0 = latent_transform.invert(z0_k)
            x_dec = autoencoder.decode(z0)
            eval_bits = expand_bits(wm_bits, x_dec.size(0)).to(device)
            residual = decoder_adapter(x_dec, eval_bits)
            x_w = torch.clamp(x_dec + wm_cfg["alpha"] * residual, -1.0, 1.0)
            probs = extractor(x_w)
            final_acc = bit_accuracy(probs, eval_bits)
            final_ber = bit_error_rate(probs, eval_bits)
            final_dmse = image_delta_mse(x_w, x_dec)
            z_re = autoencoder.encode(x_w)
            z_re_k = latent_transform(z_re)
            probs_latent = latent_detector(z_re_k)
            final_acc_latent = bit_accuracy(probs_latent, eval_bits)
            final_ber_latent = bit_error_rate(probs_latent, eval_bits)
            clean_img_probs = extractor(x_dec)
            final_clean_acc_img = bit_accuracy(clean_img_probs, eval_bits)
            z_clean_k = latent_transform(autoencoder.encode(x_dec))
            clean_lat_probs = latent_detector(z_clean_k)
            final_clean_acc_latent = bit_accuracy(clean_lat_probs, eval_bits)
        wm_report: Dict[str, Any] = {
            "enabled": True,
            "type": wm_type,
            "bit_length": watermark["config"]["bit_length"],
            "alpha": watermark["config"]["alpha"],
            "eval_weights": "ema",
            "generated_image_bit_acc": final_acc,
            "ber_img": final_ber,
            "image_delta_mse": final_dmse,
        }
        wm_report["generated_latent_bit_acc"] = final_acc_latent
        wm_report["ber_latent"] = final_ber_latent
        wm_report["clean_false_positive_img"] = final_clean_acc_img
        wm_report["clean_false_positive_latent"] = final_clean_acc_latent
        report["watermark"] = wm_report
        print(
            f"[train] Watermark final eval ({wm_type}): generated_image_bit_acc={final_acc:.4f} "
            f"ber_img={final_ber:.4f} image_delta_mse={final_dmse:.3e}"
            + f" generated_latent_bit_acc={final_acc_latent:.4f} ber_latent={final_ber_latent:.4f}"
            + f" clean_false_positive_img={final_clean_acc_img:.4f}"
            + f" clean_false_positive_latent={final_clean_acc_latent:.4f}"
        )

    report_path = output_dir / ("smoke_report.json" if smoke else "train_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[train] Report: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TraceFlow latent flow transformer.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument(
        "--init-from", dest="init_from", type=str, default=None,
        help="Warm-start the flow model + EMA from this checkpoint (e.g. the "
             "baseline generator) without restoring optimizer/step. Watermark "
             "modules stay freshly initialised. Ignored when --resume is set.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this run (used in output/checkpoint sub-directories). "
             "Auto-generated from timestamp if not supplied. Smoke always uses 'smoke'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
