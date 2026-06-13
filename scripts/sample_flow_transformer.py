"""
scripts/sample_flow_transformer.py
=====================================
Sampling script for the TraceFlow latent rectified flow transformer.

Usage
-----
    python -m scripts.sample_flow_transformer \
      --config configs/flow_transformer.yml \
      --checkpoint checkpoints/flow_transformer/latest.pt \
      --num-samples 16 \
      --steps 50 \
      --output-dir outputs/flow_transformer/samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def sample(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides (explicit CLI values take precedence over config defaults)
    num_samples = args.num_samples or cfg["sampling"].get("num_samples", 16)
    steps = args.steps or cfg["sampling"].get("steps", 50)
    sampler = args.sampler or cfg["sampling"].get("sampler", "euler")
    seed = args.seed
    output_dir = Path(
        args.output_dir
        or (cfg["project"].get("output_dir", "outputs/flow_transformer") + "/samples")
    )
    use_ema = not args.no_ema

    device = _resolve_device(cfg["project"].get("device", "auto"))
    print(f"[sample] Device: {device}")

    # Reproducible sampling
    if seed is not None:
        from src.utils.seed import seed_everything
        seed_everything(seed)
        print(f"[sample] Seed: {seed}")

    # ------------------------------------------------------------------
    # 2. Load checkpoint — resolve architecture from saved config
    # ------------------------------------------------------------------
    print(f"[sample] Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Use architecture that was saved at checkpoint time (overrides YAML config).
    # This lets the smoke checkpoint be loaded with the full config YAML present.
    m_cfg = state.get("model_cfg", cfg["model"])
    ae_saved = state.get("ae_cfg", None)

    # ------------------------------------------------------------------
    # Transform metadata (informational — sampling runs in protected space)
    # ------------------------------------------------------------------
    # The FlowTransformer was trained on z_k = latent_transform(z_data), i.e.
    # in the PROTECTED latent space.  Sampling generates z_0_k (protected space)
    # from random Gaussian noise z_1 using the flow ODE.
    #
    # For identity transform (baseline): ae.decode(z_0_k) gives real-looking images.
    # For keyed transform: ae.decode(z_0_k) does NOT give meaningful images because
    # z_0_k lives in the rotated/biased protected space, not the autoencoder's
    # native latent space.  To obtain a meaningful image, the defender must apply:
    #
    #     z_0 = keyed_transform.invert(z_0_k)   (requires secret_key)
    #     image = ae.decode(z_0)
    #
    # This defender-side inversion is handled by future evaluation scripts.
    # The sampling script intentionally omits it so that generated samples can
    # be shared without leaking information about the transform.
    transform_meta = state.get("transform_meta", {"type": "identity"})
    transform_type = transform_meta.get("type", "identity")

    # Build latent transform for defender-side inversion.
    # Always builds a transform so that TraceFlow latent detector metrics run for all
    # ablations (identity and keyed). Falls back to IdentityLatentTransform when
    # the checkpoint is keyed but no secret_key is present in the config.
    sec_cfg = cfg.get("security", {})
    _lt_secret = sec_cfg.get("latent_transform", {}).get("secret_key", "")
    _lt_lc = ae_saved["latent_channels"] if ae_saved is not None else cfg["autoencoder"]["latent_channels"]
    _lt_ls = ae_saved["latent_size"] if ae_saved is not None else cfg["autoencoder"]["latent_size"]
    if transform_type == "keyed" and _lt_secret:
        from src.security.factory import build_latent_transform
        latent_transform = build_latent_transform(
            sec_cfg,
            latent_channels=_lt_lc,
            latent_size=_lt_ls,
        ).to(device)
        print(f"[sample] Loaded keyed latent transform for defender-side inversion.")
    else:
        from src.security.identity_transform import IdentityLatentTransform
        latent_transform = IdentityLatentTransform().to(device)
        if transform_type == "keyed":
            print(
                f"[sample] Note: checkpoint uses keyed latent transform but no secret_key "
                f"in config — samples remain in protected latent space (identity fallback)."
            )
        else:
            print(f"[sample] Using identity latent transform.")

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
        ae_latent_channels = ae_saved["latent_channels"]
        ae_latent_size = ae_saved["latent_size"]
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
        ae_latent_channels = ae_cfg["latent_channels"]
        ae_latent_size = ae_cfg["latent_size"]

    autoencoder.eval()

    # ------------------------------------------------------------------
    # 4. FlowTransformer (built from checkpoint's saved architecture)
    # ------------------------------------------------------------------
    # Use preset=None when checkpoint stores fully resolved values (Phase 1.5+).
    # This prevents build_flow_transformer from overwriting stored values with preset defaults.
    # Fall back to preset for older checkpoints that may have unresolved field values.
    from src.models.flow_transformer import build_flow_transformer, PRESETS

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

    # ------------------------------------------------------------------
    # 5. Load weights
    # ------------------------------------------------------------------
    if use_ema and "ema_model" in state:
        print("[sample] Using EMA weights.")
        ema_state = state["ema_model"]
        model_state = model.state_dict()
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema_state:
                model_state[name] = ema_state[name].to(device)
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(state["model"])

    model.eval()
    print(f"[sample] Model loaded. Generating {num_samples} samples | "
          f"sampler={sampler} | steps={steps}")

    # Optional class conditioning, matching DiT/SiT ImageNet-style training.
    y = None
    if m_cfg.get("class_conditional", False):
        num_classes = int(m_cfg.get("num_classes") or cfg.get("model", {}).get("num_classes") or 1000)
        if args.class_id is not None:
            if args.class_id < 0 or args.class_id >= num_classes:
                raise ValueError(f"--class-id must be in [0, {num_classes - 1}], got {args.class_id}")
            y = torch.full((num_samples,), int(args.class_id), device=device, dtype=torch.long)
            print(f"[sample] Class conditioning: fixed class_id={args.class_id}")
        else:
            y = torch.randint(num_classes, (num_samples,), device=device, dtype=torch.long)
            print(f"[sample] Class conditioning: random labels in [0, {num_classes - 1}]")

    # ------------------------------------------------------------------
    # 6. Sample
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.generation.rectified_flow import sample_euler, sample_heun

    latent_shape = (num_samples, ae_latent_channels, ae_latent_size, ae_latent_size)

    with torch.no_grad():
        if sampler == "heun":
            z0_k = sample_heun(model, latent_shape, steps, device, y=y)
        else:
            z0_k = sample_euler(model, latent_shape, steps, device, y=y)

        # Defender-side inversion: z_0_k → z_0 (AE latent space).
        # For identity transform this is a no-op; for keyed it undoes the key rotation.
        z0 = latent_transform.invert(z0_k)
        if transform_type == "keyed" and _lt_secret:
            print("[sample] Applied defender-side key inversion.")

        # Decode latents to images in [-1, 1]
        images = autoencoder.decode(z0).cpu()

    # Save sample grid
    grid_path = output_dir / "sample_grid.png"
    nrow = max(1, int(num_samples ** 0.5))
    grid = vutils.make_grid(images.clamp(-1, 1), nrow=nrow, normalize=True, value_range=(-1, 1))
    to_pil_image(grid).save(str(grid_path))
    print(f"[sample] Saved grid: {grid_path}")

    # Save individual PNGs — convert [-1, 1] → [0, 1] correctly
    for i, img in enumerate(images):
        img_path = output_dir / f"sample_{i:04d}.png"
        img_norm = (img.clamp(-1, 1) + 1.0) / 2.0  # maps [-1,1] → [0,1]
        to_pil_image(img_norm).save(str(img_path))

    print(f"[sample] Saved {num_samples} individual images to {output_dir}")

    # ------------------------------------------------------------------
    # 6b. TraceFlow watermark evaluation — only if present in checkpoint
    # ------------------------------------------------------------------
    wm_state = state.get("watermark", None)
    if wm_state is not None and wm_state.get("config", {}).get("enabled", False):
        from src.watermarking.image_watermark import ImageWatermarkDetector
        from src.watermarking.decoder_watermark import TraceDecoderAdapter
        from src.watermarking.latent_watermark import TraceLatentDetector
        from src.watermarking.message import generate_watermark_bits, expand_bits
        from src.watermarking.metrics import (
            bit_accuracy,
            bit_error_rate,
            detection_passed,
            image_delta_mse,
        )
        from src.utils.quality_metrics import pair_quality

        wm_cfg = wm_state["config"]
        wm_type = wm_state.get("type", wm_cfg.get("type", "traceflow"))
        if wm_type != "traceflow":
            raise ValueError(f"Unsupported watermark checkpoint type: {wm_type!r}")
        wm_image_size = ae_saved["image_size"] if ae_saved is not None else cfg["data"]["image_size"]

        extractor = ImageWatermarkDetector(
            bit_length=wm_cfg["bit_length"],
            image_size=wm_image_size,
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
            image_size=wm_image_size,
            base_channels=wm_cfg.get("adapter_base_channels", 64),
            num_blocks=wm_cfg.get("adapter_num_blocks", 3),
            max_channels=wm_cfg.get("adapter_max_channels", 512),
        ).to(device)
        decoder_adapter.load_state_dict(wm_state["decoder_adapter"])
        decoder_adapter.eval()

        latent_det = TraceLatentDetector(
            bit_length=wm_cfg["bit_length"],
            latent_channels=ae_latent_channels,
            hidden_dim=wm_cfg.get("latent_detector_hidden_dim", 128),
            base_channels=wm_cfg.get("latent_detector_base_channels", 64),
            num_blocks=wm_cfg.get("latent_detector_num_blocks", 3),
            max_channels=wm_cfg.get("latent_detector_max_channels", 512),
        ).to(device)
        latent_det.load_state_dict(wm_state["latent_detector"])
        latent_det.eval()

        if "bits" in wm_state:
            wm_bits = wm_state["bits"].to(device).float()
        else:
            wm_bits = generate_watermark_bits(wm_cfg["bit_length"], wm_cfg["seed"], device=device)

        with torch.no_grad():
            imgs_dev = images.to(device)
            batch_bits = expand_bits(wm_bits, imgs_dev.size(0)).to(device)
            residual = decoder_adapter(imgs_dev, batch_bits)
            images_w = torch.clamp(imgs_dev + wm_cfg["alpha"] * residual, -1.0, 1.0)
            probs = extractor(images_w)
            acc = bit_accuracy(probs, batch_bits)
            ber = bit_error_rate(probs, batch_bits)
            passed = detection_passed(probs, batch_bits, wm_cfg["detection_threshold_acc"])
            dmse = image_delta_mse(images_w, imgs_dev)
            invisibility_metrics = pair_quality(images_w, imgs_dev, prefix="watermarked_vs_clean")
            z_re = autoencoder.encode(images_w)
            z_re_k = latent_transform(z_re)
            probs_latent = latent_det(z_re_k)
            acc_latent = bit_accuracy(probs_latent, batch_bits)
            ber_latent = bit_error_rate(probs_latent, batch_bits)

        images_w_cpu = images_w.cpu()
        wm_grid_path = output_dir / "sample_grid_watermarked.png"
        wm_grid = vutils.make_grid(
            images_w_cpu.clamp(-1, 1), nrow=nrow, normalize=True, value_range=(-1, 1)
        )
        to_pil_image(wm_grid).save(str(wm_grid_path))
        print(f"[sample] Saved watermarked grid: {wm_grid_path}")

        wm_metrics = {
            "type": wm_type,
            "bit_length": wm_cfg["bit_length"],
            "alpha": wm_cfg["alpha"],
            "detection_threshold_acc": wm_cfg["detection_threshold_acc"],
            "generated_image_bit_acc": acc,
            "ber_img": ber,
            "generated_latent_bit_acc": acc_latent,
            "ber_latent": ber_latent,
            "detection_passed": passed,
            "image_delta_mse": dmse,
            **invisibility_metrics,
            "num_samples": num_samples,
        }
        wm_metrics_path = output_dir / "watermark_metrics.json"
        with open(wm_metrics_path, "w") as f:
            json.dump(wm_metrics, f, indent=2)
        print(
            f"[sample] Watermark ({wm_type}): generated_image_bit_acc={acc:.4f} "
            f"ber_img={ber:.4f} generated_latent_bit_acc={acc_latent:.4f} "
            f"ber_latent={ber_latent:.4f} detection_passed={passed} "
            f"image_delta_mse={dmse:.3e}"
        )
        print(f"[sample] Watermark metrics: {wm_metrics_path}")

    # Save sampling metadata
    sampling_cfg = {
        "checkpoint": args.checkpoint,
        "sampler": sampler,
        "steps": steps,
        "num_samples": num_samples,
        "seed": seed,
        "use_ema": use_ema,
        "device": str(device),
        "latent_shape": list(latent_shape),
        "model_cfg": m_cfg,
        "transform_type": transform_type,
        "class_labels": None if y is None else y.detach().cpu().tolist(),
    }
    cfg_path = output_dir / "sampling_config.json"
    with open(cfg_path, "w") as f:
        json.dump(sampling_cfg, f, indent=2)
    print(f"[sample] Sampling config: {cfg_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained TraceFlow model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--sampler", type=str, default=None, choices=["euler", "heun"],
        help="ODE integrator. Defaults to config value or 'euler'.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible samples.")
    parser.add_argument("--class-id", type=int, default=None, help="Fixed class label for class-conditional checkpoints. Defaults to random labels.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-ema", action="store_true", help="Do not use EMA weights.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sample(args)
