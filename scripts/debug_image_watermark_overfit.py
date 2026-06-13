"""
Quick overfit probe for the TraceFlow image watermark path.

This intentionally excludes the flow model and latent detector.  It answers one
question: can ``TraceDecoderAdapter`` place a bit-conditioned signal into decoded
images that ``ImageWatermarkDetector`` can read back?  If this tiny probe cannot
drive BCE below 0.693, the image watermark branch has a structural bug and a long
TraceFlow run should be stopped.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


print = functools.partial(print, flush=True)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _autocast(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _bit_acc(probs: torch.Tensor, bits: torch.Tensor) -> float:
    return ((probs >= 0.5).to(bits.dtype) == bits).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit TraceFlow image watermark path.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", default="bf16", choices=["none", "bf16", "fp16"])
    parser.add_argument("--output-dir", default="runs/debug_image_watermark_overfit")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(cfg.get("training", {}).get("matmul_precision", "high"))
    print(f"[debug-image-wm] device={device}")

    from src.data.image_datasets import build_dataset, build_dataloader
    from src.models.autoencoder_backend import AutoencoderBackend
    from src.watermarking.factory import build_watermark_modules
    from src.watermarking.message import generate_random_batch_bits

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    ae_cfg = cfg["autoencoder"]

    dataset = build_dataset(
        name=data_cfg["name"],
        root=data_cfg.get("root", "./data"),
        image_size=data_cfg["image_size"],
        download=data_cfg.get("download", False),
    )
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", False),
        drop_last=True,
        persistent_workers=data_cfg.get("persistent_workers", False),
        prefetch_factor=data_cfg.get("prefetch_factor"),
    )
    x, _labels = next(iter(loader))
    x = x.to(device, non_blocking=True)

    autoencoder = AutoencoderBackend(
        backend=ae_cfg.get("backend", "local"),
        pretrained_model_name_or_path=ae_cfg.get("pretrained_model_name_or_path"),
        latent_channels=ae_cfg["latent_channels"],
        image_size=data_cfg["image_size"],
        latent_size=ae_cfg["latent_size"],
        scaling_factor=ae_cfg.get("scaling_factor", 1.0),
        freeze=True,
        base_channels=ae_cfg.get("base_channels", 64),
    ).to(device).eval()

    watermark = build_watermark_modules(cfg, image_size=data_cfg["image_size"], device=device)
    if watermark is None:
        raise RuntimeError("watermark.enabled must be true for this probe")

    extractor = watermark["extractor"].train()
    decoder_adapter = watermark["decoder_adapter"].train()
    wm_cfg = watermark["config"]
    bit_length = int(wm_cfg["bit_length"])
    alpha = float(wm_cfg["alpha"])

    with torch.no_grad():
        z = autoencoder.encode(x)
        x_hat = autoencoder.decode(z).detach()

    opt = torch.optim.AdamW(
        list(extractor.parameters()) + list(decoder_adapter.parameters()),
        lr=args.lr,
        weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    bce = nn.BCEWithLogitsLoss()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(
        "[debug-image-wm] fixed batch "
        f"shape={tuple(x_hat.shape)} alpha={alpha} bit_length={bit_length} lr={args.lr}"
    )
    for step in range(1, args.steps + 1):
        bits = generate_random_batch_bits(bit_length, x_hat.size(0), device=device)
        opt.zero_grad(set_to_none=True)
        with _autocast(device, args.mixed_precision):
            residual = decoder_adapter(x_hat, bits)
            x_w = torch.clamp(x_hat + alpha * residual, -1.0, 1.0)
            logits = extractor.logits(x_w)
            loss = bce(logits, bits)
            delta_mse = F.mse_loss(x_w, x_hat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(extractor.parameters()) + list(decoder_adapter.parameters()), 1.0)
        opt.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            with torch.no_grad():
                probs = torch.sigmoid(logits.float())
                acc = _bit_acc(probs, bits)
            print(
                f"[debug-image-wm] step={step:04d} "
                f"wm_img={loss.item():.4f} acc_img={acc:.3f} delta_mse={delta_mse.item():.3e}"
            )


if __name__ == "__main__":
    main()
