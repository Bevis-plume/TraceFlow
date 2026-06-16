"""
scripts/pretrain_autoencoder.py
===============================
Pretrain (and diagnose) the project-native local autoencoder.

A randomly initialised local autoencoder destroys generation quality, because
TraceFlow trains the flow model in the AE latent space and decodes through the
(frozen) AE. This script trains the local AE on real images, saves a checkpoint
that ``AutoencoderBackend(backend="local", checkpoint_path=...)`` can load, and
writes a reconstruction diagnostic grid + quality metrics so AE quality can be
verified *before* spending compute on the generator.

Usage
-----
    # CIFAR-10 32x32 paper AE:
    python -m scripts.pretrain_autoencoder --config configs/traceflow_cifar32.yml

    # quick smoke test (random data, few steps):
    python -m scripts.pretrain_autoencoder --config configs/traceflow_cifar32.yml --smoke

    # diagnose-only from an existing checkpoint (no training):
    python -m scripts.pretrain_autoencoder --config configs/traceflow_cifar32.yml \
        --diagnose-only --checkpoint checkpoints/cifar32_ae/latest.pt

The AE checkpoint path defaults to ``autoencoder.checkpoint_path`` in the config
so that downstream training (generator / keyed / identity / full TraceFlow)
loads exactly the AE that was diagnosed here.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml


print = functools.partial(print, flush=True)


# ---------------------------------------------------------------------------
# Helpers (kept consistent with scripts/train_flow_transformer.py)
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
            print("[ae-pretrain] CUDA unavailable, falling back to mps")
            return torch.device("mps")
        print("[ae-pretrain] CUDA unavailable, falling back to cpu")
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("[ae-pretrain] MPS unavailable, falling back to cpu")
        return torch.device("cpu")
    return device


def _autocast_ctx(mixed_precision: str, device: torch.device):
    if mixed_precision == "none" or device.type in ("cpu", "mps"):
        return contextlib.nullcontext()
    if device.type == "cuda":
        dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _save_image_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, path)


def _free_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Reconstruction diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _reconstruction_metrics(autoencoder, x: torch.Tensor) -> Dict[str, float]:
    """Compute per-batch reconstruction quality metrics on images in [-1, 1]."""
    from src.utils.quality_metrics import psnr_01, ssim_01, mse_01

    x_hat = autoencoder.decode(autoencoder.encode(x)).clamp(-1.0, 1.0)
    l1 = F.l1_loss(x_hat, x).item()
    return {
        "l1": l1,
        "mse": mse_01(x_hat, x),
        "psnr": psnr_01(x_hat, x),
        "ssim": ssim_01(x_hat, x),
    }


@torch.no_grad()
def _write_recon_grid(autoencoder, x: torch.Tensor, path: str, max_images: int = 8) -> None:
    """Save a top=originals / bottom=reconstructions diagnostic grid."""
    n = min(max_images, x.size(0))
    x = x[:n]
    x_hat = autoencoder.decode(autoencoder.encode(x)).clamp(-1.0, 1.0)
    grid = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    _save_image_grid(grid, path, nrow=n)


def _kl_map(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(q(z|x) || N(0,I)) for a diagonal Gaussian posterior."""
    logvar = logvar.float().clamp(min=-20.0, max=10.0)
    mu = mu.float()
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())


def _vae_kl_losses(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (raw_kl, free_bits_kl) averaged over latent dimensions."""
    kl = _kl_map(mu, logvar)
    raw = kl.mean()
    if free_bits <= 0:
        return raw, raw
    per_dim = kl.mean(dim=(0, 2, 3))
    free = (per_dim - float(free_bits)).clamp_min(0.0).mean()
    return raw, free


@torch.no_grad()
def _denormalize_with_latent_stats(z: torch.Tensor, latent_stats: Dict[str, Any]) -> torch.Tensor:
    if not latent_stats or not latent_stats.get("enabled"):
        raise RuntimeError(
            "Flow-prior AE diagnostic requires latent_stats. Retrain or diagnose the AE "
            "with the current script so the checkpoint stores per-channel latent stats."
        )
    mean = torch.as_tensor(latent_stats.get("mean"), device=z.device, dtype=z.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(latent_stats.get("std"), device=z.device, dtype=z.dtype).view(1, -1, 1, 1)
    if mean.size(1) != z.size(1) or std.size(1) != z.size(1):
        raise RuntimeError(
            f"latent_stats channels do not match prior z: mean={mean.size(1)} "
            f"std={std.size(1)} z={z.size(1)}"
        )
    return z * std.clamp_min(1e-6) + mean


@torch.no_grad()
def _decode_prior(
    autoencoder,
    z: torch.Tensor,
    *,
    mode: str,
    latent_stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Decode standard-normal prior latents in either raw or flow-normalised mode."""
    decode = autoencoder.decode_raw if hasattr(autoencoder, "decode_raw") else autoencoder.decode
    if mode == "raw":
        z_native = z
    elif mode == "flow":
        z_native = _denormalize_with_latent_stats(z, latent_stats or {})
    else:
        raise ValueError(f"unknown prior decode mode: {mode!r}")
    return decode(z_native).clamp(-1.0, 1.0)


@torch.no_grad()
def _write_prior_grid(
    autoencoder,
    path: str,
    *,
    latent_channels: int,
    latent_size: int,
    device: torch.device,
    max_images: int = 8,
    mode: str = "raw",
    latent_stats: Optional[Dict[str, Any]] = None,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Save z~N(0,I) decoded images; returns the decoded batch."""
    if z is None:
        z = torch.randn(max_images, latent_channels, latent_size, latent_size, device=device)
    x_prior = _decode_prior(autoencoder, z, mode=mode, latent_stats=latent_stats)
    _save_image_grid(x_prior.cpu(), path, nrow=max_images)
    return x_prior


@torch.no_grad()
def _write_prior_diagnostic_grids(
    autoencoder,
    *,
    raw_path: str,
    flow_path: str,
    legacy_path: str,
    latent_channels: int,
    latent_size: int,
    device: torch.device,
    latent_stats: Dict[str, Any],
    max_images: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Write raw-prior, flow-prior, and legacy alias grids from the same noise."""
    z = torch.randn(max_images, latent_channels, latent_size, latent_size, device=device)
    x_raw = _write_prior_grid(
        autoencoder,
        raw_path,
        latent_channels=latent_channels,
        latent_size=latent_size,
        device=device,
        max_images=max_images,
        mode="raw",
        z=z,
    )
    x_flow = _write_prior_grid(
        autoencoder,
        flow_path,
        latent_channels=latent_channels,
        latent_size=latent_size,
        device=device,
        max_images=max_images,
        mode="flow",
        latent_stats=latent_stats,
        z=z,
    )
    _save_image_grid(x_flow.cpu(), legacy_path, nrow=max_images)
    return x_raw, x_flow


@torch.no_grad()
def _write_posterior_grid(autoencoder, x: torch.Tensor, path: str, max_images: int = 8) -> None:
    """Save original / posterior-sampled reconstruction grid."""
    n = min(max_images, x.size(0))
    x = x[:n]
    if hasattr(autoencoder, "sample_posterior_raw"):
        z, _, _ = autoencoder.sample_posterior_raw(x)
        x_hat = autoencoder.decode_raw(z).clamp(-1.0, 1.0)
    else:
        x_hat = autoencoder.decode(autoencoder.encode(x)).clamp(-1.0, 1.0)
    grid = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    _save_image_grid(grid, path, nrow=n)


@torch.no_grad()
def _vae_diagnostics(
    autoencoder,
    x: torch.Tensor,
    *,
    latent_channels: int,
    latent_size: int,
    device: torch.device,
    latent_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Numerical checks that distinguish reconstruction from sampleable prior."""
    if hasattr(autoencoder, "encode_stats_raw"):
        mu, logvar = autoencoder.encode_stats_raw(x)
    else:
        mu = autoencoder.encode(x)
        logvar = torch.zeros_like(mu)
    raw_kl, _ = _vae_kl_losses(mu, logvar, free_bits=0.0)
    posterior_std = torch.exp(0.5 * logvar.float().clamp(min=-20.0, max=10.0))
    z_prior = torch.randn(min(16, x.size(0)), latent_channels, latent_size, latent_size, device=device)
    x_prior_raw = _decode_prior(autoencoder, z_prior, mode="raw")
    x_prior_flow = _decode_prior(autoencoder, z_prior, mode="flow", latent_stats=latent_stats)
    return {
        "kl_loss": float(raw_kl.item()),
        "posterior_std_mean": float(posterior_std.mean().item()),
        "posterior_std_min": float(posterior_std.min().item()),
        "posterior_std_max": float(posterior_std.max().item()),
        "latent_mu_mean": float(mu.float().mean().item()),
        "latent_mu_std": float(mu.float().std(unbiased=False).item()),
        "latent_logvar_mean": float(logvar.float().mean().item()),
        "prior_raw_decode_pixel_std": float(x_prior_raw.float().std(unbiased=False).item()),
        "prior_raw_decode_pixel_mean": float(x_prior_raw.float().mean().item()),
        "prior_flow_decode_pixel_std": float(x_prior_flow.float().std(unbiased=False).item()),
        "prior_flow_decode_pixel_mean": float(x_prior_flow.float().mean().item()),
        # Backward-compatible aliases now refer to the real generator decode path.
        "prior_decode_pixel_std": float(x_prior_flow.float().std(unbiased=False).item()),
        "prior_decode_pixel_mean": float(x_prior_flow.float().mean().item()),
        "prior_sample_path": "flow_normalized_prior",
    }


@torch.no_grad()
def _latent_stats(autoencoder, loader, device: torch.device, max_batches: int = 64) -> Dict[str, Any]:
    """Estimate per-channel native latent mean/std for sampleable flow training.

    The local AE can reconstruct well while producing latents with arbitrary
    scale/offset. Rectified flow samples start from N(0,I), so downstream flow
    training must see standardised AE latents and sampling must denormalise
    before decoding. Stats are computed on raw native AE latents and stored in
    the AE checkpoint.
    """
    original_device = next(autoencoder.parameters()).device
    stats_device = torch.device("cpu") if original_device.type == "mps" else device
    if stats_device != original_device:
        autoencoder.to(stats_device)
    sums = None
    sums_sq = None
    count = 0
    batches = 0
    try:
        for batch in loader:
            x = batch[0].to(stats_device, non_blocking=True)
            if hasattr(autoencoder, "encode_raw"):
                z = autoencoder.encode_raw(x).float()
            else:
                z = autoencoder.encode(x).float()
            if not torch.isfinite(z).all():
                bad = (~torch.isfinite(z)).float().mean().item()
                raise RuntimeError(
                    f"AE produced non-finite latent values while estimating stats "
                    f"(fraction={bad:.6f}). Increase latent regularization or restart AE training."
                )
            # Do reductions on CPU. This avoids occasional MPS reduction NaNs and is
            # cheap because stats are estimated only at checkpoint/diagnostic time.
            z = z.detach().cpu().float()
            reduce_dims = (0, 2, 3)
            batch_sum = z.sum(dim=reduce_dims)
            batch_sum_sq = (z * z).sum(dim=reduce_dims)
            n = z.size(0) * z.size(2) * z.size(3)
            sums = batch_sum if sums is None else sums + batch_sum
            sums_sq = batch_sum_sq if sums_sq is None else sums_sq + batch_sum_sq
            count += int(n)
            batches += 1
            if batches >= max_batches:
                break
    finally:
        if stats_device != original_device:
            autoencoder.to(original_device)
    if count <= 1 or sums is None or sums_sq is None:
        raise RuntimeError("Could not estimate AE latent stats; loader produced no batches.")
    mean = sums / float(count)
    var = (sums_sq / float(count)) - mean.pow(2)
    std = var.clamp_min(1e-8).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError(
            "Estimated non-finite AE latent stats. The AE latent space is numerically unstable; "
            "increase autoencoder_pretrain.lambda_latent_dist and retrain the AE."
        )
    return {
        "enabled": True,
        "type": "per_channel_bhw",
        "mean": mean.detach().cpu().tolist(),
        "std": std.detach().cpu().tolist(),
        "num_values_per_channel": count,
        "num_batches": batches,
        "max_batches": max_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pretrain(args: argparse.Namespace) -> int:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    smoke = args.smoke
    ae_cfg = dict(cfg["autoencoder"])
    pre_cfg = dict(cfg.get("autoencoder_pretrain", {}) or {})
    data_cfg = dict(cfg["data"])

    # ------------------------------------------------------------------
    # Resolve hyper-parameters (config defaults, overridable by CLI)
    # ------------------------------------------------------------------
    image_size = int(data_cfg.get("image_size", 32))
    latent_channels = int(ae_cfg["latent_channels"])
    latent_size = int(ae_cfg["latent_size"])
    base_channels = int(ae_cfg.get("base_channels", 64))

    steps = int(args.steps if args.steps is not None else pre_cfg.get("num_steps", 20000))
    batch_size = int(args.batch_size if args.batch_size is not None else pre_cfg.get("batch_size", 128))
    lr = float(pre_cfg.get("learning_rate", 2.0e-4))
    weight_decay = float(pre_cfg.get("weight_decay", 0.0))
    warmup_steps = int(pre_cfg.get("warmup_steps", 500))
    ae_kind = str(ae_cfg.get("kind", pre_cfg.get("kind", "deterministic"))).lower()
    kl_weight = float(pre_cfg.get("kl_weight", 0.0))
    kl_warmup_steps = int(pre_cfg.get("kl_warmup_steps", 0) or 0)
    free_bits = float(pre_cfg.get("free_bits", 0.0) or 0.0)
    use_vae = ae_kind in {"vae", "kl", "autoencoderkl"} or kl_weight > 0.0
    mu_recon_weight = float(pre_cfg.get("mu_recon_weight", 0.25 if use_vae else 0.0) or 0.0)
    lambda_latent = float(pre_cfg.get("lambda_latent_reg", 0.0))
    lambda_latent_dist = float(pre_cfg.get("lambda_latent_dist", 0.1))
    mixed_precision = str(pre_cfg.get("mixed_precision", cfg.get("training", {}).get("mixed_precision", "bf16")))
    log_interval = int(pre_cfg.get("log_interval", 100))
    sample_interval = int(pre_cfg.get("sample_interval", 1000))
    save_interval = int(pre_cfg.get("save_interval", 2000))
    latent_stats_batches = int(pre_cfg.get("latent_stats_batches", 64))
    num_workers = int(pre_cfg.get("num_workers", data_cfg.get("num_workers", 4)))

    if smoke:
        steps = min(steps, 5)
        batch_size = min(batch_size, 4)
        warmup_steps = 0
        kl_warmup_steps = min(kl_warmup_steps, steps)
        log_interval = 1
        sample_interval = 5
        save_interval = 5
        mixed_precision = "none"
        num_workers = 0

    if ae_cfg.get("backend", "local") != "local":
        print(
            f"[ae-pretrain] WARNING: autoencoder.backend is "
            f"{ae_cfg.get('backend')!r}; pretraining only applies to the local "
            "backend. Forcing backend='local' for this run."
        )

    # ------------------------------------------------------------------
    # Output + checkpoint destinations
    # ------------------------------------------------------------------
    run_name = args.run_name or ("smoke" if smoke else "cifar32_ae")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/autoencoder") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = Path(
        args.checkpoint
        or ae_cfg.get("checkpoint_path")
        or (Path("checkpoints") / run_name / "latest.pt")
    )

    from src.utils.seed import seed_everything
    seed_everything(int(cfg.get("project", {}).get("seed", 42)))

    device = _resolve_device(cfg.get("project", {}).get("device", "auto"))
    print(f"[ae-pretrain] device={device} backend=local "
          f"latent={latent_channels}x{latent_size}x{latent_size} image_size={image_size}")
    print(
        f"[ae-pretrain] kind={'vae' if use_vae else 'deterministic'} "
        f"kl_weight={kl_weight:g} kl_warmup_steps={kl_warmup_steps} "
        f"free_bits={free_bits:g} mu_recon_weight={mu_recon_weight:g}"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    from src.data.image_datasets import build_dataset, build_dataloader

    dataset = build_dataset(
        name="random" if smoke else data_cfg.get("name", "cifar10"),
        root=data_cfg.get("root", "./data"),
        image_size=image_size,
        download=bool(data_cfg.get("download", False)),
        smoke=smoke,
        smoke_samples=max(batch_size * 4, 16),
    )
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False if smoke else bool(data_cfg.get("pin_memory", False)),
        drop_last=True,
        persistent_workers=False if smoke else bool(data_cfg.get("persistent_workers", False)),
        prefetch_factor=None if smoke else data_cfg.get("prefetch_factor"),
    )
    print(f"[ae-pretrain] dataset={('random' if smoke else data_cfg.get('name'))} "
          f"samples={len(dataset)} batch_size={batch_size}")

    # ------------------------------------------------------------------
    # Build trainable local autoencoder
    # ------------------------------------------------------------------
    from src.models.autoencoder_backend import AutoencoderBackend, save_local_autoencoder

    autoencoder = AutoencoderBackend(
        backend="local",
        latent_channels=latent_channels,
        image_size=image_size,
        latent_size=latent_size,
        freeze=False,
        base_channels=base_channels,
        require_latent_stats=False,
    ).to(device)

    if args.diagnose_only:
        if not ckpt_out.is_file():
            print(f"[ae-pretrain] ERROR: --diagnose-only set but checkpoint not found: {ckpt_out}")
            return 2
        autoencoder.load_local_checkpoint(str(ckpt_out), map_location=str(device))
        print(f"[ae-pretrain] loaded AE checkpoint for diagnosis: {ckpt_out}")

    ae_config_meta = {
        "backend": "local",
        "latent_channels": latent_channels,
        "latent_size": latent_size,
        "base_channels": base_channels,
        "image_size": image_size,
        "kind": "vae" if use_vae else "deterministic",
        "kl_weight": kl_weight,
        "kl_warmup_steps": kl_warmup_steps,
        "free_bits": free_bits,
        "mu_recon_weight": mu_recon_weight,
        "latent_normalization": "per_channel_bhw",
    }

    # ------------------------------------------------------------------
    # Diagnose-only path: write grid + metrics and exit.
    # ------------------------------------------------------------------
    eval_batch = next(iter(loader))[0].to(device)

    if args.diagnose_only:
        stats = autoencoder.latent_stats_metadata() if hasattr(autoencoder, "latent_stats_metadata") else {}
        if not stats.get("enabled"):
            stats = _latent_stats(autoencoder, loader, device, max_batches=latent_stats_batches)
        metrics = _reconstruction_metrics(autoencoder, eval_batch)
        metrics.update(
            _vae_diagnostics(
                autoencoder,
                eval_batch,
                latent_channels=latent_channels,
                latent_size=latent_size,
                device=device,
                latent_stats=stats,
            )
        )
        metrics["latent_stats"] = stats
        _write_recon_grid(autoencoder, eval_batch, str(output_dir / "ae_recon_grid.png"))
        _write_prior_diagnostic_grids(
            autoencoder,
            raw_path=str(output_dir / "ae_prior_raw_grid.png"),
            flow_path=str(output_dir / "ae_prior_flow_grid.png"),
            legacy_path=str(output_dir / "ae_prior_grid.png"),
            latent_channels=latent_channels,
            latent_size=latent_size,
            device=device,
            latent_stats=stats,
        )
        _write_posterior_grid(autoencoder, eval_batch, str(output_dir / "ae_posterior_grid.png"))
        _write_metrics(output_dir, metrics, ae_config_meta, steps=0, checkpoint=str(ckpt_out), latent_stats=stats)
        print(f"[ae-pretrain] diagnosis: {json.dumps(metrics)}")
        return 0

    # ------------------------------------------------------------------
    # Optimiser + training loop
    # ------------------------------------------------------------------
    opt_kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.99)}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(autoencoder.parameters(), **opt_kwargs)

    log_path = output_dir / "ae_train_log.jsonl"
    with open(log_path, "w") as _f:
        pass

    autoencoder.train()
    data_iter = iter(loader)
    t0 = time.time()
    accum_loss = accum_rec = accum_rec_mu = accum_lat = accum_lat_dist = accum_kl = accum_kl_raw = 0.0
    n_accum = 0

    for step in range(steps):
        try:
            x = next(data_iter)[0]
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)[0]
        x = x.to(device, non_blocking=True)

        if step < warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * (step + 1) / max(warmup_steps, 1)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(mixed_precision, device):
            if use_vae:
                z, mu, logvar = autoencoder.sample_posterior_with_grad_raw(x)
            else:
                z = autoencoder.encode_with_grad_raw(x)
                mu = z
                logvar = torch.zeros_like(z)
            x_hat = autoencoder.decode_with_grad_raw(z)
            rec_loss = F.l1_loss(x_hat, x)
            rec_mu_loss = rec_loss.new_zeros(())
            if use_vae and mu_recon_weight > 0.0:
                x_mu_hat = autoencoder.decode_with_grad_raw(mu)
                rec_mu_loss = F.l1_loss(x_mu_hat, x)
            z_f = z.float()
            lat_loss = z_f.pow(2).mean()
            z_mean = z_f.mean(dim=(0, 2, 3))
            z_std = z_f.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-6)
            lat_dist_loss = z_mean.pow(2).mean() + (z_std - 1.0).pow(2).mean()
            kl_raw, kl_free = _vae_kl_losses(mu, logvar, free_bits=free_bits)
            kl_scale = 0.0
            if use_vae and kl_weight > 0.0:
                if kl_warmup_steps > 0:
                    kl_scale = min(1.0, float(step + 1) / float(kl_warmup_steps))
                else:
                    kl_scale = 1.0
            latent_dist_weight = 0.0 if use_vae else lambda_latent_dist
            loss = (
                rec_loss
                + mu_recon_weight * rec_mu_loss
                + lambda_latent * lat_loss
                + latent_dist_weight * lat_dist_loss
                + (kl_weight * kl_scale) * kl_free
            )
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
        accum_rec += rec_loss.item()
        accum_rec_mu += rec_mu_loss.item()
        accum_lat += lat_loss.item()
        accum_lat_dist += lat_dist_loss.item()
        accum_kl += kl_free.item()
        accum_kl_raw += kl_raw.item()
        n_accum += 1

        if (step + 1) % log_interval == 0 or step == 0:
            avg = accum_loss / n_accum
            record = {
                "step": step + 1,
                "loss": avg,
                "loss_recon": accum_rec / n_accum,
                "loss_recon_mu": accum_rec_mu / n_accum,
                "loss_latent": accum_lat / n_accum,
                "loss_latent_dist": accum_lat_dist / n_accum,
                "loss_kl": accum_kl / n_accum,
                "loss_kl_raw": accum_kl_raw / n_accum,
                "kl_weight_effective": kl_weight * kl_scale,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - t0, 2),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[ae-pretrain] step {step + 1}/{steps} "
                  f"loss={avg:.4f} recon={record['loss_recon']:.4f} "
                  f"recon_mu={record['loss_recon_mu']:.4f} "
                  f"kl={record['loss_kl_raw']:.4f} kl_w={record['kl_weight_effective']:.2e}")
            accum_loss = accum_rec = accum_rec_mu = accum_lat = accum_lat_dist = accum_kl = accum_kl_raw = 0.0
            n_accum = 0

        if (step + 1) % sample_interval == 0 or step == steps - 1:
            autoencoder.eval()
            sample_stats = _latent_stats(autoencoder, loader, device, max_batches=max(1, min(latent_stats_batches, 8)))
            _write_recon_grid(autoencoder, eval_batch, str(output_dir / f"ae_recon_step{step + 1:06d}.png"))
            _write_prior_diagnostic_grids(
                autoencoder,
                raw_path=str(output_dir / f"ae_prior_raw_step{step + 1:06d}.png"),
                flow_path=str(output_dir / f"ae_prior_flow_step{step + 1:06d}.png"),
                legacy_path=str(output_dir / f"ae_prior_step{step + 1:06d}.png"),
                latent_channels=latent_channels,
                latent_size=latent_size,
                device=device,
                latent_stats=sample_stats,
            )
            _write_posterior_grid(autoencoder, eval_batch, str(output_dir / f"ae_posterior_step{step + 1:06d}.png"))
            autoencoder.train()

        if (step + 1) % save_interval == 0 or step == steps - 1:
            autoencoder.eval()
            stats = _latent_stats(autoencoder, loader, device, max_batches=max(1, min(latent_stats_batches, 16)))
            metrics = _reconstruction_metrics(autoencoder, eval_batch)
            metrics.update(
                _vae_diagnostics(
                    autoencoder,
                    eval_batch,
                    latent_channels=latent_channels,
                    latent_size=latent_size,
                    device=device,
                    latent_stats=stats,
                )
            )
            metrics["latent_stats"] = stats
            save_local_autoencoder(autoencoder, str(ckpt_out), config=ae_config_meta, metrics=metrics, latent_stats=stats)
            autoencoder.train()
            print(f"[ae-pretrain] saved checkpoint -> {ckpt_out} "
                  f"(psnr={metrics['psnr']:.2f} ssim={metrics['ssim']:.3f} "
                  f"latent_std={sum(stats['std']) / len(stats['std']):.3f})")

    # ------------------------------------------------------------------
    # Final diagnostics
    # ------------------------------------------------------------------
    autoencoder.eval()
    final_stats = _latent_stats(autoencoder, loader, device, max_batches=latent_stats_batches)
    final_metrics = _reconstruction_metrics(autoencoder, eval_batch)
    final_metrics.update(
        _vae_diagnostics(
            autoencoder,
            eval_batch,
            latent_channels=latent_channels,
            latent_size=latent_size,
            device=device,
            latent_stats=final_stats,
        )
    )
    final_metrics["latent_stats"] = final_stats
    _write_recon_grid(autoencoder, eval_batch, str(output_dir / "ae_recon_grid.png"))
    _write_prior_diagnostic_grids(
        autoencoder,
        raw_path=str(output_dir / "ae_prior_raw_grid.png"),
        flow_path=str(output_dir / "ae_prior_flow_grid.png"),
        legacy_path=str(output_dir / "ae_prior_grid.png"),
        latent_channels=latent_channels,
        latent_size=latent_size,
        device=device,
        latent_stats=final_stats,
    )
    _write_posterior_grid(autoencoder, eval_batch, str(output_dir / "ae_posterior_grid.png"))
    save_local_autoencoder(autoencoder, str(ckpt_out), config=ae_config_meta, metrics=final_metrics, latent_stats=final_stats)
    _write_metrics(output_dir, final_metrics, ae_config_meta, steps=steps, checkpoint=str(ckpt_out), latent_stats=final_stats)
    _free_cuda(device)

    print(f"[ae-pretrain] done in {time.time() - t0:.1f}s")
    print(f"[ae-pretrain] final reconstruction: {json.dumps(final_metrics)}")
    print(f"[ae-pretrain] checkpoint: {ckpt_out}")
    print(f"[ae-pretrain] recon grid: {output_dir / 'ae_recon_grid.png'}")
    print(f"[ae-pretrain] flow prior grid: {output_dir / 'ae_prior_flow_grid.png'}")
    print(f"[ae-pretrain] raw prior grid: {output_dir / 'ae_prior_raw_grid.png'}")
    print(f"[ae-pretrain] prior grid alias: {output_dir / 'ae_prior_grid.png'}")
    print(f"[ae-pretrain] posterior grid: {output_dir / 'ae_posterior_grid.png'}")
    return 0


def _write_metrics(
    output_dir: Path,
    metrics: Dict[str, float],
    ae_config: Dict[str, Any],
    *,
    steps: int,
    checkpoint: str,
    latent_stats: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "steps": steps,
        "checkpoint": checkpoint,
        "autoencoder": ae_config,
        "reconstruction": metrics,
        "latent_stats": latent_stats or metrics.get("latent_stats", {}),
    }
    with open(output_dir / "ae_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain / diagnose the local TraceFlow autoencoder.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--smoke", action="store_true", help="Quick smoke test (random data, few steps).")
    p.add_argument("--run-name", default=None, help="Run name for output sub-directory.")
    p.add_argument("--output-dir", default=None, help="Directory for grids/metrics (default outputs/autoencoder/<run>).")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path to write/read (default autoencoder.checkpoint_path).")
    p.add_argument("--steps", type=int, default=None, help="Override training steps.")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument("--diagnose-only", action="store_true", help="Load checkpoint and write recon diagnostics only.")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(pretrain(_parse_args()))
