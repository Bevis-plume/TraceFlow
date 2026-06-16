"""src/generation/sampling.py — High-level sampling utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.generation.rectified_flow import sample_euler, sample_heun


def generate_samples(
    model: torch.nn.Module,
    autoencoder,
    latent_shape: Tuple[int, int, int, int],
    steps: int,
    device: torch.device,
    sampler: str = "euler",
    y: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Generate images from noise using the given sampler.

    Returns:
        images: (B, 3, H, W) in [-1, 1].
    """
    if sampler == "euler":
        z0 = sample_euler(
            model,
            latent_shape,
            steps,
            device,
            y,
            guidance_scale=guidance_scale,
            num_classes=num_classes,
        )
    elif sampler == "heun":
        z0 = sample_heun(
            model,
            latent_shape,
            steps,
            device,
            y,
            guidance_scale=guidance_scale,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown sampler {sampler!r}. Choose 'euler' or 'heun'.")

    images = autoencoder.decode(z0)
    return images
