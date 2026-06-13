"""
src/watermarking/augment.py
===========================
Deterministic differentiable augmentations for TraceFlow watermark training.

These views are deterministic so inversion evaluation can compute matching
real/dummy gradients without stochastic augmentation mismatch.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def _pseudo_noise_like(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Deterministic image-shaped pseudo-noise in [-scale, scale]."""
    _, _, h, w = x.shape
    yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
    pat = torch.sin(37.0 * xx + 19.0 * yy) * torch.cos(11.0 * xx - 23.0 * yy)
    return scale * pat.expand_as(x)


def jpeg_like_view(x: torch.Tensor, levels: int = 32) -> torch.Tensor:
    """Differentiable-ish JPEG proxy via quantization with straight-through gradients."""
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    q = torch.round(x01 * float(levels)) / float(levels)
    q = x01 + (q - x01).detach()
    return (q * 2.0 - 1.0).clamp(-1.0, 1.0)


def deterministic_robust_views(x: torch.Tensor) -> List[torch.Tensor]:
    """Return differentiable robustness views for images in [-1, 1]."""
    if x.dim() != 4:
        raise ValueError(f"x must be [B, C, H, W], got {tuple(x.shape)}.")
    _, _, h, w = x.shape
    views: List[torch.Tensor] = [x]

    small_h = max(8, int(round(h * 0.75)))
    small_w = max(8, int(round(w * 0.75)))
    resized = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
    resized = F.interpolate(resized, size=(h, w), mode="bilinear", align_corners=False)
    views.append(resized.clamp(-1.0, 1.0))

    blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    views.append(blurred.clamp(-1.0, 1.0))

    crop_h = max(8, int(round(h * 0.875)))
    crop_w = max(8, int(round(w * 0.875)))
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    cropped = x[:, :, top:top + crop_h, left:left + crop_w]
    cropped = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
    views.append(cropped.clamp(-1.0, 1.0))

    views.append(jpeg_like_view(x, levels=32))
    views.append((x + _pseudo_noise_like(x, scale=0.025)).clamp(-1.0, 1.0))
    return views


def robust_view_stack(x: torch.Tensor) -> torch.Tensor:
    """Concatenate deterministic robustness views along the batch dimension."""
    return torch.cat(deterministic_robust_views(x), dim=0)
