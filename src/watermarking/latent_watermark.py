"""
src/watermarking/latent_watermark.py
=====================================
Latent-domain detector for TraceFlow.

The detector reads the watermark message from the protected, key-transformed
latent produced by re-encoding a generated or inverted image.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TraceLatentDetector(nn.Module):
    """Lightweight CNN that predicts per-bit probabilities from protected latents."""

    def __init__(
        self,
        bit_length: int,
        latent_channels: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.bit_length = int(bit_length)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.SiLU(),
            )

        self.features = nn.Sequential(
            nn.Conv2d(self.latent_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(8, 32), num_channels=32),
            nn.SiLU(),
            block(32, 64),
            block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.bit_length),
        )

    def forward(self, z_k: torch.Tensor) -> torch.Tensor:
        """Predict bit probabilities from a protected latent batch."""
        if z_k.dim() != 4:
            raise ValueError(f"z_k must be [B, C, H, W], got {tuple(z_k.shape)}.")
        h = self.features(z_k)
        h = self.pool(h)
        logits = self.head(h)
        return torch.sigmoid(logits)
