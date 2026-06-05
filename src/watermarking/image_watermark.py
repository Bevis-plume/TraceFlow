"""
src/watermarking/image_watermark.py
====================================
Image-domain detector for the final TraceFlow watermark.

TraceFlow uses a learned, bit-conditioned decoder adapter to place a subtle
message-bearing signal into generated images.  ``ImageWatermarkDetector`` is the
image-domain verifier that predicts the embedded message bits from an image.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageWatermarkDetector(nn.Module):
    """Lightweight CNN that predicts per-bit probabilities from an image.

    The detector uses adaptive pooling, so the same module works for smoke
    64x64 runs and full 256x256 CUDA runs.
    """

    def __init__(
        self,
        bit_length: int,
        image_size: int,
        channels: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.bit_length = int(bit_length)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.SiLU(),
            )

        self.features = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            block(32, 64),
            block(64, 128),
            block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.bit_length),
        )

    def forward(self, x_w: torch.Tensor) -> torch.Tensor:
        """Predict bit probabilities from an image batch in ``[-1, 1]``."""
        if x_w.dim() != 4:
            raise ValueError(f"x_w must be [B, C, H, W], got {tuple(x_w.shape)}.")
        h = self.features(x_w)
        h = self.pool(h)
        logits = self.head(h)
        return torch.sigmoid(logits)
