"""
src/watermarking/latent_watermark.py
=====================================
Latent-domain detector for TraceFlow.

The detector reads message bits from protected latents produced by re-encoding a
watermarked, generated, or inverted image.  It uses a latent ResNet with spatial
attention so it can model both channel statistics and 32x32 spatial structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class _LatentResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _SEBlock(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _SpatialAttentionPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, max(16, channels // 4), 1),
            nn.SiLU(),
            nn.Conv2d(max(16, channels // 4), 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        score = torch.softmax(self.score(x).reshape(b, 1, h * w), dim=-1)
        feat = x.reshape(b, c, h * w)
        return torch.bmm(feat, score.transpose(1, 2)).squeeze(-1)


class TraceLatentDetector(nn.Module):
    """Latent ResNet that predicts watermark bits from protected latents."""

    def __init__(
        self,
        bit_length: int,
        latent_channels: int = 4,
        hidden_dim: int = 192,
        base_channels: int = 96,
        num_blocks: int = 4,
        max_channels: int = 512,
    ) -> None:
        super().__init__()
        self.bit_length = int(bit_length)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.base_channels = int(base_channels)
        self.num_blocks = int(num_blocks)
        self.max_channels = int(max_channels)

        b = self.base_channels
        c1 = min(b, self.max_channels)
        c2 = min(b * 2, self.max_channels)
        c3 = min(b * 4, self.max_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(self.latent_channels, c1, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, c1), c1),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[_LatentResidualBlock(c1) for _ in range(max(1, self.num_blocks))])
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            *[_LatentResidualBlock(c2) for _ in range(max(1, self.num_blocks // 2))],
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            *[_LatentResidualBlock(c3) for _ in range(max(1, self.num_blocks // 2))],
        )
        self.attn0 = _SpatialAttentionPool(c1)
        self.attn1 = _SpatialAttentionPool(c2)
        self.attn2 = _SpatialAttentionPool(c3)
        feature_dim = (c1 + c2 + c3) * 3
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.bit_length),
        )

    def logits(self, z_k: torch.Tensor) -> torch.Tensor:
        """Predict raw bit logits from a protected latent batch."""
        if z_k.dim() != 4:
            raise ValueError(f"z_k must be [B, C, H, W], got {tuple(z_k.shape)}.")
        h0 = self.blocks(self.stem(z_k))
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        pooled = []
        for feat, attn in ((h0, self.attn0), (h1, self.attn1), (h2, self.attn2)):
            pooled.append(F.adaptive_avg_pool2d(feat, 1).flatten(1))
            pooled.append(F.adaptive_max_pool2d(feat, 1).flatten(1))
            pooled.append(attn(feat))
        return self.head(torch.cat(pooled, dim=1))

    def forward(self, z_k: torch.Tensor) -> torch.Tensor:
        """Predict bit probabilities from a protected latent batch."""
        return torch.sigmoid(self.logits(z_k))
