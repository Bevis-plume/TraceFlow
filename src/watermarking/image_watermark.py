"""
src/watermarking/image_watermark.py
====================================
Image-domain detector for the final TraceFlow watermark.

This is a heavy multi-scale detector: a residual CNN pyramid, SE attention,
attention pooling, and an auxiliary robustness head.  ``logits(x)`` remains the
stable public interface used by training/evaluation.
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


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = min(16, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _SEBlock(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.GroupNorm(min(16, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(out_ch) for _ in range(max(1, num_blocks))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class _AttentionPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, max(16, channels // 4), kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(max(16, channels // 4), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        weights = self.score(x).reshape(b, 1, h * w)
        weights = torch.softmax(weights, dim=-1)
        feat = x.reshape(b, c, h * w)
        return torch.bmm(feat, weights.transpose(1, 2)).squeeze(-1)


class ImageWatermarkDetector(nn.Module):
    """Heavy multi-scale CNN that predicts watermark bits from an image."""

    def __init__(
        self,
        bit_length: int,
        image_size: int,
        channels: int = 3,
        hidden_dim: int = 384,
        base_channels: int = 96,
        num_scales: int = 5,
        num_blocks: int = 3,
        max_channels: int = 768,
        carrier_grid_size: int = 32,
        carrier_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bit_length = int(bit_length)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.base_channels = int(base_channels)
        self.num_scales = int(num_scales)
        self.num_blocks = int(num_blocks)
        self.max_channels = int(max_channels)
        self.carrier_grid_size = int(carrier_grid_size)
        self.carrier_weight = float(carrier_weight)

        base = self.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.channels, base, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, base), base),
            nn.SiLU(),
            _ResidualBlock(base),
        )
        downs = []
        pools = [nn.ModuleDict({"attn": _AttentionPool(base)})]
        ch = base
        pooled_dims = [ch * 3]  # avg + max + attention
        for scale in range(max(1, self.num_scales - 1)):
            out_ch = min(base * (2 ** (scale + 1)), self.max_channels)
            downs.append(_DownBlock(ch, out_ch, num_blocks=max(1, self.num_blocks)))
            pools.append(nn.ModuleDict({"attn": _AttentionPool(out_ch)}))
            ch = out_ch
            pooled_dims.append(ch * 3)
        self.downs = nn.ModuleList(downs)
        self.pools = nn.ModuleList(pools)
        feature_dim = sum(pooled_dims)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.bit_length),
        )
        self.robust_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.bit_length),
        )
        carrier_dim = self.channels * self.carrier_grid_size * self.carrier_grid_size
        self.carrier_norm = nn.LayerNorm(carrier_dim)
        self.carrier_head = nn.Linear(carrier_dim, self.bit_length)

    def _carrier_features(self, x_w: torch.Tensor) -> torch.Tensor:
        """High-frequency image features for direct spread-spectrum decoding."""
        high = x_w - F.avg_pool2d(x_w, kernel_size=5, stride=1, padding=2)
        if high.shape[-2:] != (self.carrier_grid_size, self.carrier_grid_size):
            high = F.interpolate(
                high,
                size=(self.carrier_grid_size, self.carrier_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        return high.flatten(1)

    def _features(self, x_w: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_w)
        feats = []
        all_feats = [h]
        for down in self.downs:
            h = down(h)
            all_feats.append(h)
        for feat, pool in zip(all_feats, self.pools):
            feats.append(F.adaptive_avg_pool2d(feat, 1).flatten(1))
            feats.append(F.adaptive_max_pool2d(feat, 1).flatten(1))
            feats.append(pool["attn"](feat))
        return torch.cat(feats, dim=1)

    def logits(self, x_w: torch.Tensor) -> torch.Tensor:
        """Predict raw bit logits from an image batch in [-1, 1]."""
        if x_w.dim() != 4:
            raise ValueError(f"x_w must be [B, C, H, W], got {tuple(x_w.shape)}.")
        cnn_logits = self.head(self._features(x_w))
        carrier_feat = self.carrier_norm(self._carrier_features(x_w).float()).to(dtype=cnn_logits.dtype)
        carrier_logits = self.carrier_head(carrier_feat)
        return cnn_logits + self.carrier_weight * carrier_logits

    def robustness_logits(self, x_w: torch.Tensor) -> torch.Tensor:
        """Auxiliary robust-view bit logits; optional for training."""
        if x_w.dim() != 4:
            raise ValueError(f"x_w must be [B, C, H, W], got {tuple(x_w.shape)}.")
        cnn_logits = self.robust_head(self._features(x_w))
        carrier_feat = self.carrier_norm(self._carrier_features(x_w).float()).to(dtype=cnn_logits.dtype)
        carrier_logits = self.carrier_head(carrier_feat)
        return cnn_logits + self.carrier_weight * carrier_logits

    def forward(self, x_w: torch.Tensor) -> torch.Tensor:
        """Predict bit probabilities from an image batch in [-1, 1]."""
        return torch.sigmoid(self.logits(x_w))
