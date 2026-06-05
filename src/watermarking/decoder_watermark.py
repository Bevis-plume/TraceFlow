"""
src/watermarking/decoder_watermark.py
======================================
TraceFlow decoder adapter.

The final TraceFlow model keeps the autoencoder frozen and learns a small,
bit-conditioned residual adapter over decoded images.  This adapter is part of
the training objective, so its signal contributes to the gradients seen by a
model/gradient inversion attacker.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TraceDecoderAdapter(nn.Module):
    """Bit-conditioned residual adapter over decoded images.

    Pipeline: bits -> MLP embedding -> FiLM conditioning over a small CNN feature
    map of ``x_dec`` -> convolutional head -> tanh-limited residual.
    """

    def __init__(
        self,
        bit_length: int,
        channels: int = 3,
        hidden_dim: int = 128,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.bit_length = int(bit_length)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.image_size = int(image_size)

        feat_ch = max(32, hidden_dim // 2)
        self.feat_ch = feat_ch

        self.bit_mlp = nn.Sequential(
            nn.Linear(self.bit_length, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feat_ch),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(channels, feat_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, feat_ch), num_channels=feat_ch),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
        )
        self.post_film = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, feat_ch), num_channels=feat_ch),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(feat_ch, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_dec: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Compute a bounded, bit-conditioned residual for ``x_dec``."""
        if x_dec.dim() != 4:
            raise ValueError(f"x_dec must be [B, C, H, W], got {tuple(x_dec.shape)}.")
        if bits.dim() != 2 or bits.size(1) != self.bit_length:
            raise ValueError(
                f"bits must be [B, {self.bit_length}], got {tuple(bits.shape)}."
            )

        bits = bits.to(device=x_dec.device, dtype=x_dec.dtype)
        film = self.bit_mlp(bits)
        scale, shift = film.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.stem(x_dec)
        h = h * (1.0 + scale) + shift
        h = self.post_film(h)
        return torch.tanh(self.head(h))


class TraceDecoderWrapper(nn.Module):
    """Optional convenience wrapper around an autoencoder and TraceDecoderAdapter."""

    def __init__(
        self,
        autoencoder: nn.Module,
        adapter: TraceDecoderAdapter,
        alpha: float = 0.02,
        freeze_autoencoder: bool = True,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.adapter = adapter
        self.alpha = float(alpha)
        if freeze_autoencoder:
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(
        self,
        z: torch.Tensor,
        bits: Optional[torch.Tensor] = None,
        watermark_enabled: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_dec = self.autoencoder.decode(z)
        if not watermark_enabled or bits is None:
            return x_dec, x_dec
        residual = self.adapter(x_dec.detach(), bits)
        x_w = torch.clamp(x_dec.detach() + self.alpha * residual, -1.0, 1.0)
        return x_dec, x_w
