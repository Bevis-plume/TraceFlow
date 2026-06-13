"""
src/watermarking/decoder_watermark.py
======================================
TraceFlow decoder adapter.

The final TraceFlow model keeps the autoencoder frozen and learns a
bit-conditioned residual adapter over decoded images.  The adapter is a deeper
4-level ResUNet with FiLM conditioning, squeeze/excitation attention, and a
high-frequency residual branch.  Its output remains bounded so watermark
strength is controlled by ``alpha`` in the training/evaluation code.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class _FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, bit_length: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = min(16, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.se = _SEBlock(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.bit_mlp = nn.Sequential(
            nn.Linear(bit_length, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * channels),
        )
        nn.init.zeros_(self.bit_mlp[-1].weight)
        nn.init.zeros_(self.bit_mlp[-1].bias)

    @staticmethod
    def _film(h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        params = self.bit_mlp(bits.to(device=x.device, dtype=x.dtype))
        s1, b1, s2, b2 = params.chunk(4, dim=1)
        h = self._film(self.norm1(x), s1, b1)
        h = self.conv1(F.silu(h))
        h = self.dropout(h)
        h = self._film(self.norm2(h), s2, b2)
        h = self.conv2(F.silu(h))
        h = self.se(h)
        return x + h


class _DownFiLMBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bit_length: int, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.GroupNorm(min(16, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.ModuleList([
            _FiLMResidualBlock(out_ch, bit_length, hidden_dim) for _ in range(max(1, num_blocks))
        ])

    def forward(self, x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        for block in self.blocks:
            h = block(h, bits)
        return h


class _UpFiLMBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bit_length: int, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, out_ch), out_ch),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            _FiLMResidualBlock(out_ch, bit_length, hidden_dim) for _ in range(max(1, num_blocks))
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        h = self.proj(torch.cat([h, skip], dim=1))
        for block in self.blocks:
            h = block(h, bits)
        return h


class _HighFrequencyBranch(nn.Module):
    def __init__(self, channels: int, bit_length: int, hidden_dim: int, base_channels: int) -> None:
        super().__init__()
        self.bit_proj = nn.Sequential(nn.Linear(bit_length, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, base_channels))
        self.net = nn.Sequential(
            nn.Conv2d(channels + base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high = x - low
        b = self.bit_proj(bits.to(device=x.device, dtype=x.dtype)).unsqueeze(-1).unsqueeze(-1)
        b = b.expand(-1, -1, x.shape[-2], x.shape[-1])
        return self.net(torch.cat([high, b], dim=1))


class _BitCarrierBranch(nn.Module):
    """Direct learnable spread-spectrum carrier for image-domain bits.

    The ResUNet adapter can learn a watermark from detector gradients alone, but
    in practice the image branch may stay near random because both the embedder
    and detector start uncalibrated.  This carrier gives every bit an explicit
    high-frequency basis pattern while remaining trainable and bounded by the
    global TraceFlow alpha.  It is intentionally small; the ResUNet still adapts
    the signal to image content.
    """

    def __init__(self, bit_length: int, channels: int, grid_size: int = 32) -> None:
        super().__init__()
        self.bit_length = int(bit_length)
        self.channels = int(channels)
        self.grid_size = int(grid_size)
        # A too-small carrier makes the image detector see almost no bit signal
        # at startup (alpha is applied later by the training loop).  This scale
        # is still bounded by tanh + alpha, but large enough for BCE gradients to
        # align the detector and adapter in the first few thousand steps.
        self.carriers = nn.Parameter(
            torch.randn(self.bit_length, self.channels, self.grid_size, self.grid_size) * 0.20
        )

    def forward(self, bits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        centered = bits.to(dtype=self.carriers.dtype) * 2.0 - 1.0
        pattern = torch.einsum("bl,lchw->bchw", centered, self.carriers)
        pattern = pattern / max(self.bit_length ** 0.25, 1.0)
        if pattern.shape[-2:] != size:
            pattern = F.interpolate(pattern, size=size, mode="bilinear", align_corners=False)
        # Keep the carrier mostly in texture space; the final tanh and global
        # alpha still bound the actual image perturbation.
        low = F.avg_pool2d(pattern, kernel_size=5, stride=1, padding=2)
        return pattern - low


class TraceDecoderAdapter(nn.Module):
    """Bit-conditioned 4-level ResUNet residual adapter over decoded images."""

    def __init__(
        self,
        bit_length: int,
        channels: int = 3,
        hidden_dim: int = 256,
        image_size: int = 256,
        base_channels: int = 64,
        num_blocks: int = 3,
        max_channels: int = 512,
        carrier_strength: float = 1.0,
        carrier_grid_size: int = 32,
    ) -> None:
        super().__init__()
        self.bit_length = int(bit_length)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.image_size = int(image_size)
        self.base_channels = int(base_channels)
        self.num_blocks = int(num_blocks)
        self.max_channels = int(max_channels)
        self.carrier_strength = float(carrier_strength)

        b = self.base_channels
        c1 = min(b, self.max_channels)
        c2 = min(b * 2, self.max_channels)
        c3 = min(b * 4, self.max_channels)
        c4 = min(b * 8, self.max_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(channels, c1, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, c1), c1),
            nn.SiLU(),
        )
        self.enc0 = nn.ModuleList([_FiLMResidualBlock(c1, self.bit_length, self.hidden_dim) for _ in range(max(1, num_blocks))])
        self.down1 = _DownFiLMBlock(c1, c2, self.bit_length, self.hidden_dim, num_blocks)
        self.down2 = _DownFiLMBlock(c2, c3, self.bit_length, self.hidden_dim, num_blocks)
        self.down3 = _DownFiLMBlock(c3, c4, self.bit_length, self.hidden_dim, num_blocks)
        self.mid = nn.ModuleList([_FiLMResidualBlock(c4, self.bit_length, self.hidden_dim) for _ in range(max(2, num_blocks))])
        self.up2 = _UpFiLMBlock(c4, c3, c3, self.bit_length, self.hidden_dim, num_blocks)
        self.up1 = _UpFiLMBlock(c3, c2, c2, self.bit_length, self.hidden_dim, num_blocks)
        self.up0 = _UpFiLMBlock(c2, c1, c1, self.bit_length, self.hidden_dim, num_blocks)
        self.low_head = nn.Sequential(
            nn.GroupNorm(min(16, c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, channels, kernel_size=3, padding=1),
        )
        self.high_head = _HighFrequencyBranch(channels, self.bit_length, self.hidden_dim, max(16, b // 2))
        self.carrier = _BitCarrierBranch(self.bit_length, channels, grid_size=carrier_grid_size)
        nn.init.zeros_(self.low_head[-1].weight)
        nn.init.zeros_(self.low_head[-1].bias)

    def forward(self, x_dec: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Compute a bounded, bit-conditioned residual for ``x_dec``."""
        if x_dec.dim() != 4:
            raise ValueError(f"x_dec must be [B, C, H, W], got {tuple(x_dec.shape)}.")
        if bits.dim() != 2 or bits.size(1) != self.bit_length:
            raise ValueError(f"bits must be [B, {self.bit_length}], got {tuple(bits.shape)}.")
        bits = bits.to(device=x_dec.device, dtype=x_dec.dtype)
        h0 = self.stem(x_dec)
        for block in self.enc0:
            h0 = block(h0, bits)
        h1 = self.down1(h0, bits)
        h2 = self.down2(h1, bits)
        h3 = self.down3(h2, bits)
        hm = h3
        for block in self.mid:
            hm = block(hm, bits)
        h = self.up2(hm, h2, bits)
        h = self.up1(h, h1, bits)
        h = self.up0(h, h0, bits)
        low_residual = self.low_head(h)
        high_residual = self.high_head(x_dec, bits)
        carrier_residual = self.carrier(bits, x_dec.shape[-2:]).to(dtype=x_dec.dtype, device=x_dec.device)
        return torch.tanh(low_residual + high_residual + self.carrier_strength * carrier_residual)


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
