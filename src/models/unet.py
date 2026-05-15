"""
src/models/unet.py
==================
Lightweight DDPM-style UNet that operates entirely in the *permuted*
latent space  z' ∈ ℝ^{C_z × H_z × W_z} = ℝ^{4 × 8 × 8}.

Architecture overview
---------------------
Input:  (z'_t, t)  where
    z'_t = √ᾱ_t · z' + √(1−ᾱ_t) · ε,  ε ~ N(0,I)    (forward diffusion)
Output: ε_θ(z'_t, t) ≈ ε                              (noise prediction)

The UNet contains:
  • Time embedding: sinusoidal position encoding → 2-layer MLP → τ ∈ ℝ^{d_τ}
  • Encoder path: [ResBlock × num_res_blocks + optional Downsample] × len(mult)
  • Bottleneck: 2 × ResBlock
  • Decoder path: [ResBlock × (num_res_blocks+1) + optional Upsample] × len(mult)
  • Output: GroupNorm → SiLU → Conv2d → predicted noise ε_θ

Because the spatial size of z' is already 8×8, we use only 2 downsampling
levels (8→4→2) with channel_mult=[1,2,2] to keep the model cheap while
still having a meaningful receptive field.

Time embedding uses sinusoidal encoding of dimension `model_channels` as in
the original "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time-step embedding (Ho et al., 2020).

    For timestep t, produces a vector of length `dim`:
        PE(t, 2i)   = sin(t / 10000^{2i/dim})
        PE(t, 2i+1) = cos(t / 10000^{2i/dim})

    Args:
        dim: Embedding dimension (must be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "Sinusoidal embedding dim must be even."
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings for a batch of time-steps.

        Args:
            t: Integer time-step indices of shape (B,).

        Returns:
            Embedding tensor of shape (B, dim).
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )                                                    # (half,)
        args = t[:, None].float() * freqs[None, :]          # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class TimeEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal time encoding to τ ∈ ℝ^{d_τ}.

    Args:
        model_channels: Input sinusoidal encoding dimension.
        time_embed_dim: Output embedding dimension d_τ.
    """

    def __init__(self, model_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(model_channels)
        self.mlp = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time-step indices (B,).
        Returns:
            τ: Time embedding (B, time_embed_dim).
        """
        return self.mlp(self.sinusoidal(t))


# ---------------------------------------------------------------------------
# ResBlock with time conditioning
# ---------------------------------------------------------------------------

def _group_count(channels: int, max_groups: int = 32) -> int:
    """Return the largest GroupNorm group count that divides channels."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlockTime(nn.Module):
    """Pre-activation residual block conditioned on a time embedding τ.

    The time signal is projected to channel width and added after the first
    convolution (AdaGN-style additive conditioning):
        h = Conv(GroupNorm(SiLU(x)))  +  Linear(SiLU(τ))   (scale/shift free)

    Args:
        in_channels:    Number of input feature-map channels.
        out_channels:   Number of output feature-map channels.
        time_embed_dim: Dimension of the time embedding τ.
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection: 1×1 conv if channel count changes
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   Feature map (B, C_in, H, W).
            tau: Time embedding (B, time_embed_dim).

        Returns:
            Output feature map (B, C_out, H, W).
        """
        h = self.conv1(F.silu(self.norm1(x)))
        # Broadcast time signal over spatial dims: (B, C_out) → (B, C_out, 1, 1)
        h = h + self.time_proj(tau)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Down / Up sampling helpers
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """Strided 2×2 average-pool followed by a channel-preserving conv.

    Halves spatial dimensions: (B, C, H, W) → (B, C, H/2, W/2).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour 2× upsample followed by a channel-preserving conv.

    Doubles spatial dimensions: (B, C, H, W) → (B, C, 2H, 2W).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Lightweight DDPM UNet for 4×8×8 permuted latent space.

    Predicts the noise ε_θ(z'_t, t) given a noisy permuted latent z'_t and
    the integer diffusion timestep t.

    Args:
        in_channels:    Channels of the input latent C_z (default 4).
        model_channels: Base feature-map width (default 64).
        channel_mult:   Multipliers per resolution level (default [1, 2, 2]).
        num_res_blocks: ResBlocks per resolution (default 2).
        time_embed_dim: Time embedding MLP output dimension d_τ (default 128).
        dropout:        Dropout probability (default 0.1).

    Input / output tensor sizes (with defaults):
        Spatial path:  4×8×8  →  [64ch@8] →  [128ch@4] →  [128ch@2]
                               →  bottleneck  →  [128ch@4] → [64ch@8]
        Output:        4×8×8  (same as input, matches ε shape)
    """

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 64,
        channel_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        time_embed_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channel_mult is None:
            channel_mult = [1, 2, 2]

        self.time_embedding = TimeEmbedding(model_channels, time_embed_dim)

        # Channel widths at each resolution level
        ch_list = [model_channels * m for m in channel_mult]

        # ----------------------------------------------------------------
        # Input projection
        # ----------------------------------------------------------------
        self.input_proj = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # ----------------------------------------------------------------
        # Encoder path
        # ----------------------------------------------------------------
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.down_samples: nn.ModuleList = nn.ModuleList()
        skip_ch_list: list[int] = []

        in_ch = model_channels
        for level, out_ch in enumerate(ch_list):
            level_blocks: list[nn.Module] = []
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlockTime(in_ch, out_ch, time_embed_dim, dropout)
                )
                skip_ch_list.append(out_ch)
                in_ch = out_ch
            self.down_blocks.append(nn.ModuleList(level_blocks))
            if level < len(ch_list) - 1:   # no downsample after last level
                self.down_samples.append(Downsample(in_ch))
            else:
                self.down_samples.append(nn.Identity())  # placeholder

        # ----------------------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------------------
        self.mid_block1 = ResBlockTime(in_ch, in_ch, time_embed_dim, dropout)
        self.mid_block2 = ResBlockTime(in_ch, in_ch, time_embed_dim, dropout)

        # ----------------------------------------------------------------
        # Decoder path (skip connections concatenated)
        # ----------------------------------------------------------------
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.up_samples: nn.ModuleList = nn.ModuleList()

        for level, out_ch in reversed(list(enumerate(ch_list))):
            level_blocks = []
            for i in range(num_res_blocks):
                sk_ch = skip_ch_list.pop()
                level_blocks.append(
                    ResBlockTime(in_ch + sk_ch, out_ch, time_embed_dim, dropout)
                )
                in_ch = out_ch
            self.up_blocks.append(nn.ModuleList(level_blocks))
            if level > 0:
                self.up_samples.append(Upsample(in_ch))
            else:
                self.up_samples.append(nn.Identity())  # placeholder

        # ----------------------------------------------------------------
        # Output projection
        # ----------------------------------------------------------------
        self.output_proj = nn.Sequential(
            nn.GroupNorm(_group_count(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_prime_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise ε_θ(z'_t, t).

        Args:
            z_prime_t: Noisy permuted latent of shape (B, C_z, H_z, W_z).
                       This is  z'_t = √ᾱ_t · z' + √(1−ᾱ_t) · ε.
            t:         Integer diffusion timestep indices of shape (B,).
                       Values in [0, T−1].

        Returns:
            eps_pred: Predicted noise of shape (B, C_z, H_z, W_z),
                      same spatial-channel layout as z'_t.
        """
        # Time embedding τ ∈ ℝ^{d_τ}
        tau = self.time_embedding(t)                          # (B, time_embed_dim)

        # Input projection
        h = self.input_proj(z_prime_t)                        # (B, C_base, H_z, W_z)

        # Encoder: collect skip connections
        skips: list[torch.Tensor] = []
        for level_blocks, down in zip(self.down_blocks, self.down_samples):
            for blk in level_blocks:
                h = blk(h, tau)
                skips.append(h)
            h = down(h) if not isinstance(down, nn.Identity) else h

        # Bottleneck
        h = self.mid_block1(h, tau)
        h = self.mid_block2(h, tau)

        # Decoder: concatenate skip connections
        for level_blocks, up in zip(self.up_blocks, self.up_samples):
            for blk in level_blocks:
                skip = skips.pop()
                h = blk(torch.cat([h, skip], dim=1), tau)
            h = up(h) if not isinstance(up, nn.Identity) else h

        return self.output_proj(h)                            # (B, C_z, H_z, W_z)
