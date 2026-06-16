"""
src/models/flow_transformer.py
================================
Latent Rectified Flow Transformer for TraceFlow.

Architecture is inspired by DiT (Peebles & Xie, 2023) and SiT, adapted for
rectified flow (continuous-time, t in [0, 1]) over latent feature maps.

All components are implemented project-natively (no timm dependency).

Model presets
-------------
DiT-XS   hidden=256,  depth=6,  heads=4   (smoke / dev)
DiT-S    hidden=512,  depth=12, heads=8   (single-GPU training)
DiT-B    hidden=768,  depth=12, heads=12  (large / 5090-level)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2-D sin-cos positional embedding
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Return (grid_size^2, embed_dim) sin-cos positional embeddings."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    half = embed_dim // 2

    # Build 1-D frequency indices
    omega = np.arange(half // 2, dtype=float)
    omega /= half // 2
    omega = 1.0 / (10000 ** omega)  # (D/4,)

    # Grid positions
    pos = np.arange(grid_size, dtype=float)
    out = np.einsum("i,j->ij", pos, omega)  # (G, D/4)

    emb_sin = np.sin(out)  # (G, D/4)
    emb_cos = np.cos(out)  # (G, D/4)

    emb_1d = np.concatenate([emb_sin, emb_cos], axis=-1)  # (G, D/2)

    # Create 2-D grid by combining H and W embeddings
    emb_h = np.repeat(emb_1d, grid_size, axis=0)   # (G*G, D/2)
    emb_w = np.tile(emb_1d, (grid_size, 1))         # (G*G, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (G*G, D)
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2-D patch embedding for latent feature maps.

    Input:  (B, C, H, W)
    Output: (B, N, D)  where N = (H/p) * (W/p)
    """

    def __init__(self, latent_size: int, patch_size: int, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        assert latent_size % patch_size == 0, (
            f"latent_size ({latent_size}) must be divisible by patch_size ({patch_size})"
        )
        self.patch_size = patch_size
        self.num_patches = (latent_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, D, H/p, W/p) -> (B, N, D)
        x = self.proj(x)
        B, D, Hg, Wg = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps t in [0,1] into dense vectors."""

    def __init__(self, hidden_size: int, freq_dim: int = 256, time_scale: float = 1.0) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.time_scale = float(time_scale)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """Create sinusoidal embeddings from scalar t in [0, 1].

        t: (B,) float tensor
        Returns: (B, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float()[:, None] * freqs[None]  # (B, D/2)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, D)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Diffusion/DiT sinusoidal embeddings expect timestep-like magnitudes
        # rather than tiny [0, 1] phases. Scaling continuous flow time restores
        # meaningful frequency variation while keeping the solver convention
        # t=0 clean, t=1 noise unchanged.
        freq = self._sinusoidal(t * self.time_scale, self.freq_dim)
        return self.mlp(freq)


# ---------------------------------------------------------------------------
# Label embedding (optional class conditioning)
# ---------------------------------------------------------------------------

class LabelEmbedder(nn.Module):
    """Embeds class labels into vectors with optional dropout for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        # Extra slot for the "null" (dropped) class
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, y: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training and self.dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return self.embedding(y)


# ---------------------------------------------------------------------------
# Modulation helper
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift.

    x:     (B, N, D)
    shift: (B, D)
    scale: (B, D)
    """
    return x * (1 + scale[:, None]) + shift[:, None]


# ---------------------------------------------------------------------------
# Self-attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)            # each (B, H, N, d)

        # PyTorch SDPA dispatches to Flash / memory-efficient kernels on CUDA
        # during normal training. Inversion eval explicitly forces math SDPA
        # because it needs higher-order gradients.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Feed-forward MLP
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DiT block with adaLN-Zero conditioning
# ---------------------------------------------------------------------------

class FlowTransformerBlock(nn.Module):
    """Transformer block with adaLN-Zero timestep conditioning.

    The adaLN-Zero trick initialises the gate parameters to zero so that
    each block starts as an identity mapping at the start of training.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mlp_ratio, dropout)

        # 6 modulation params: shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-init the linear so gates start at 0 (identity residual)
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)  patch tokens
        c: (B, D)     conditioning vector (timestep [+ class])
        """
        params = self.adaLN_mod(c)  # (B, 6D)
        shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = params.chunk(6, dim=-1)

        # Attention branch
        x = x + gate_msa[:, None] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # FF branch
        x = x + gate_ff[:, None] * self.ff(modulate(self.norm2(x), shift_ff, scale_ff))
        return x


# ---------------------------------------------------------------------------
# Final projection layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """adaLN + linear projection back to patch pixels."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # Zero-init for stable starts
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_mod(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# FlowTransformer
# ---------------------------------------------------------------------------

class FlowTransformer(nn.Module):
    """DiT/SiT-style transformer backbone for rectified flow.

    Input/output shapes:
        z_t: (B, C, H, W)   noisy latent at time t
        t:   (B,)           continuous time in [0, 1]
        y:   (B,)  or None  class labels (optional)

    Forward returns a velocity field v of shape (B, C, H, W).

    Args:
        latent_channels: C in input latent.
        latent_size:     H = W of input latent.
        patch_size:      Spatial patch size p (latent_size must be divisible by p).
        hidden_size:     Transformer hidden dimension D.
        depth:           Number of transformer blocks.
        num_heads:       Number of attention heads.
        mlp_ratio:       MLP expansion ratio.
        dropout:         Dropout in feed-forward.
        class_conditional: Enable class conditioning.
        num_classes:     Number of classes (only when class_conditional=True).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 32,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        class_conditional: bool = False,
        num_classes: Optional[int] = None,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.class_conditional = class_conditional

        # Store the fully-resolved architecture config (used for checkpoint serialization).
        # This is set before any preset/kwargs application happens in build_flow_transformer,
        # so it always reflects the *actual* values the model was constructed with.
        self.config: Dict[str, Any] = dict(
            latent_channels=latent_channels,
            latent_size=latent_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            class_conditional=class_conditional,
            num_classes=num_classes,
            time_scale=float(time_scale),
        )

        # Patch embedding
        self.patch_embed = PatchEmbed(latent_size, patch_size, latent_channels, hidden_size)
        num_patches = self.patch_embed.num_patches
        grid_size = int(math.sqrt(num_patches))

        # Fixed sin-cos positional embedding (not learned)
        pos_emb = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_emb).float().unsqueeze(0))

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size, time_scale=time_scale)

        # Optional class embedding
        self.y_embedder: Optional[LabelEmbedder] = None
        if class_conditional:
            assert num_classes is not None, "num_classes required when class_conditional=True"
            self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final projection
        self.final_layer = FinalLayer(hidden_size, patch_size, latent_channels)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Xavier-uniform init for all linear layers
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic_init)

        # Re-apply zero-init for final layer (overrides xavier above)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

        # Xavier-uniform for patch-embed conv
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

        # Normal init for timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to latent feature map.

        x: (B, N, p*p*C) -> (B, C, H, W)
        """
        p = self.patch_size
        C = self.latent_channels
        H = W = self.latent_size
        Hp = H // p
        Wp = W // p
        x = x.reshape(x.shape[0], Hp, Wp, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], C, H, W)
        return x

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity field v(z_t, t, [y]).

        Args:
            z_t: Noisy latent, shape (B, C, H, W).
            t:   Continuous time in [0, 1], shape (B,).
            y:   Class labels, shape (B,) or None.

        Returns:
            v: Predicted velocity, same shape as z_t.
        """
        # Patch embed + positional encoding
        assert z_t.shape[1] == self.latent_channels, (
            f"Expected {self.latent_channels} latent channels, got {z_t.shape[1]}"
        )
        assert z_t.shape[2] == z_t.shape[3] == self.latent_size, (
            f"Expected latent_size={self.latent_size}, got spatial {z_t.shape[2]}x{z_t.shape[3]}"
        )
        x = self.patch_embed(z_t)          # (B, N, D)
        x = x + self.pos_embed             # (B, N, D)

        # Conditioning vector
        c = self.t_embedder(t)             # (B, D)
        if self.class_conditional and y is not None and self.y_embedder is not None:
            c = c + self.y_embedder(y, self.training)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final projection + unpatchify
        x = self.final_layer(x, c)         # (B, N, p*p*C)
        v = self._unpatchify(x)            # (B, C, H, W)
        return v


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "DiT-XS": dict(hidden_size=256, depth=6,  num_heads=4),
    "DiT-S":  dict(hidden_size=512, depth=12, num_heads=8),
    "DiT-B":  dict(hidden_size=768, depth=12, num_heads=12),
}


def build_flow_transformer(
    preset: Optional[str] = None,
    latent_channels: int = 4,
    latent_size: int = 32,
    patch_size: int = 2,
    hidden_size: int = 512,
    depth: int = 12,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    class_conditional: bool = False,
    num_classes: Optional[int] = None,
    time_scale: float = 1.0,
    **kwargs,
) -> FlowTransformer:
    """Build a FlowTransformer from a preset name or explicit kwargs."""
    cfg: dict = dict(
        latent_channels=latent_channels,
        latent_size=latent_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        class_conditional=class_conditional,
        num_classes=num_classes,
        time_scale=float(time_scale),
    )
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Choose from {list(PRESETS)}")
        cfg.update(PRESETS[preset])
    cfg.update(kwargs)
    return FlowTransformer(**cfg)
