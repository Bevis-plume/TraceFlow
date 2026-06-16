"""
src/security/keyed_bottleneck.py
==================================
Keyed Latent Bottleneck — a deterministic, invertible linear transform on latents.

Design
------
The transform is controlled by a ``secret_key`` string and is NOT learned.
All transform matrices are derived from SHA-256(secret_key) at instantiation time.

Important distinctions
-----------------------
- ``z``          — the latent produced by the autoencoder.  It is NOT the key.
- ``secret_key`` — controls the permutation/rotation (known only to the defender).
- ``z_k``        — the transformed latent; what the FlowTransformer is trained on.

The transform is invertible (z_k → z) by any party who knows secret_key.
An adversary without secret_key sees only z_k and cannot efficiently recover z.

Math
----
Given z ∈ ℝ^{B × D}  (D = C × H × W):

  Forward:   z_k = Q * z + β    (block-wise; Q = block-diagonal orthogonal matrix)
  Inverse:   z   = Qᵀ * (z_k − β)

Q is block-diagonal: Q = diag(Q₁, Q₂, …, Q_{D/block_size}).
Each Qᵢ is an orthogonal matrix of shape (block_size × block_size).
Since Qᵀ Q = I (orthogonal), the inverse is exact in exact arithmetic.

Numerical precision: matrices are computed in float64 and stored as float32.
Expected reconstruction error: < 1e-5 in float32.

Buffers W and beta are registered as non-persistent — they are NOT saved to
checkpoint state_dicts and must be re-derived from the same key at load time.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class KeyedLatentBottleneck(nn.Module):
    """Block-orthogonal keyed latent transform.

    All matrices are derived from ``secret_key`` and are parameter-free.
    Buffers are non-persistent (not stored in checkpoints).

    Args:
        secret_key:       Defender's secret string.  NEVER stored publicly.
        latent_channels:  C in [B, C, H, W].
        latent_size:      H = W (assumed square).
        block_size:       Dimension of each orthogonal sub-block.
                          D = C * H * W must be divisible by block_size.
        block_layout:     "flat" keeps the legacy contiguous flatten layout.
                          "patch" groups each block as one DiT-style
                          C×p×p latent patch, preserving spatial locality.
        bias_scale:       Std-dev of the additive key-derived bias vector β.
                          Set to 0.0 to disable the bias term.
        mode:             Transform variant.  Currently only "block_orthogonal".
    """

    def __init__(
        self,
        secret_key: str,
        latent_channels: int,
        latent_size: int,
        block_size: int = 16,
        block_layout: str = "flat",
        bias_scale: float = 0.1,
        mode: str = "block_orthogonal",
    ) -> None:
        super().__init__()

        D = latent_channels * latent_size * latent_size
        if D % block_size != 0:
            raise ValueError(
                f"D={D} (={latent_channels}×{latent_size}×{latent_size}) "
                f"must be divisible by block_size={block_size}."
            )

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.block_size = block_size
        block_layout = str(block_layout or "flat").lower()
        if block_layout not in {"flat", "patch"}:
            raise ValueError(f"Unknown keyed block_layout={block_layout!r}; expected 'flat' or 'patch'.")

        self.block_layout = block_layout
        self.bias_scale = bias_scale
        self.mode = mode
        self._D = D
        self._num_blocks = D // block_size

        self._patch_size: Optional[int] = None
        if self.block_layout == "patch":
            patch_area = block_size / float(latent_channels)
            patch_size = int(math.sqrt(patch_area))
            if patch_size * patch_size * latent_channels != block_size:
                raise ValueError(
                    "block_layout='patch' requires block_size = latent_channels * p * p "
                    f"for an integer p; got block_size={block_size}, latent_channels={latent_channels}."
                )
            if latent_size % patch_size != 0:
                raise ValueError(
                    f"latent_size={latent_size} must be divisible by patch size {patch_size} "
                    "for block_layout='patch'."
                )
            self._patch_size = patch_size

        # ------------------------------------------------------------------
        # Derive seed from SHA-256(secret_key).
        # Only the first 8 hex characters (32 bits) are used for numpy's RNG.
        # The full 256-bit digest provides ample uniqueness; we truncate only
        # for the seed integer.
        # The secret_key is intentionally NOT stored as an instance attribute
        # to prevent accidental serialisation into logs or checkpoints.
        # ------------------------------------------------------------------
        digest = hashlib.sha256(secret_key.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)

        # ------------------------------------------------------------------
        # Build block-orthogonal matrices via QR decomposition.
        # Computed in float64 for precision; stored as float32.
        # ------------------------------------------------------------------
        rng = np.random.default_rng(seed)
        blocks = []
        for _ in range(self._num_blocks):
            # float64 by default for standard_normal
            R = rng.standard_normal((block_size, block_size))
            Q, _ = np.linalg.qr(R)        # Q is orthogonal in float64
            blocks.append(Q.astype(np.float32))

        # W shape: (num_blocks, block_size, block_size)
        W = np.stack(blocks, axis=0)

        # persistent=False → excluded from state_dict → not saved in checkpoints.
        # To reproduce the transform, re-instantiate with the same secret_key.
        self.register_buffer("W", torch.from_numpy(W), persistent=False)

        # ------------------------------------------------------------------
        # Build key-derived bias vector β ∈ ℝ^D.
        # ------------------------------------------------------------------
        if bias_scale > 0.0:
            beta = rng.standard_normal(D).astype(np.float32) * float(bias_scale)
            self.register_buffer("beta", torch.from_numpy(beta), persistent=False)
        else:
            self.register_buffer("beta", None, persistent=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_for_blocks(self, z: torch.Tensor) -> torch.Tensor:
        """Flatten a latent tensor into the configured keyed block layout."""
        if self.block_layout == "flat":
            return z.flatten(1)

        assert self._patch_size is not None
        B, C, H, W = z.shape
        p = self._patch_size
        if C != self.latent_channels or H != self.latent_size or W != self.latent_size:
            raise ValueError(
                f"Expected latent shape (B,{self.latent_channels},{self.latent_size},{self.latent_size}) "
                f"for patch keyed transform, got {tuple(z.shape)}."
            )
        # (B,C,H,W) -> (B,H/p,W/p,C,p,p) -> (B,D). Each block is one DiT patch.
        return (
            z.contiguous().view(B, C, H // p, p, W // p, p)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(B, self._D)
        )

    def _unflatten_from_blocks(self, z_flat: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`_flatten_for_blocks` back to (B,C,H,W)."""
        if self.block_layout == "flat":
            return z_flat.view_as(like)

        assert self._patch_size is not None
        B = z_flat.shape[0]
        C = self.latent_channels
        H = W = self.latent_size
        p = self._patch_size
        return (
            z_flat.view(B, H // p, W // p, C, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view_as(like)
        )

    def _apply_blocks(self, z_flat: torch.Tensor, transpose: bool) -> torch.Tensor:
        """Apply the block-diagonal orthogonal transform (or its transpose).

        Args:
            z_flat:    (B, D) float32 tensor.
            transpose: If True, applies Wᵀ per block (used for inversion).

        Returns:
            (B, D) float32 tensor.
        """
        B, D = z_flat.shape
        bs = self.block_size
        nb = D // bs

        z_r = z_flat.view(B, nb, bs)   # (B, num_blocks, block_size)

        # W: (num_blocks, block_size, block_size)
        #
        # Forward (W @ z):
        #   output[b, i, k] = Σ_j W[i, k, j] * z[b, i, j]
        #   einsum pattern: "ikj, bij -> bik"
        #
        # Inverse (Wᵀ @ z):
        #   output[b, i, k] = Σ_j W[i, j, k] * z[b, i, j]
        #   einsum pattern: "ijk, bij -> bik"
        if not transpose:
            z_out = torch.einsum("ikj,bij->bik", self.W, z_r)
        else:
            z_out = torch.einsum("ijk,bij->bik", self.W, z_r)

        return z_out.reshape(B, D)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply keyed transform: z_k = Q * z + β  (block-wise).

        Args:
            z: Clean latent (B, C, H, W).

        Returns:
            z_k: Transformed latent in protected space (B, C, H, W).

        Note:
            Operations run in float32 for numerical stability.
            Output is cast back to the input dtype.
        """
        original_dtype = z.dtype
        z_f = z.float()
        z_flat = self._flatten_for_blocks(z_f)             # (B, D)

        z_k_flat = self._apply_blocks(z_flat, transpose=False)   # Q * z

        if self.beta is not None:
            z_k_flat = z_k_flat + self.beta               # + β

        return self._unflatten_from_blocks(z_k_flat, z_f).to(original_dtype)

    def invert(self, z_k: torch.Tensor) -> torch.Tensor:
        """Invert the keyed transform: z = Qᵀ * (z_k − β).

        Only the defender who knows secret_key can reconstruct z from z_k.
        Adversaries who see only z_k (generated samples) cannot invert without the key.

        Args:
            z_k: Protected latent (B, C, H, W).

        Returns:
            z: Reconstructed clean latent (B, C, H, W).
        """
        original_dtype = z_k.dtype
        z_k_f = z_k.float()
        z_k_flat = self._flatten_for_blocks(z_k_f)        # (B, D)

        if self.beta is not None:
            z_k_flat = z_k_flat - self.beta               # subtract bias first

        z_flat = self._apply_blocks(z_k_flat, transpose=True)    # Qᵀ * (z_k - β)

        return self._unflatten_from_blocks(z_flat, z_k_f).to(original_dtype)
