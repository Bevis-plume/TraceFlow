"""
src/crypto/latent_permute.py
============================
Keyed latent-space mixing layer.

Instead of a plain latent permutation, this module applies a deterministic
block-wise orthogonal transform controlled by a secret key K:

    z' = M_K(z) + beta_K

where M_K is block diagonal and each block is an orthogonal matrix.  The
inverse is exact:

    z = M_K^T(z' - beta_K)

This keeps the transform differentiable and information-preserving for the
defender, while making the attacker-facing latent topology harder to interpret
than a simple shuffle.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch
import torch.nn as nn


class LatentPermuter(nn.Module):
    """Deterministic, key-controlled invertible mixing layer.

    Args:
        secret_key: Secret string used to derive all mixing matrices.
        latent_dim: Flat latent dimension D = C * H * W.
        block_size: Size of each independently mixed latent block.
        bias_scale: Scale for the deterministic additive bias.
    """

    def __init__(
        self,
        secret_key: str,
        latent_dim: int,
        block_size: int = 16,
        bias_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim % block_size != 0:
            raise ValueError("latent_dim must be divisible by block_size")

        seed = self._derive_seed(secret_key)
        rng = np.random.default_rng(seed)

        self.latent_dim = latent_dim
        self.block_size = block_size
        self.num_blocks = latent_dim // block_size

        mix_mats = []
        inv_mats = []
        for _ in range(self.num_blocks):
            raw = rng.standard_normal((block_size, block_size)).astype(np.float32)
            q, r = np.linalg.qr(raw)
            signs = np.sign(np.diag(r)).astype(np.float32)
            signs[signs == 0] = 1.0
            q = (q * signs).astype(np.float32)
            mix_mats.append(q)
            inv_mats.append(q.T.astype(np.float32))

        # Derived from the key, so do not persist in checkpoints.  Rebuilding
        # from the same key gives identical buffers without leaking key material.
        self.register_buffer(
            "mix_mats",
            torch.from_numpy(np.stack(mix_mats, axis=0)),
            persistent=False,
        )
        self.register_buffer(
            "inv_mats",
            torch.from_numpy(np.stack(inv_mats, axis=0)),
            persistent=False,
        )

        bias = rng.uniform(-bias_scale, bias_scale, size=latent_dim).astype(np.float32)
        self.register_buffer("bias", torch.from_numpy(bias), persistent=False)

    @staticmethod
    def _derive_seed(secret_key: str) -> int:
        digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
        (seed,) = struct.unpack(">Q", digest[:8])
        return seed

    def _flatten(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        original_shape = z.shape
        return z.reshape(original_shape[0], -1), original_shape

    def _to_blocks(self, z_flat: torch.Tensor) -> torch.Tensor:
        return z_flat.view(z_flat.size(0), self.num_blocks, self.block_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply z' = M_K(z) + beta_K."""
        z_flat, original_shape = self._flatten(z)
        z_blocks = self._to_blocks(z_flat)
        z_mix = torch.einsum("bnd,ndk->bnk", z_blocks, self.mix_mats)
        z_prime_flat = z_mix.reshape(z_flat.size(0), -1) + self.bias.unsqueeze(0)
        return z_prime_flat.view(original_shape)

    def invert(self, z_prime: torch.Tensor) -> torch.Tensor:
        """Apply z = M_K^T(z' - beta_K)."""
        z_prime_flat, original_shape = self._flatten(z_prime)
        z_unbiased = z_prime_flat - self.bias.unsqueeze(0)
        z_blocks = self._to_blocks(z_unbiased)
        z_inv = torch.einsum("bnd,ndk->bnk", z_blocks, self.inv_mats)
        z_flat = z_inv.reshape(z_prime_flat.size(0), -1)
        return z_flat.view(original_shape)

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, "
            f"block_size={self.block_size}, "
            "keyed_block_orthogonal=True"
        )
