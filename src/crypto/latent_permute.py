"""
src/crypto/latent_permute.py
============================
Core cryptographic permutation layer.

Mathematical formulation
------------------------
Let  z  ∈ ℝ^{B × C × H × W}  be the latent produced by the VAE encoder.
The permuted latent is

    z' = π_K( z ) + β_K                                            (1)

where
    π_K  : ℝ^{D} → ℝ^{D}   index-permutation derived from secret key K
    β_K  : ℝ^{D}            deterministic bias vector derived from key K
    D    = C × H × W        total spatial-channel dimension

The inverse transform recovers z from z':

    z = π_K^{-1}( z' − β_K )                                       (2)

implemented via argsort( perm_indices ).

Security property
-----------------
An attacker who inverts gradients obtains ẑ_dummy ≈ z' (the permuted
representation).  Passing ẑ_dummy through the VAE decoder without first
applying π_K^{-1} yields a semantically incoherent image (visual noise),
because the spatial-channel topology is globally scrambled.  However, the
watermark detector, which operates on z' directly, can still extract the
embedded copyright bit-stream with high confidence (the permuted space is
internally consistent).

All operations are differentiable; gradients flow through gather/scatter
unmodified, making the layer transparent to back-propagation.
"""

import hashlib
import struct

import numpy as np
import torch
import torch.nn as nn


class LatentPermuter(nn.Module):
    """Deterministic, key-controlled permutation layer for latent tensors.

    Implements Equation (1):  z' = π_K(z) + β_K

    The permutation indices and bias are derived deterministically from a
    SHA-256 digest of the secret key, so the same key always produces the
    same transform (reproducibility) while remaining opaque to key-less
    attackers.

    Args:
        secret_key: A human-readable string that acts as the cryptographic
            seed K.  Must be kept private; never store in plain-text logs.
        latent_dim: Total dimension D = C × H × W of the flattened latent.
            The permuter is registered as a buffer (not a learned parameter),
            so it is saved/loaded with the model state_dict.
        bias_scale: Scale factor s for the additive bias β_K, which is
            sampled uniformly from U(−s, s) using the derived seed.
    """

    def __init__(
        self,
        secret_key: str,
        latent_dim: int,
        bias_scale: float = 0.1,
    ) -> None:
        super().__init__()

        seed = self._derive_seed(secret_key)
        rng = np.random.default_rng(seed)

        # π_K : permutation indices, shape (D,)  — stored as a buffer so it
        #        travels to the correct device with .to(device) / .cuda()
        perm = torch.from_numpy(rng.permutation(latent_dim).astype(np.int64))
        self.register_buffer("perm_indices", perm)

        # π_K^{-1} : inverse permutation via argsort, shape (D,)
        inv_perm = torch.argsort(perm)
        self.register_buffer("inv_perm_indices", inv_perm)

        # β_K : additive bias, shape (D,)
        bias_np = rng.uniform(-bias_scale, bias_scale, size=latent_dim).astype(
            np.float32
        )
        bias = torch.from_numpy(bias_np)
        self.register_buffer("bias", bias)

        self.latent_dim = latent_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_seed(secret_key: str) -> int:
        """Derive a 64-bit unsigned integer seed from a UTF-8 secret key.

        Uses the first 8 bytes of the SHA-256 digest so that the seed space
        is 2^64, making brute-force enumeration infeasible.

        Args:
            secret_key: Arbitrary-length UTF-8 string.

        Returns:
            A non-negative Python int suitable for seeding numpy RNG.
        """
        digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
        # Unpack first 8 bytes as big-endian unsigned 64-bit integer
        (seed,) = struct.unpack(">Q", digest[:8])
        return seed

    def _flatten(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        """Flatten spatial-channel dims while preserving the batch dimension.

        Args:
            z: Latent tensor of shape (B, C, H, W).

        Returns:
            Tuple of (z_flat, original_shape) where z_flat ∈ ℝ^{B × D}.
        """
        original_shape = z.shape
        return z.view(original_shape[0], -1), original_shape

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the permutation transform z' = π_K(z) + β_K.

        The operation is fully differentiable: torch.index_select / advanced
        indexing on the last dimension passes gradients through unchanged.

        Args:
            z: VAE encoder output of shape (B, C, H, W).
                This is the *unpermuted* latent representation.

        Returns:
            z_prime: Permuted-and-shifted latent of shape (B, C, H, W).
                     This is what gets fed into the UNet and Watermarker.
        """
        z_flat, original_shape = self._flatten(z)          # (B, D)

        # π_K(z): reorder elements along the feature dimension
        # perm_indices is a 1-D LongTensor of shape (D,)
        z_perm = z_flat[:, self.perm_indices]               # (B, D)  — Eq.(1) first term

        # + β_K: broadcast bias across the batch dimension
        z_prime_flat = z_perm + self.bias.unsqueeze(0)      # (B, D)  — Eq.(1) full

        return z_prime_flat.view(original_shape)            # (B, C, H, W)

    def invert(self, z_prime: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transform  z = π_K^{-1}( z' − β_K ).

        Used by the *defence side* to reconstruct the original latent from
        a recovered (potentially attacker-supplied) permuted sample, before
        feeding it to the VAE decoder.

        Args:
            z_prime: Permuted latent of shape (B, C, H, W).

        Returns:
            z: Reconstructed original latent of shape (B, C, H, W).
               If z_prime came from an attacker who never had access to K,
               this will not equal the true z, but the spatial structure
               will at least be "un-shuffled" in the permuter's frame of
               reference — enabling watermark extraction.
        """
        z_prime_flat, original_shape = self._flatten(z_prime)   # (B, D)

        # − β_K  (undo the bias shift)
        z_unbiased = z_prime_flat - self.bias.unsqueeze(0)       # (B, D)

        # π_K^{-1}: inverse permutation via pre-computed argsort indices
        z_flat = z_unbiased[:, self.inv_perm_indices]            # (B, D)

        return z_flat.view(original_shape)                       # (B, C, H, W)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, "
            f"bias_scale derived from SHA-256 key"
        )
