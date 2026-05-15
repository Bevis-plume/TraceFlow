"""
src/models/watermarker.py
=========================
Block-coded watermark detector for mixed latent representations.

The latent is flattened and split into equal-sized blocks.  A shared MLP
maps each block to a small bit-group, and the concatenation of all groups
forms the full watermark vector:

    z' -> block_1, ..., block_N -> bits_1, ..., bits_N -> w_hat

This makes the watermark more structured than a single global classifier and
aligns better with the block-wise invertible mixing used by the permuter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Watermarker(nn.Module):
    """Shared block-wise MLP watermark detector.

    Args:
        input_dim: Flat latent dimension D.
        hidden_dims: Hidden widths for the per-block MLP.
        output_dim: Total watermark length M.
        block_size: Size of each latent block.
        dropout: Dropout after hidden layers.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
        block_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if input_dim % block_size != 0:
            raise ValueError("input_dim must be divisible by block_size")
        if output_dim % (input_dim // block_size) != 0:
            raise ValueError("output_dim must be divisible by num_blocks")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.block_size = block_size
        self.num_blocks = input_dim // block_size
        self.bits_per_block = output_dim // self.num_blocks

        layers: list[nn.Module] = []
        in_dim = block_size
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.bits_per_block))
        self.block_net = nn.Sequential(*layers)

    def forward(self, z_prime: torch.Tensor) -> torch.Tensor:
        """Predict the watermark bits from a mixed latent."""
        if z_prime.dim() > 2:
            z_flat = z_prime.view(z_prime.size(0), -1)
        else:
            z_flat = z_prime

        assert z_flat.size(1) == self.input_dim, (
            f"Watermarker expects input_dim={self.input_dim}, got {z_flat.size(1)}"
        )

        blocks = z_flat.view(z_flat.size(0) * self.num_blocks, self.block_size)
        bits = self.block_net(blocks)
        bits = torch.sigmoid(bits)
        return bits.view(z_flat.size(0), self.output_dim)

    @staticmethod
    def loss(w_hat: torch.Tensor, w_target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(w_hat, w_target.expand_as(w_hat))

    @staticmethod
    def bit_accuracy(w_hat: torch.Tensor, w_target: torch.Tensor) -> torch.Tensor:
        predicted_bits = (w_hat >= 0.5).float()
        target_bits = w_target.expand_as(predicted_bits)
        return (predicted_bits == target_bits).float().mean()


def generate_random_watermark(bit_length: int, seed: int | None = None) -> torch.Tensor:
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    return torch.bernoulli(torch.full((bit_length,), 0.5), generator=rng)
