"""
src/models/watermarker.py
=========================
Lightweight MLP watermark detector that operates in the *permuted*
latent space z' ∈ ℝ^{C_z × H_z × W_z} = ℝ^{4 × 8 × 8}.

Mathematical role
-----------------
Given the permuted (and cryptographically scrambled) latent z', the
Watermarker W_φ predicts a copyright bit-stream w ∈ {0,1}^M:

    ŵ = σ( W_φ( flatten(z') ) )   ∈ (0,1)^M                        (1)

where σ(·) is the element-wise sigmoid.  During training, ŵ is pushed
towards a *fixed* target bit-string w* via Binary Cross-Entropy loss:

    L_wm = BCE( ŵ, w* )                                              (2)

This loss term is combined with the diffusion denoising loss:

    L_total = L_diffusion + λ · L_wm                                 (3)

Traceability argument
---------------------
An attacker who inverts gradients recovers ẑ_dummy ≈ z' (a corrupted
permuted latent).  Because the permutation π_K is unknown to the attacker,
ẑ_dummy decoded through the VAE produces visual noise.  However, running
ẑ_dummy (or its re-encoded latent) through W_φ still produces ŵ ≈ w*,
because the Watermarker was trained on z' which has the same
spatial-channel statistics as ẑ_dummy — enabling forensic attribution.

Dimension contract (matches configs/default.yml)
-------------------------------------------------
    input_dim  = C_z × H_z × W_z = 4 × 8 × 8 = 256
    hidden_dims = [512, 256]
    output_dim = 64           (64-bit copyright message)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Watermarker(nn.Module):
    """Multi-layer perceptron that extracts a watermark from a permuted latent.

    Architecture:
        flatten(z')  →  [Linear(D, h_0) → BN → SiLU → Dropout]
                     →  [Linear(h_{i-1}, h_i) → BN → SiLU → Dropout] × n
                     →  Linear(h_n, M) → Sigmoid
                     →  ŵ ∈ (0,1)^M

    Args:
        input_dim:   Flat dimension D of z' (default 256 = 4×8×8).
        hidden_dims: Sequence of hidden layer widths (default [512, 256]).
        output_dim:  Watermark bit-length M (default 64).
        dropout:     Dropout probability applied after each hidden layer
                     (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        layers += [
            nn.Linear(in_dim, output_dim),
            nn.Sigmoid(),    # ŵ ∈ (0,1)^M, interpreted as bit probabilities
        ]

        self.net = nn.Sequential(*layers)

        # Store dimension for shape-checks
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, z_prime: torch.Tensor) -> torch.Tensor:
        """Predict the copyright bit-stream from a permuted latent.

        Args:
            z_prime: Permuted latent z' of shape (B, C_z, H_z, W_z)
                     **or** already-flattened shape (B, D).
                     This must be the *permuted* representation (after
                     LatentPermuter.forward) — passing the original z
                     will only degrade watermark accuracy.

        Returns:
            w_hat: Predicted bit probabilities of shape (B, M),
                   where M = output_dim.  Values ∈ (0,1).
                   Threshold at 0.5 to recover binary bits.
        """
        if z_prime.dim() > 2:
            # Flatten spatial-channel dims while preserving batch axis
            z_flat = z_prime.view(z_prime.size(0), -1)           # (B, D)
        else:
            z_flat = z_prime                                      # (B, D)

        assert z_flat.size(1) == self.input_dim, (
            f"Watermarker expects input_dim={self.input_dim}, "
            f"got {z_flat.size(1)}.  Check latent shape."
        )

        return self.net(z_flat)                                   # (B, M)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def loss(w_hat: torch.Tensor, w_target: torch.Tensor) -> torch.Tensor:
        """Binary Cross-Entropy watermark loss  L_wm = BCE(ŵ, w*).

        Args:
            w_hat:    Predicted bit probabilities (B, M), from forward().
            w_target: Target bit-string (M,) or (B, M), dtype float, values
                      in {0.0, 1.0}.  Broadcast across batch if 1-D.

        Returns:
            Scalar BCE loss.
        """
        return F.binary_cross_entropy(w_hat, w_target.expand_as(w_hat))

    @staticmethod
    def bit_accuracy(w_hat: torch.Tensor, w_target: torch.Tensor) -> torch.Tensor:
        """Fraction of correctly decoded bits (threshold at 0.5).

        Args:
            w_hat:    Predicted bit probabilities (B, M).
            w_target: Ground-truth bits (M,) or (B, M), float {0., 1.}.

        Returns:
            Scalar accuracy in [0, 1].
        """
        predicted_bits = (w_hat >= 0.5).float()
        target_bits = w_target.expand_as(predicted_bits)
        return (predicted_bits == target_bits).float().mean()


# ---------------------------------------------------------------------------
# Target watermark utilities
# ---------------------------------------------------------------------------

def generate_random_watermark(bit_length: int, seed: int | None = None) -> torch.Tensor:
    """Generate a fixed random binary target watermark w* ∈ {0,1}^M.

    The watermark is deterministic given `seed` so that the same target is
    used both during training and during forensic evaluation.

    Args:
        bit_length: Watermark bit-length M (e.g. 64).
        seed:       Optional integer seed for reproducibility.

    Returns:
        w_star: Float tensor of shape (M,) with values in {0., 1.}.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    return torch.bernoulli(torch.full((bit_length,), 0.5), generator=rng)
