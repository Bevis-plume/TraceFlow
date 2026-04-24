"""
src/models/vae.py
=================
Lightweight Convolutional Variational Autoencoder for CIFAR-10 (32×32).

Architecture summary
--------------------
Encoder:  3×32×32  →  (μ, log σ²) ∈ ℝ^{4×8×8}
                   (3 strided-conv blocks, channel progression 3→64→128→8,
                    followed by a split head that emits μ and log σ²)

Reparameterisation:
    z = μ + σ · ε,   ε ~ N(0, I)                                    (1)

Decoder:  4×8×8  →  3×32×32
          (3 transposed-conv / upsample blocks, inverse of the encoder)

The latent space dimension is:
    D = C_z × H_z × W_z = 4 × 8 × 8 = 256

This matches the `watermarker.input_dim` and `permuter` settings in
configs/default.yml and the flat dimension expected by LatentPermuter.

KL divergence (for optional VAE training signal):
    L_KL = −½ Σ_j (1 + log σ²_j − μ²_j − σ²_j)                    (2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Pre-activation residual block with GroupNorm.

    Args:
        channels: Number of input (and output) channels.
        dropout:  Dropout probability applied after the second conv.
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Feature map (B, C, H, W).
        Returns:
            Residual-added output of the same shape.
        """
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Convolutional encoder:  x ∈ ℝ^{3×32×32}  →  (μ, log σ²) ∈ ℝ^{4×8×8}.

    Spatial downsampling:  32 → 16 → 8  (two stride-2 convolutions).

    Args:
        in_channels:      Image channels C_x (default 3 for RGB).
        latent_channels:  Channels in the latent C_z (default 4).
        base_channels:    Base feature-map width (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        C = base_channels

        self.encoder = nn.Sequential(
            # 3×32×32  →  C×32×32
            nn.Conv2d(in_channels, C, kernel_size=3, padding=1),
            nn.SiLU(),
            ResBlock(C),
            # C×32×32  →  2C×16×16
            nn.Conv2d(C, C * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock(C * 2),
            # 2C×16×16  →  4C×8×8
            nn.Conv2d(C * 2, C * 4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock(C * 4),
        )

        # Two parallel 1×1 conv heads: one for μ, one for log σ²
        self.mu_head = nn.Conv2d(C * 4, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(C * 4, latent_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x into the parameters of the posterior q(z|x).

        Args:
            x: Input image batch of shape (B, C_x, 32, 32).

        Returns:
            mu:     Posterior mean,     shape (B, C_z, 8, 8).
            logvar: Posterior log-var,  shape (B, C_z, 8, 8).
                    Corresponds to log σ² in Eq.(2).
        """
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Convolutional decoder: z ∈ ℝ^{4×8×8} → x̂ ∈ ℝ^{3×32×32}.

    Mirrors the encoder: two transposed-conv upsamplings 8 → 16 → 32.

    Args:
        latent_channels: Channels in the latent C_z (default 4).
        out_channels:    Image channels C_x (default 3).
        base_channels:   Base feature-map width (default 64).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        C = base_channels

        self.decoder = nn.Sequential(
            # C_z×8×8  →  4C×8×8
            nn.Conv2d(latent_channels, C * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            ResBlock(C * 4),
            # 4C×8×8  →  2C×16×16
            nn.ConvTranspose2d(C * 4, C * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock(C * 2),
            # 2C×16×16  →  C×32×32
            nn.ConvTranspose2d(C * 2, C, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock(C),
            # C×32×32  →  C_x×32×32,  pixel values ∈ [0,1] after sigmoid
            nn.Conv2d(C, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent sample to pixel space.

        Args:
            z: Latent tensor of shape (B, C_z, 8, 8).
               NOTE: this z may be either the *original* latent or the
               *permuted* z'.  Passing z' without first inverting via
               LatentPermuter.invert() will produce semantically incoherent
               images — which is the intended attacker-failure mode.

        Returns:
            x_hat: Reconstructed image of shape (B, C_x, 32, 32) in [0, 1].
        """
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """β-VAE wrapping Encoder + reparameterisation + Decoder.

    The full generative model is:
        p(x|z) implemented by Decoder
        q(z|x) = N(μ_φ(x), diag(σ²_φ(x)))  implemented by Encoder
        p(z)   = N(0, I)

    Args:
        in_channels:     Image channels C_x (default 3).
        latent_channels: Latent channels C_z (default 4).
        base_channels:   Feature-map width at first conv layer (default 64).
        kl_weight:       β scaling factor for the KL term Eq.(2) (default 1e-4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        kl_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, base_channels)
        self.decoder = Decoder(latent_channels, in_channels, base_channels)
        self.kl_weight = kl_weight

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ q(z|x) using the reparameterisation trick.

        z = μ + σ · ε,   ε ~ N(0, I)                               (1)

        Args:
            mu:     Mean tensor,       shape (B, C_z, H_z, W_z).
            logvar: Log-variance,      shape (B, C_z, H_z, W_z).

        Returns:
            z: Sampled latent,         shape (B, C_z, H_z, W_z).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( q(z|x) || p(z) ) analytically for diagonal Gaussians.

        L_KL = −½ Σ_j (1 + log σ²_j − μ²_j − σ²_j)               (2)

        Args:
            mu:     Posterior mean,    shape (B, C_z, H_z, W_z).
            logvar: Posterior log-var, shape (B, C_z, H_z, W_z).

        Returns:
            Scalar KL loss averaged over the batch.
        """
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder only; returns (μ, log σ²) without sampling.

        Args:
            x: Input image batch (B, C_x, 32, 32).

        Returns:
            mu:     Shape (B, C_z, 8, 8).
            logvar: Shape (B, C_z, 8, 8).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Run decoder only.

        Args:
            z: Latent tensor (B, C_z, 8, 8).  May be raw z or permuted z'.

        Returns:
            x_hat: Shape (B, C_x, 32, 32).
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterise → decode.

        Args:
            x: Input image batch of shape (B, C_x, 32, 32) in [0, 1].

        Returns:
            x_hat:  Reconstruction,    shape (B, C_x, 32, 32).
            z:      Sampled latent,    shape (B, C_z, 8, 8).  This z is
                    the *unpermuted* latent; pass it to LatentPermuter next.
            mu:     Posterior mean,    shape (B, C_z, 8, 8).
            logvar: Posterior log-var, shape (B, C_z, 8, 8).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, z, mu, logvar

    def loss(
        self, x: torch.Tensor, x_hat: torch.Tensor,
        mu: torch.Tensor, logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the VAE ELBO loss.

        L_VAE = L_recon + β · L_KL
        L_recon = MSE(x̂, x)

        Args:
            x:      Original images (B, C_x, 32, 32).
            x_hat:  Reconstructed images (B, C_x, 32, 32).
            mu:     Posterior mean (B, C_z, 8, 8).
            logvar: Posterior log-var (B, C_z, 8, 8).

        Returns:
            total:   Scalar total VAE loss.
            recon:   Scalar reconstruction loss.
            kl:      Scalar KL loss.
        """
        recon = F.mse_loss(x_hat, x)
        kl = self.kl_divergence(mu, logvar)
        total = recon + self.kl_weight * kl
        return total, recon, kl
