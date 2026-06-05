"""
src/models/autoencoder_backend.py
==================================
Unified autoencoder backend abstraction for TraceFlow.

Provides a consistent API:
    encode(x: Tensor[B,3,H,W]) -> z: Tensor[B,C,h,w]
    decode(z: Tensor[B,C,h,w]) -> x: Tensor[B,3,H,W]  in [-1, 1]
    latent_shape(image_size) -> (C, h, w)

Supported backends:
    "local"     — project-native convolutional VAE; random init by default.
    "diffusers" — Hugging Face diffusers AutoencoderKL (optional).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_num_downsample_stages(image_size: int, latent_size: int) -> int:
    """Return number of stride-2 stages needed so image_size / 2^n == latent_size."""
    ratio = image_size / latent_size
    n = math.log2(ratio)
    if not n.is_integer():
        raise ValueError(
            f"image_size ({image_size}) / latent_size ({latent_size}) must be a power of 2, got ratio {ratio}"
        )
    return int(n)


# ---------------------------------------------------------------------------
# Local convolutional autoencoder
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        num_groups = min(32, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _LocalEncoder(nn.Module):
    """Flexible convolutional encoder: RGB image -> latent mean + logvar."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        num_stages: int = 3,
    ) -> None:
        super().__init__()
        C = base_channels
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, C, 3, padding=1),
            nn.SiLU(),
            _ResBlock(C),
        ]
        in_c = C
        for i in range(num_stages):
            out_c = in_c * 2
            layers += [
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
                nn.SiLU(),
                _ResBlock(out_c),
            ]
            in_c = out_c

        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Conv2d(in_c, latent_channels, 1)
        self.logvar_head = nn.Conv2d(in_c, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)


class _LocalDecoder(nn.Module):
    """Flexible convolutional decoder: latent -> RGB in [-1, 1]."""

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 3,
    ) -> None:
        super().__init__()
        # Mirror encoder channel progression
        C = base_channels
        in_c = C * (2 ** num_stages)

        layers: list[nn.Module] = [
            nn.Conv2d(latent_channels, in_c, 3, padding=1),
            nn.SiLU(),
            _ResBlock(in_c),
        ]
        cur_c = in_c
        for _ in range(num_stages):
            out_c = cur_c // 2
            layers += [
                nn.ConvTranspose2d(cur_c, out_c, 4, stride=2, padding=1),
                nn.SiLU(),
                _ResBlock(out_c),
            ]
            cur_c = out_c

        layers += [
            nn.Conv2d(cur_c, out_channels, 3, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        ]
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class LocalAutoencoderBackend(nn.Module):
    """Project-native autoencoder backend (no pretrained weights required).

    Suitable for smoke tests and development without internet access.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        image_size: int = 256,
        latent_size: int = 32,
        base_channels: int = 64,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        num_stages = _compute_num_downsample_stages(image_size, latent_size)
        self._latent_channels = latent_channels
        self._latent_size = latent_size

        self.encoder = _LocalEncoder(in_channels, latent_channels, base_channels, num_stages)
        self.decoder = _LocalDecoder(latent_channels, in_channels, base_channels, num_stages)

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent (deterministic, uses mean)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image in [-1, 1]."""
        return self.decoder(z)

    def encode_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent, preserving gradient flow through the computation."""
        mu, _ = self.encoder(x)
        return mu

    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image in [-1, 1], preserving gradient flow through the computation."""
        return self.decoder(z)

    def latent_shape(self, image_size: int) -> Tuple[int, int, int]:
        return (self._latent_channels, self._latent_size, self._latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode (reconstruction). Returns (x_hat, z)."""
        z = self.encode(x)
        return self.decode(z), z


# ---------------------------------------------------------------------------
# Diffusers backend (optional)
# ---------------------------------------------------------------------------

class DiffusersAutoencoderBackend(nn.Module):
    """Wraps diffusers AutoencoderKL with the unified TraceFlow API."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        latent_channels: int = 4,
        latent_size: int = 32,
        scaling_factor: float = 0.18215,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise ImportError(
                "diffusers is required for backend='diffusers'. "
                "Install it with: pip install diffusers"
            ) from exc

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        self.scaling_factor = scaling_factor
        self._latent_channels = latent_channels
        self._latent_size = latent_size

        if freeze:
            for p in self.vae.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to scaled latent."""
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample() * self.scaling_factor
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent to image in [-1, 1]."""
        x = self.vae.decode(z / self.scaling_factor).sample
        return x.clamp(-1.0, 1.0)

    def encode_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to scaled latent, preserving gradient flow through the computation."""
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample() * self.scaling_factor
        return z

    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent to image in [-1, 1], preserving gradient flow through the computation."""
        x = self.vae.decode(z / self.scaling_factor).sample
        return x.clamp(-1.0, 1.0)

    def latent_shape(self, image_size: int) -> Tuple[int, int, int]:
        return (self._latent_channels, self._latent_size, self._latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class AutoencoderBackend(nn.Module):
    """Unified autoencoder backend for TraceFlow.

    Wraps either a local or diffusers backend.  This class is the only
    entry-point the rest of the codebase should use.

    Args:
        backend: "local" or "diffusers".
        pretrained_model_name_or_path: Only used for backend="diffusers".
        latent_channels: Number of latent channels (C).
        image_size: Input image spatial size.
        latent_size: Latent spatial size (H_z = W_z).
        scaling_factor: Latent scaling factor (diffusers backend only).
        freeze: Freeze autoencoder weights.
        base_channels: Base channel width (local backend only).
    """

    def __init__(
        self,
        backend: str = "local",
        pretrained_model_name_or_path: Optional[str] = None,
        latent_channels: int = 4,
        image_size: int = 256,
        latent_size: int = 32,
        scaling_factor: float = 1.0,
        freeze: bool = True,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        if backend == "local":
            self._backend = LocalAutoencoderBackend(
                in_channels=3,
                latent_channels=latent_channels,
                image_size=image_size,
                latent_size=latent_size,
                base_channels=base_channels,
                freeze=freeze,
            )
        elif backend == "diffusers":
            if pretrained_model_name_or_path is None:
                raise ValueError(
                    "pretrained_model_name_or_path is required for backend='diffusers'"
                )
            self._backend = DiffusersAutoencoderBackend(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                latent_channels=latent_channels,
                latent_size=latent_size,
                scaling_factor=scaling_factor,
                freeze=freeze,
            )
        else:
            raise ValueError(f"Unknown autoencoder backend: {backend!r}. Choose 'local' or 'diffusers'.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._backend.decode(z)

    def encode_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend.encode_with_grad(x)

    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        return self._backend.decode_with_grad(z)

    def latent_shape(self, image_size: int) -> Tuple[int, int, int]:
        return self._backend.latent_shape(image_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._backend(x)
