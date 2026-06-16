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

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log-variance for a local VAE latent."""
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from q(z|x) with the reparameterization trick."""
        logvar = logvar.clamp(min=-20.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def sample_posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z, mu, logvar) sampled from q(z|x)."""
        mu, logvar = self.encode_stats(x)
        return self.reparameterize(mu, logvar), mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent (deterministic, uses mean)."""
        mu, _ = self.encode_stats(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image in [-1, 1]."""
        return self.decoder(z)

    def encode_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent, preserving gradient flow through the computation."""
        mu, _ = self.encode_stats(x)
        return mu

    def sample_posterior_with_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gradient-preserving posterior sample for VAE pretraining."""
        return self.sample_posterior(x)

    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image in [-1, 1], preserving gradient flow through the computation."""
        return self.decoder(z)

    def latent_shape(self, image_size: int) -> Tuple[int, int, int]:
        return (self._latent_channels, self._latent_size, self._latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode (reconstruction). Returns (x_hat, z)."""
        z = self.encode(x)
        return self.decode(z), z

    def trainable(self) -> "LocalAutoencoderBackend":
        """Re-enable gradients on all parameters (used by AE pretraining)."""
        for p in self.parameters():
            p.requires_grad_(True)
        return self


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
        checkpoint_path: Optional[str] = None,
        require_latent_stats: bool = False,
    ) -> None:
        super().__init__()

        self.backend_kind = backend
        self._require_latent_stats = bool(require_latent_stats)
        self._latent_stats_enabled = False
        self._latent_stats_meta: dict = {}
        self.register_buffer("_latent_mean", torch.zeros(1, latent_channels, 1, 1), persistent=False)
        self.register_buffer("_latent_std", torch.ones(1, latent_channels, 1, 1), persistent=False)

        if backend == "local":
            self._backend = LocalAutoencoderBackend(
                in_channels=3,
                latent_channels=latent_channels,
                image_size=image_size,
                latent_size=latent_size,
                base_channels=base_channels,
                freeze=freeze,
            )
            # Load pretrained local AE weights when provided. This is required
            # for serious training: a randomly initialised local AE destroys
            # generation quality. Loading happens before freezing matters,
            # because load_state_dict is independent of requires_grad.
            if checkpoint_path:
                self.load_local_checkpoint(checkpoint_path)
            elif self._require_latent_stats:
                raise FileNotFoundError(
                    "require_latent_stats=True but no local AE checkpoint_path was provided. "
                    "Pretrain the local autoencoder and load its checkpoint."
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

    def _normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self._latent_stats_enabled:
            return z
        mean = self._latent_mean.to(device=z.device, dtype=z.dtype)
        std = self._latent_std.to(device=z.device, dtype=z.dtype).clamp_min(1e-6)
        return (z - mean) / std

    def _denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self._latent_stats_enabled:
            return z
        mean = self._latent_mean.to(device=z.device, dtype=z.dtype)
        std = self._latent_std.to(device=z.device, dtype=z.dtype).clamp_min(1e-6)
        return z * std + mean

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to the autoencoder's native, unnormalised latent."""
        return self._backend.encode(x)

    def encode_stats_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return native posterior mean/logvar when the backend supports it."""
        if not hasattr(self._backend, "encode_stats"):
            z = self._backend.encode(x)
            return z, torch.zeros_like(z)
        return self._backend.encode_stats(x)

    def decode_raw(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a native, unnormalised autoencoder latent."""
        return self._backend.decode(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize_latent(self._backend.encode(x))

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats_raw(x)
        return self._normalize_latent(mu), logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._backend.decode(self._denormalize_latent(z))

    def encode_with_grad_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend.encode_with_grad(x)

    def encode_stats_with_grad_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self._backend, "encode_stats"):
            z = self._backend.encode_with_grad(x)
            return z, torch.zeros_like(z)
        return self._backend.encode_stats(x)

    def decode_with_grad_raw(self, z: torch.Tensor) -> torch.Tensor:
        return self._backend.decode_with_grad(z)

    def encode_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize_latent(self._backend.encode_with_grad(x))

    def encode_stats_with_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats_with_grad_raw(x)
        return self._normalize_latent(mu), logvar

    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        return self._backend.decode_with_grad(self._denormalize_latent(z))

    def sample_posterior_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return native (z, mu, logvar) sampled from q(z|x)."""
        if hasattr(self._backend, "sample_posterior"):
            return self._backend.sample_posterior(x)
        mu = self._backend.encode(x)
        return mu, mu, torch.zeros_like(mu)

    def sample_posterior_with_grad_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hasattr(self._backend, "sample_posterior_with_grad"):
            return self._backend.sample_posterior_with_grad(x)
        mu = self._backend.encode_with_grad(x)
        return mu, mu, torch.zeros_like(mu)

    def sample_posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.sample_posterior_raw(x)
        return self._normalize_latent(z), self._normalize_latent(mu), logvar

    def sample_posterior_with_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.sample_posterior_with_grad_raw(x)
        return self._normalize_latent(z), self._normalize_latent(mu), logvar

    def latent_stats_enabled(self) -> bool:
        return bool(self._latent_stats_enabled)

    def latent_stats_metadata(self) -> dict:
        if not self._latent_stats_enabled:
            return {"enabled": False}
        meta = dict(self._latent_stats_meta)
        meta.update({
            "enabled": True,
            "mean": self._latent_mean.detach().cpu().view(-1).tolist(),
            "std": self._latent_std.detach().cpu().view(-1).tolist(),
        })
        return meta

    def set_latent_stats(self, stats: Optional[dict]) -> None:
        """Enable per-channel latent standardisation from checkpoint metadata."""
        if not stats or not stats.get("enabled", True):
            self._latent_stats_enabled = False
            self._latent_stats_meta = {"enabled": False}
            return
        mean = torch.as_tensor(stats.get("mean"), dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.as_tensor(stats.get("std"), dtype=torch.float32).view(1, -1, 1, 1)
        if mean.numel() != self._latent_mean.numel() or std.numel() != self._latent_std.numel():
            raise RuntimeError(
                "Local AE latent stats do not match latent_channels. "
                f"stats mean={mean.numel()} std={std.numel()} expected={self._latent_mean.numel()}"
            )
        if torch.any(std <= 0):
            raise RuntimeError("Local AE latent stats contain non-positive std values.")
        self._latent_mean.copy_(mean.to(self._latent_mean.device))
        self._latent_std.copy_(std.to(self._latent_std.device).clamp_min(1e-6))
        self._latent_stats_enabled = True
        self._latent_stats_meta = {k: v for k, v in stats.items() if k not in {"mean", "std"}}

    def latent_shape(self, image_size: int) -> Tuple[int, int, int]:
        return self._backend.latent_shape(image_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    # ------------------------------------------------------------------
    # Local-backend checkpoint helpers
    # ------------------------------------------------------------------
    def local_state_dict(self) -> dict:
        """Return the inner local backend state dict (unprefixed keys)."""
        if not isinstance(self._backend, LocalAutoencoderBackend):
            raise RuntimeError("local_state_dict is only available for backend='local'.")
        return self._backend.state_dict()

    def load_local_checkpoint(self, path: str, map_location: str = "cpu") -> dict:
        """Load pretrained local-autoencoder weights from a checkpoint file.

        The checkpoint may either be a raw ``state_dict`` or a dict produced by
        :func:`save_local_autoencoder` containing a ``"state_dict"`` key. Keys
        may optionally be prefixed with ``"_backend."`` (when the whole
        :class:`AutoencoderBackend` was saved); the prefix is stripped.
        """
        if not isinstance(self._backend, LocalAutoencoderBackend):
            raise RuntimeError("load_local_checkpoint is only valid for backend='local'.")
        import os

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Local autoencoder checkpoint not found: {path}. "
                "Pretrain it first with `python -m scripts.pretrain_autoencoder` "
                "or `python -m scripts.traceflow train-autoencoder`."
            )
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if any(k.startswith("_backend.") for k in state):
            state = {k[len("_backend."):]: v for k, v in state.items() if k.startswith("_backend.")}
        missing, unexpected = self._backend.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Local autoencoder checkpoint does not match the configured AE "
                f"architecture. Missing keys: {list(missing)[:6]} ... "
                f"Unexpected keys: {list(unexpected)[:6]} ... "
                "Check that latent_channels/latent_size/base_channels/image_size "
                "match the pretrained AE."
            )
        payload = ckpt if isinstance(ckpt, dict) else {}
        stats = payload.get("latent_stats") or payload.get("metrics", {}).get("latent_stats")
        if stats:
            self.set_latent_stats(stats)
        elif self._require_latent_stats:
            raise RuntimeError(
                f"Local autoencoder checkpoint {path} has no latent_stats metadata. "
                "Regenerate it with the current pretrain_autoencoder script so flow training "
                "uses a sampleable standard-normal latent space."
            )
        return payload


def save_local_autoencoder(
    backend: "AutoencoderBackend",
    path: str,
    *,
    config: Optional[dict] = None,
    metrics: Optional[dict] = None,
    latent_stats: Optional[dict] = None,
) -> None:
    """Persist a local autoencoder backend to ``path``.

    Stores the unprefixed local state dict plus optional ``config``/``metrics``
    so the checkpoint can be validated and re-loaded by
    :meth:`AutoencoderBackend.load_local_checkpoint`.
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "state_dict": backend.local_state_dict(),
        "config": config or {},
        "metrics": metrics or {},
        "latent_stats": latent_stats or {},
    }
    torch.save(payload, path)
