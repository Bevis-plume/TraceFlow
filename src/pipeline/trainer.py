"""
src/pipeline/trainer.py
=======================
Joint training pipeline for the Trackable Latent Diffusion defence system.

Training data-flow (one step)
------------------------------
    x  ──► VAE.encode ──► z ──► LatentPermuter.forward ──► z'
                                        │                    │
                           [frozen after VAE pre-train]      ├──► Watermarker ──► ŵ
                                                             │
                                                     q(z'_t | z', t)
                                                             │
                                                           UNet ──► ε_θ

Joint loss
----------
    L_diffusion = MSE( ε_θ(z'_t, t), ε )                              (1)
    L_wm        = BCE( ŵ, w* )                                         (2)
    L_total     = L_diffusion + λ · L_wm                               (3)

Because LatentPermuter is a pure index-select + add operation (no
learned parameters), gradients flow through it unchanged:
    ∂L/∂z = ∂L/∂z' · ∂z'/∂z  (identity up to permutation ordering)

Diffusion noise schedule (DDPM, linear β schedule)
---------------------------------------------------
    β_t   = β_start + t/(T−1) · (β_end − β_start)
    ᾱ_t   = ∏_{s=1}^{t} (1 − β_s)
    z'_t  = √ᾱ_t · z' + √(1−ᾱ_t) · ε,   ε ~ N(0,I)                 (4)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.crypto.latent_permute import LatentPermuter
from src.models.vae import VAE
from src.models.unet import UNet
from src.models.watermarker import Watermarker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass (mirrors configs/default.yml)
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Flat configuration container for the joint trainer.

    All numeric hyper-parameters match the keys in configs/default.yml so
    that callers can simply unpack the YAML dict.

    Args:
        device:          Torch device string, e.g. "cuda" or "cpu".
        num_epochs:      Total training epochs.
        batch_size:      Batch size (informational; DataLoader is passed in).
        learning_rate:   Adam learning rate.
        weight_decay:    Adam L2 weight-decay.
        grad_clip_norm:  Max norm for gradient clipping (0 = disabled).
        lambda_wm:       λ in L_total = L_diffusion + λ · L_wm.
        log_interval:    Log every N optimiser steps.
        save_interval:   Save checkpoint every N epochs.
        checkpoint_dir:  Directory for saving model checkpoints.
        diffusion_timesteps: Total diffusion steps T.
        beta_start:      β_1 in the linear schedule.
        beta_end:        β_T in the linear schedule.
        wm_seed:         Seed for the fixed target watermark w*.
        wm_bits:         Watermark bit-length M (must match Watermarker.output_dim).
    """
    device: str = "cuda"
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    lambda_wm: float = 1.0
    log_interval: int = 50
    save_interval: int = 10
    checkpoint_dir: str = "./checkpoints"
    diffusion_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    wm_seed: int = 0
    wm_bits: int = 64


# ---------------------------------------------------------------------------
# Noise schedule helper
# ---------------------------------------------------------------------------

class LinearNoiseSchedule:
    """Pre-computes and caches the DDPM linear β/ᾱ schedule on the GPU.

    Tensors are pre-allocated once and reused every training step to avoid
    repeated device transfers.

    Args:
        timesteps: Total diffusion steps T.
        beta_start: β_1.
        beta_end:   β_T.
        device:     Torch device.
    """

    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
    ) -> None:
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)          # ᾱ_t, shape (T,)

        # √ᾱ_t and √(1−ᾱ_t), pre-expanded for broadcasting with (B,C,H,W)
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
        self.timesteps = timesteps

    def q_sample(
        self,
        z_prime: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample z'_t from q(z'_t | z', t).

        z'_t = √ᾱ_t · z'  +  √(1−ᾱ_t) · ε,   ε ~ N(0,I)            (4)

        Args:
            z_prime: Clean permuted latent (B, C, H, W).
            t:       Timestep indices (B,), values in [0, T−1].
            noise:   Pre-sampled ε (B, C, H, W).  Sampled internally if None.

        Returns:
            z_prime_t: Noisy latent (B, C, H, W).
            noise:     The ε actually used (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(z_prime)

        # Gather schedule values for each sample's timestep t_i
        # Shape: (B,) → (B, 1, 1, 1) for broadcasting over C×H×W
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        z_prime_t = sqrt_a * z_prime + sqrt_1a * noise
        return z_prime_t, noise


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the joint training of VAE + LatentPermuter + UNet + Watermarker.

    Training objective per Eq.(3):
        L_total = L_diffusion + λ · L_wm

    The VAE encoder acts as a *feature extractor* here.  Its weights can be
    either frozen (two-stage training) or jointly optimised (end-to-end).
    By default, all four components are updated together.

    Args:
        vae:       Pre-built VAE instance.
        permuter:  Pre-built LatentPermuter instance (fixed, no grad required).
        unet:      Pre-built UNet instance.
        watermarker: Pre-built Watermarker instance.
        config:    TrainerConfig with all hyper-parameters.
    """

    def __init__(
        self,
        vae: VAE,
        permuter: LatentPermuter,
        unet: UNet,
        watermarker: Watermarker,
        config: TrainerConfig,
    ) -> None:
        self.device = torch.device(config.device)
        self.config = config

        # Move all models to target device
        self.vae = vae.to(self.device)
        self.permuter = permuter.to(self.device)   # buffers (perm/bias) follow
        self.unet = unet.to(self.device)
        self.watermarker = watermarker.to(self.device)

        # LatentPermuter has no trainable parameters; confirm this explicitly
        assert sum(p.numel() for p in self.permuter.parameters()) == 0, (
            "LatentPermuter must not have learnable parameters — "
            "only registered buffers."
        )

        # Joint optimiser: VAE + UNet + Watermarker
        trainable_params = (
            list(self.vae.parameters())
            + list(self.unet.parameters())
            + list(self.watermarker.parameters())
        )
        self.optimiser = optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Noise schedule
        self.schedule = LinearNoiseSchedule(
            config.diffusion_timesteps,
            config.beta_start,
            config.beta_end,
            self.device,
        )

        # Fixed target watermark w* ∈ {0,1}^M, moved to device
        from src.models.watermarker import generate_random_watermark
        w_star = generate_random_watermark(config.wm_bits, seed=config.wm_seed)
        self.w_star: torch.Tensor = w_star.to(self.device)   # shape (M,)

        # Tracking
        self.global_step: int = 0
        self.best_loss: float = float("inf")

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, x: torch.Tensor) -> dict[str, float]:
        """Execute one joint optimisation step on a batch x.

        Pipeline:
            x → VAE.encode → z → Permuter → z' → q_sample → z'_t
            z'_t + t → UNet → ε_θ          (diffusion arm)
            z'       → Watermarker → ŵ     (watermark arm)

        Loss:
            L_total = MSE(ε_θ, ε) + λ · BCE(ŵ, w*)

        Args:
            x: Image batch (B, C_x, H, W) on self.device, normalised to [0,1].

        Returns:
            Dict with scalar loss values for logging.
        """
        B = x.size(0)
        self.optimiser.zero_grad()

        # ── 1. Encode: x → z  ──────────────────────────────────────────
        # We use the reparameterised sample (not μ alone) to keep the VAE
        # encoder in a stochastic training regime.
        mu, logvar = self.vae.encode(x)                           # (B, C_z, 8, 8)
        z = self.vae.reparameterise(mu, logvar)                   # (B, C_z, 8, 8)

        # ── 2. Permute: z → z'  ────────────────────────────────────────
        # z' = π_K(z) + β_K   [Eq.(1) in latent_permute.py]
        # LatentPermuter.forward is a pure index-select + add with no
        # trainable weights, so ∂z'/∂z is a permutation matrix (full rank).
        # Gradients pass through without accumulation in permuter.parameters().
        z_prime = self.permuter(z)                                # (B, C_z, 8, 8)

        # ── 3. Watermark arm: z' → ŵ  ──────────────────────────────────
        w_hat = self.watermarker(z_prime)                         # (B, M)
        loss_wm = Watermarker.loss(w_hat, self.w_star)            # scalar  Eq.(2)

        # ── 4. Diffusion arm: z' → z'_t, predict ε  ───────────────────
        # Sample random timesteps t_i ~ Uniform{0, T−1} for each sample
        t = torch.randint(
            0, self.config.diffusion_timesteps, (B,), device=self.device
        )                                                         # (B,)

        # Forward diffusion: q(z'_t | z', t)   [Eq.(4)]
        # z_prime is detached here only from the *noise schedule* perspective;
        # we do NOT detach it so that UNet loss gradients flow back through
        # the permuter into the VAE encoder.
        z_prime_t, eps_true = self.schedule.q_sample(z_prime, t) # both (B,C,H,W)

        # UNet predicts the noise  ε_θ(z'_t, t)
        eps_pred = self.unet(z_prime_t, t)                        # (B, C_z, 8, 8)
        loss_diff = nn.functional.mse_loss(eps_pred, eps_true)    # scalar  Eq.(1)

        # ── 5. Optional VAE reconstruction regularisation  ─────────────
        # KL keeps the encoder posterior close to N(0,I) so z stays
        # well-structured for the permuter.
        loss_kl = self.vae.kl_divergence(mu, logvar)              # scalar

        # ── 6. Joint loss  L_total = L_diff + λ·L_wm + β·L_KL  ────────
        loss_total = (
            loss_diff
            + self.config.lambda_wm * loss_wm
            + self.vae.kl_weight * loss_kl
        )

        # ── 7. Backward + gradient clip + update  ──────────────────────
        loss_total.backward()

        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                self.optimiser.param_groups[0]["params"],
                self.config.grad_clip_norm,
            )

        self.optimiser.step()
        self.global_step += 1

        return {
            "loss_total": loss_total.item(),
            "loss_diff":  loss_diff.item(),
            "loss_wm":    loss_wm.item(),
            "loss_kl":    loss_kl.item(),
        }

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Train for one full epoch and return mean total loss.

        Args:
            loader: DataLoader yielding (image, label) batches.
            epoch:  Current epoch index (0-based), used for logging.

        Returns:
            Mean total loss over all batches.
        """
        self.vae.train()
        self.unet.train()
        self.watermarker.train()
        self.permuter.eval()   # no BN/Dropout in permuter; call for correctness

        total_loss_acc = 0.0
        num_batches = len(loader)

        for batch_idx, batch in enumerate(loader):
            # DataLoader may return (x, y) or just x depending on dataset
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device, non_blocking=True)

            metrics = self._train_step(x)
            total_loss_acc += metrics["loss_total"]

            if self.global_step % self.config.log_interval == 0:
                logger.info(
                    "Epoch %d  step %d/%d  "
                    "L_total=%.4f  L_diff=%.4f  L_wm=%.4f  L_kl=%.6f",
                    epoch + 1, batch_idx + 1, num_batches,
                    metrics["loss_total"], metrics["loss_diff"],
                    metrics["loss_wm"],    metrics["loss_kl"],
                )

        return total_loss_acc / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, loss: float, tag: str = "") -> str:
        """Serialise all model states, optimiser state and metadata.

        Args:
            epoch: Current epoch (0-based).
            loss:  Validation / training loss for bookkeeping.
            tag:   Optional suffix for the filename (e.g. "best").

        Returns:
            Absolute path to the saved checkpoint file.
        """
        suffix = f"_{tag}" if tag else f"_epoch{epoch+1:04d}"
        path = os.path.join(self.config.checkpoint_dir, f"ckpt{suffix}.pt")

        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "loss": loss,
                "vae_state": self.vae.state_dict(),
                "unet_state": self.unet.state_dict(),
                "watermarker_state": self.watermarker.state_dict(),
                # permuter has no learned params but save buffers for key audit
                "permuter_state": self.permuter.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "w_star": self.w_star.cpu(),
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)
        return path

    def load_checkpoint(self, path: str) -> int:
        """Restore all model states from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.

        Returns:
            Epoch index to resume from (0-based).
        """
        ckpt = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(ckpt["vae_state"])
        self.unet.load_state_dict(ckpt["unet_state"])
        self.watermarker.load_state_dict(ckpt["watermarker_state"])
        self.permuter.load_state_dict(ckpt["permuter_state"])
        self.optimiser.load_state_dict(ckpt["optimiser_state"])
        self.global_step = ckpt.get("global_step", 0)
        self.w_star = ckpt["w_star"].to(self.device)
        logger.info("Checkpoint loaded ← %s  (resume from epoch %d)", path, ckpt["epoch"] + 1)
        return ckpt["epoch"] + 1

    # ------------------------------------------------------------------
    # Main training entry-point
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        start_epoch: int = 0,
    ) -> None:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for the training split.
            start_epoch:  Epoch to begin at (0 for fresh run, or value
                          returned by load_checkpoint() for resumption).
        """
        logger.info(
            "Starting training: %d epochs, device=%s, λ_wm=%.3f",
            self.config.num_epochs, self.device, self.config.lambda_wm,
        )

        for epoch in range(start_epoch, self.config.num_epochs):
            mean_loss = self._run_epoch(train_loader, epoch)

            logger.info(
                "── Epoch %d/%d  mean loss = %.4f",
                epoch + 1, self.config.num_epochs, mean_loss,
            )

            # Save best checkpoint
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.save_checkpoint(epoch, mean_loss, tag="best")

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, mean_loss)

        logger.info("Training complete.  Best loss = %.4f", self.best_loss)

    # ------------------------------------------------------------------
    # Utility: extract target gradients (used by attack simulator)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_target_gradients(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute and return the gradients of L_total w.r.t. UNet parameters.

        This method is called by the attacker simulator to obtain the
        "leaked" gradients that the attacker attempts to invert.

        Args:
            x: A small image batch (B, C, H, W) — typically B=1.

        Returns:
            List of gradient tensors, one per UNet parameter, detached and
            cloned onto CPU to simulate a gradient-leakage scenario.
        """
        self.vae.eval()
        self.unet.eval()
        self.watermarker.eval()

        x = x.to(self.device)

        # Enable grad for a single forward-backward pass
        with torch.enable_grad():
            mu, logvar = self.vae.encode(x)
            z = self.vae.reparameterise(mu, logvar)
            z_prime = self.permuter(z)

            B = x.size(0)
            t = torch.randint(
                0, self.config.diffusion_timesteps, (B,), device=self.device
            )
            z_prime_t, eps_true = self.schedule.q_sample(z_prime, t)
            eps_pred = self.unet(z_prime_t, t)

            w_hat = self.watermarker(z_prime)
            loss_diff = nn.functional.mse_loss(eps_pred, eps_true)
            loss_wm = Watermarker.loss(w_hat, self.w_star)
            loss_total = loss_diff + self.config.lambda_wm * loss_wm

            # Compute grads only w.r.t. UNet params (what attacker observes)
            grads = torch.autograd.grad(
                loss_total,
                self.unet.parameters(),
                create_graph=False,
                allow_unused=True,
            )

        # Detach, clone to CPU, replace None with zero tensor
        result: list[torch.Tensor] = []
        for g, p in zip(grads, self.unet.parameters()):
            if g is None:
                result.append(torch.zeros_like(p).cpu())
            else:
                result.append(g.detach().clone().cpu())

        return result
