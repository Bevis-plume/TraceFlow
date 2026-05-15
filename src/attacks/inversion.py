"""
src/attacks/inversion.py
========================
Gradient-inversion attack simulator (DLG / gradient-matching).

Threat model
------------
The attacker:
  • Has white-box access to the *complete* network graph, including the
    LatentPermuter layer (weights/buffers are visible).
  • Does NOT know the secret key K, so:
      - cannot call LatentPermuter.invert()
      - does not know that π_K is a permutation — treats it as an opaque layer
  • Observes a gradient signal ∇_{θ_UNet} L_total leaked from one training step.
  • Goal: recover the original training image x by optimising a dummy latent
    ẑ_dummy until its gradient matches the observed gradient.

Attack algorithm
----------------
Initialise  ẑ_dummy in the pixel domain and optimise it directly in
[0, 1] with a sigmoid parameterisation.
Repeat for max_iter steps:
    1. z_dummy = VAE.encode(x_dummy).sample
    2. z'_dummy = LatentPermuter.forward(z_dummy)
    3. z'_dummy_t = q_sample(z'_dummy, t_fixed, fixed_noise)
    4. ε_pred = UNet(z'_dummy_t, t_fixed)
    5. ŵ_dummy = Watermarker(z'_dummy)
    6. L_match = MSE(∇_θ(L_dummy), target_grads)          (gradient-matching loss)
    7. Back-prop through L_match → update ẑ_dummy

Output
------
x_reconstructed = VAE.decode(ẑ_dummy)
Because ẑ_dummy ≈ z'  (the *permuted* representation), decoding without
π_K^{-1} produces semantically incoherent images — confirming the defence.

Reference
---------
Zhao et al., "iDLG: Improved Deep Leakage from Gradients", arXiv 2001.02610.
Geiping et al., "Inverting Gradients — How easy is it to break privacy in
federated learning?", NeurIPS 2020.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.crypto.latent_permute import LatentPermuter
from src.models.vae import VAE
from src.models.unet import UNet
from src.models.watermarker import Watermarker
from src.pipeline.trainer import LinearNoiseSchedule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AttackConfig:
    """Hyper-parameters for the gradient-inversion attack.

    Args:
        optimizer:    "lbfgs" or "adam".
        max_iter:     Maximum number of optimiser iterations.
        lr:           Learning rate for the dummy variable.
        dummy_init:   "random" (Gaussian) or "zeros".
        tv_weight:    Total-variation regularisation weight on x_dummy
                      (0 disables; helps avoid extreme pixel values).
        verbose:      Log progress every N steps (0 = silent).
        diffusion_timesteps: T for noise schedule (must match trainer).
        beta_start:   β_1.
        beta_end:     β_T.
        attack_t:     Fixed timestep used during attack (default 1, near-clean).
        latent_spatial: Spatial size of the latent feature map.
        wm_bits:      Watermark output dimension (must match Watermarker).
        lambda_wm:    λ used in the original training loss (must match trainer).
    """
    optimizer: str = "lbfgs"
    max_iter: int = 300
    lr: float = 0.01
    dummy_init: str = "random"
    tv_weight: float = 1e-4
    verbose: int = 50
    diffusion_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    attack_t: int = 1
    latent_spatial: int = 8
    wm_bits: int = 64
    lambda_wm: float = 1.0


# ---------------------------------------------------------------------------
# Gradient-matching loss helper
# ---------------------------------------------------------------------------

def _gradient_matching_loss(
    dummy_grads: tuple[Optional[torch.Tensor], ...],
    target_grads: list[torch.Tensor],
) -> torch.Tensor:
    """Cosine-similarity gradient-matching loss (Geiping et al., 2020).

    L_match = 1 − cos( vec(∇_dummy), vec(∇_target) )

    Cosine similarity is more robust than plain MSE because it is invariant
    to the magnitude of the gradient signal.

    Args:
        dummy_grads:  Gradients from current forward pass (may contain None).
        target_grads: Observed (leaked) gradients, on the same device.

    Returns:
        Scalar loss in [0, 2].
    """
    device = target_grads[0].device
    vec_dummy_parts = []
    for g, t in zip(dummy_grads, target_grads):
        if g is None:
            vec_dummy_parts.append(torch.zeros_like(t).reshape(-1))
        else:
            vec_dummy_parts.append(g.reshape(-1))
    vec_dummy = torch.cat(vec_dummy_parts)
    vec_target = torch.cat([g.to(device).reshape(-1) for g in target_grads])
    # Cosine distance: 0 = perfectly aligned; 2 = opposite
    cos_sim = nn.functional.cosine_similarity(
        vec_dummy.unsqueeze(0), vec_target.unsqueeze(0)
    )
    return 1.0 - cos_sim


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV regularisation on a batch of images.

    Args:
        x: Image tensor (B, C, H, W).

    Returns:
        Scalar TV loss.
    """
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


# ---------------------------------------------------------------------------
# Main attack class
# ---------------------------------------------------------------------------

class GradientInversionAttack:
    """Simulation of a gradient-matching inversion attack against the
    LatentPermuter defence.

    The attacker runs the full defended pipeline (VAE → Permuter → UNet +
    Watermarker) but cannot undo the permutation because key K is unknown.
    The recovered image is therefore visually incoherent.

    Args:
        vae:         The target VAE (encoder + decoder).
        permuter:    The target LatentPermuter (attacker has its buffers but
                     not the key K, so invert() is unavailable to them).
        unet:        The target UNet.
        watermarker: The target Watermarker.
        config:      AttackConfig hyper-parameters.
        device:      Torch device string.
    """

    def __init__(
        self,
        vae: VAE,
        permuter: LatentPermuter,
        unet: UNet,
        watermarker: Watermarker,
        config: AttackConfig,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.config = config

        self.vae = vae.to(self.device).eval()
        self.permuter = permuter.to(self.device).eval()
        self.unet = unet.to(self.device).eval()
        self.watermarker = watermarker.to(self.device).eval()

        self.schedule = LinearNoiseSchedule(
            config.diffusion_timesteps,
            config.beta_start,
            config.beta_end,
            self.device,
        )

        # Fixed w* placeholder (attacker assumes a zero watermark target)
        self.w_dummy_target = torch.zeros(config.wm_bits, device=self.device)

        # Pre-compute a fixed noise vector ε for the attack timestep
        # (kept constant so the matching loss is smooth across iterations)
        self._fixed_t = torch.tensor(
            [config.attack_t], device=self.device
        )

    # ------------------------------------------------------------------
    # Internal: compute grad-matching loss for one dummy state
    # ------------------------------------------------------------------

    def _compute_dummy_grads(
        self,
        x_dummy: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[Optional[torch.Tensor], ...]]:
        """Forward-backward pass through the defended pipeline on x_dummy.

        Args:
            x_dummy: Dummy pixel image (1, C_x, H, W), requires_grad=False.
                     Gradient is carried via x_dummy itself when it is a
                     leaf with requires_grad=True.

        Returns:
            loss_dummy: Combined loss (diffusion + wm) for x_dummy.
            dummy_grads: Gradient tuple w.r.t. UNet parameters.
        """
        mu, logvar = self.vae.encode(x_dummy)
        z_dummy = self.vae.reparameterise(mu, logvar)

        # Attacker runs permuter as a black-box forward pass
        z_prime_dummy = self.permuter(z_dummy)

        # Fixed timestep for stable gradient landscape
        z_prime_dummy_t, eps_true = self.schedule.q_sample(
            z_prime_dummy, t, noise=noise
        )
        eps_pred = self.unet(z_prime_dummy_t, t)

        w_hat = self.watermarker(z_prime_dummy)

        loss_diff = nn.functional.mse_loss(eps_pred, eps_true)
        loss_wm = nn.functional.binary_cross_entropy(
            w_hat, self.w_dummy_target.expand_as(w_hat)
        )
        loss_dummy = loss_diff + self.config.lambda_wm * loss_wm

        dummy_grads = torch.autograd.grad(
            loss_dummy,
            self.unet.parameters(),
            create_graph=True,    # must be True for grad of grad (L-BFGS)
            allow_unused=True,
        )
        return loss_dummy, dummy_grads

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        target_grads: list[torch.Tensor],
        image_shape: tuple[int, int, int, int],
        fixed_noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Execute the gradient-inversion attack.

        Args:
            target_grads: Leaked gradients of L_total w.r.t. UNet parameters,
                          as a list of CPU tensors (returned by
                          Trainer.get_target_gradients).
            image_shape:  Shape of the dummy image to optimise: (B, C, H, W).
            fixed_noise:  Optional pre-sampled noise for q_sample (for
                          reproducibility).  If None, sampled internally.

        Returns:
            Dict with:
              "x_reconstructed": Decoded dummy image (B, C, H, W) in [0,1].
              "z_prime_dummy":   Final permuted latent (B, C_z, H_z, W_z).
              "final_loss":      Scalar tensor with final matching loss.
        """
        # Move target gradients to attack device
        target_grads_dev = [g.to(self.device) for g in target_grads]

        # ── Initialise dummy image in a stable [0,1] range ────────────
        if self.config.dummy_init == "random":
            x_init = torch.rand(image_shape, device=self.device)
        else:
            x_init = torch.full(image_shape, 0.5, device=self.device)

        # Optimise in logit space: x = sigmoid(x_logit)
        x_logit = torch.logit(x_init.clamp(1e-4, 1 - 1e-4)).detach().requires_grad_(True)

        if fixed_noise is None:
            fixed_noise = torch.randn(
                image_shape[0],
                self.vae.encoder.mu_head.out_channels,
                self.config.latent_spatial,
                self.config.latent_spatial,
                device=self.device,
            )
        fixed_noise = fixed_noise.to(self.device)
        t = self._fixed_t.expand(image_shape[0])

        # ── Choose optimiser ───────────────────────────────────────────
        if self.config.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(
                [x_logit],
                lr=self.config.lr,
                max_iter=self.config.max_iter,
                history_size=100,
                line_search_fn="strong_wolfe",
            )
        else:
            optimizer = torch.optim.Adam(
                [x_logit],
                lr=self.config.lr,
            )

        final_loss = torch.tensor(0.0)
        step = [0]   # mutable int for closure capture

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            x_candidate = torch.sigmoid(x_logit)            # constrained to (0,1)

            _, dummy_grads = self._compute_dummy_grads(x_candidate, t=t, noise=fixed_noise)

            loss_match = _gradient_matching_loss(dummy_grads, target_grads_dev)
            loss_tv    = _total_variation(x_candidate)
            loss_opt   = loss_match + self.config.tv_weight * loss_tv

            loss_opt.backward()
            nonlocal final_loss
            final_loss = loss_opt.detach()

            if self.config.verbose > 0 and step[0] % self.config.verbose == 0:
                logger.info(
                    "Attack step %d  L_match=%.4f  L_tv=%.6f",
                    step[0], loss_match.item(), loss_tv.item(),
                )
            step[0] += 1
            return loss_opt

        # ── Optimise ───────────────────────────────────────────────────
        if self.config.optimizer == "lbfgs":
            # L-BFGS calls closure multiple times internally
            optimizer.step(closure)
        else:
            for _ in range(self.config.max_iter):
                optimizer.step(closure)

        # ── Decode recovered dummy ─────────────────────────────────────
        with torch.no_grad():
            x_final = torch.sigmoid(x_logit)                 # (B, C_x, H, W)
            mu_final, logvar_final = self.vae.encode(x_final)
            z_dummy_final = self.vae.reparameterise(mu_final, logvar_final)

            # Attacker decodes WITHOUT inversion — expected to be noise
            z_prime_dummy_final = self.permuter(z_dummy_final)
            x_reconstructed = self.vae.decode(z_prime_dummy_final)

        logger.info(
            "Attack finished.  Final L_opt=%.4f.  "
            "Decoded image is semantically incoherent (expected).",
            final_loss.item(),
        )

        return {
            "x_reconstructed": x_reconstructed.detach().cpu(),
            "z_prime_dummy":   z_prime_dummy_final.detach().cpu(),
            "final_loss":      final_loss.cpu(),
            "x_dummy_final":   x_final.detach().cpu(),
        }
