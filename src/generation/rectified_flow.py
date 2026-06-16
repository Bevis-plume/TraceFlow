"""
src/generation/rectified_flow.py
==================================
Rectified flow training and sampling utilities for TraceFlow.

Math convention  (IMPORTANT — must remain consistent throughout the project)
---------------------------------------------------------------------------
Time axis:  t = 0  →  clean data (z_data)
            t = 1  →  pure noise (eps ~ N(0,I))

Forward path (noising, training):
    z_t = (1 - t) * z_data + t * eps

Target velocity (dz_t / dt along the *forward* direction):
    v* = d(z_t)/dt = eps - z_data

Training objective (MSE on velocity):
    L_flow = E_{t,z,eps}[ ||v_theta(z_t, t) - v*||^2 ]

Sampling (reverse: t goes from 1 → 0)
    The model predicts v = dz/dt in the FORWARD (data→noise) direction.
    To reverse, we step in the NEGATIVE t direction:

        z_{t - dt} = z_t - dt * v_theta(z_t, t)

    Starting from z_1 ~ N(0,I) and stepping dt > 0 until t ≈ 0.
    Each Euler step decreases t by dt = 1/steps.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


def _guided_velocity(
    model: torch.nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    y: Optional[torch.Tensor],
    *,
    guidance_scale: float = 1.0,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Predict velocity with optional classifier-free guidance.

    ``LabelEmbedder`` reserves class index ``num_classes`` as the null label.
    Training already drops labels into that slot; sampling needs to combine the
    conditional and null predictions explicitly to use the learned CFG branch.
    """
    scale = float(guidance_scale)
    if y is None or num_classes is None or abs(scale - 1.0) < 1e-8:
        return model(z, t, y)

    null_y = torch.full_like(y, int(num_classes))
    z_in = torch.cat([z, z], dim=0)
    t_in = torch.cat([t, t], dim=0)
    y_in = torch.cat([y, null_y], dim=0)
    v_cond, v_uncond = model(z_in, t_in, y_in).chunk(2, dim=0)
    return v_uncond + scale * (v_cond - v_uncond)


def sample_t(
    batch_size: int,
    device: torch.device,
    *,
    strategy: str = "uniform",
    high_t_prob: float = 0.0,
    high_t_min: float = 0.6,
) -> torch.Tensor:
    """Sample continuous time t in [0, 1].

    ``mixed_high`` keeps part of each batch in the high-noise regime. This
    prevents the model from minimizing the easy low/mid-t denoising objective
    while failing to learn the structure-forming velocity near t=1.
    """
    strategy = str(strategy or "uniform")
    if strategy == "uniform":
        return torch.rand(batch_size, device=device)
    if strategy != "mixed_high":
        raise ValueError(f"Unknown flow t_sampling strategy {strategy!r}")

    high_t_prob = float(max(0.0, min(1.0, high_t_prob)))
    high_t_min = float(max(0.0, min(1.0, high_t_min)))
    t = torch.rand(batch_size, device=device)
    if high_t_prob <= 0.0:
        return t
    high_mask = torch.rand(batch_size, device=device) < high_t_prob
    high = high_t_min + (1.0 - high_t_min) * torch.rand(batch_size, device=device)
    return torch.where(high_mask, high, t)


def _flow_options(flow_cfg: Optional[Mapping[str, Any]]) -> dict[str, float | str]:
    cfg = dict(flow_cfg or {})
    return {
        "t_sampling": str(cfg.get("t_sampling", "uniform")),
        "high_t_prob": float(cfg.get("high_t_prob", 0.0)),
        "high_t_min": float(cfg.get("high_t_min", 0.6)),
        "x0_loss_weight": float(cfg.get("x0_loss_weight", 0.0)),
        "x0_t_power": float(cfg.get("x0_t_power", 1.0)),
        "velocity_t_weight_scale": float(cfg.get("velocity_t_weight_scale", 0.0)),
    }


def interpolate(
    z: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation between clean latent and noise.

    z_t = (1 - t) * z + t * eps

    Args:
        z:   Clean latent  (B, C, H, W).
        eps: Noise sample  (B, C, H, W).
        t:   Time          (B,) in [0, 1].

    Returns:
        z_t: Noisy latent  (B, C, H, W).
    """
    # Broadcast t: (B,) -> (B, 1, 1, 1)
    t_ = t.view(-1, 1, 1, 1)
    return (1.0 - t_) * z + t_ * eps


def target_velocity(z: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Compute target velocity for rectified flow.

    v* = eps - z

    Args:
        z:   Clean latent  (B, C, H, W).
        eps: Noise sample  (B, C, H, W).

    Returns:
        v_target: (B, C, H, W).
    """
    return eps - z


def flow_loss(
    model: torch.nn.Module,
    z: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    flow_cfg: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Compute rectified flow training loss."""
    return flow_loss_with_state(model, z, y=y, flow_cfg=flow_cfg)["loss"]


def flow_loss_with_state(
    model: torch.nn.Module,
    z: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    eps: Optional[torch.Tensor] = None,
    flow_cfg: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Compute rectified flow loss and return intermediate state.

    If ``flow_cfg`` is omitted, this keeps the original velocity-only uniform-t
    objective. Passing ``flow_cfg`` enables high-noise t sampling, velocity
    weighting, and optional x0/clean-latent supervision. Explicit ``t`` and
    ``eps`` are still honored exactly for inversion-gradient determinism.
    """
    B = z.shape[0]
    device = z.device
    opts = _flow_options(flow_cfg)

    if eps is None:
        eps = torch.randn_like(z)
    else:
        eps = eps.to(device=device, dtype=z.dtype)

    if t is None:
        t = sample_t(
            B,
            device,
            strategy=str(opts["t_sampling"]),
            high_t_prob=float(opts["high_t_prob"]),
            high_t_min=float(opts["high_t_min"]),
        )
    else:
        t = t.to(device=device, dtype=z.dtype)

    z_t = interpolate(z, eps, t)
    v_target = target_velocity(z, eps)
    v_pred = model(z_t, t, y)

    reduce_dims = tuple(range(1, v_pred.ndim))
    velocity_per = (v_pred - v_target).pow(2).mean(dim=reduce_dims)
    velocity_weight = 1.0 + float(opts["velocity_t_weight_scale"]) * t.float()
    loss_velocity = (velocity_per * velocity_weight).mean() / velocity_weight.mean().detach().clamp_min(1e-6)

    z_hat = z_t - t.view(B, 1, 1, 1) * v_pred
    x0_per = (z_hat - z).pow(2).mean(dim=reduce_dims)
    x0_weight = float(opts["x0_loss_weight"]) * t.float().clamp_min(0.0).pow(float(opts["x0_t_power"]))
    loss_x0 = (x0_per * x0_weight).mean()
    loss = loss_velocity + loss_x0

    high_t_min = float(opts["high_t_min"])
    high_t_fraction = (t.float() >= high_t_min).float().mean()

    return {
        "loss": loss,
        "loss_velocity": loss_velocity,
        "loss_velocity_raw": velocity_per.mean(),
        "loss_x0": loss_x0,
        "loss_x0_raw": x0_per.mean(),
        "t_mean": t.float().mean(),
        "high_t_fraction": high_t_fraction,
        "z_t": z_t,
        "t": t,
        "eps": eps,
        "v_pred": v_pred,
        "v_target": v_target,
        "z_hat": z_hat,
    }


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    latent_shape: Tuple[int, int, int, int],
    steps: int,
    device: torch.device,
    y: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Generate samples using the Euler integrator.

    Integrates the learned velocity field from t=1 (pure noise) to t=0 (data).

    Convention:
        v_theta predicts dz/dt in the FORWARD (noise-ward) direction.
        Reverse sampling subtracts dt * v at each step:
            z_{t-dt} = z_t - dt * v_theta(z_t, t)

    Args:
        model:        FlowTransformer (or any nn.Module with forward(z, t, y) -> v).
        latent_shape: (B, C, H, W).
        steps:        Number of Euler steps (higher = better quality).
        device:       Target device.
        y:            Optional class labels (B,).
        guidance_scale: Classifier-free guidance scale. 1.0 preserves old behavior.
        num_classes:  Number of real classes; the null label is this index.

    Returns:
        z_0: Estimated clean latent (B, C, H, W).
    """
    model.eval()
    B = latent_shape[0]

    # Start at t=1: pure Gaussian noise
    z = torch.randn(*latent_shape, device=device)

    # Step from t=1 down to t=0 in `steps` equal increments.
    # t schedule: [1, 1-dt, 1-2dt, ..., dt]  (steps values; last step lands at t=dt≈0)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = 1.0 - i * dt        # t: 1 → dt
        t = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v = _guided_velocity(
            model,
            z,
            t,
            y,
            guidance_scale=guidance_scale,
            num_classes=num_classes,
        )                           # predicted dz/dt (forward direction)
        z = z - dt * v              # reverse step: move against forward direction

    return z


@torch.no_grad()
def sample_euler_trajectory(
    model: torch.nn.Module,
    latent_shape: Tuple[int, int, int, int],
    steps: int,
    device: torch.device,
    y: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    num_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate samples and keep the latent trajectory.

    Returns:
        z_0: Final estimated clean latent.
        trajectory: Tensor of shape ``(steps + 1, B, C, H, W)`` containing the
            initial noise latent and every Euler update. This is intended for
            lightweight visualisation, not for training.
    """
    model.eval()
    B = latent_shape[0]
    z = torch.randn(*latent_shape, device=device)
    traj = [z.detach().cpu()]
    dt = 1.0 / steps
    for i in range(steps):
        t_val = 1.0 - i * dt
        t = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v = _guided_velocity(
            model,
            z,
            t,
            y,
            guidance_scale=guidance_scale,
            num_classes=num_classes,
        )
        z = z - dt * v
        traj.append(z.detach().cpu())
    return z, torch.stack(traj, dim=0)


@torch.no_grad()
def sample_heun(
    model: torch.nn.Module,
    latent_shape: Tuple[int, int, int, int],
    steps: int,
    device: torch.device,
    y: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Generate samples using the Heun (2nd-order Runge-Kutta) integrator.

    Args:
        model:        FlowTransformer.
        latent_shape: (B, C, H, W).
        steps:        Number of Heun steps.
        device:       Target device.
        y:            Optional class labels (B,).
        guidance_scale: Classifier-free guidance scale. 1.0 preserves old behavior.
        num_classes:  Number of real classes; the null label is this index.

    Returns:
        z_0: Generated clean latent (B, C, H, W).
    """
    model.eval()
    B = latent_shape[0]
    z = torch.randn(*latent_shape, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t_val = 1.0 - i * dt
        t_next_val = t_val - dt

        t = torch.full((B,), t_val, device=device, dtype=torch.float32)
        t_next = torch.full((B,), max(t_next_val, 0.0), device=device, dtype=torch.float32)

        vel_start = _guided_velocity(
            model,
            z,
            t,
            y,
            guidance_scale=guidance_scale,
            num_classes=num_classes,
        )
        z_pred = z - dt * vel_start
        vel_end = _guided_velocity(
            model,
            z_pred,
            t_next,
            y,
            guidance_scale=guidance_scale,
            num_classes=num_classes,
        )
        z = z - dt * 0.5 * (vel_start + vel_end)

    return z
