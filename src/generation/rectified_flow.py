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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample continuous time t ~ Uniform(0, 1).

    Returns:
        t: (B,) float tensor on `device`.
    """
    return torch.rand(batch_size, device=device)


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
) -> torch.Tensor:
    """Compute rectified flow training loss.

    Samples t ~ Uniform(0,1), eps ~ N(0,I), and computes
    MSE between predicted and target velocity.

    Args:
        model: FlowTransformer (or any module with forward(z_t, t, y) -> v).
        z:     Clean latent batch (B, C, H, W).
        y:     Optional class labels (B,).

    Returns:
        loss: Scalar MSE loss.
    """
    B = z.shape[0]
    device = z.device

    eps = torch.randn_like(z)
    t = sample_t(B, device)

    z_t = interpolate(z, eps, t)
    v_target = target_velocity(z, eps)

    v_pred = model(z_t, t, y)
    return F.mse_loss(v_pred, v_target)


def flow_loss_with_state(
    model: torch.nn.Module,
    z: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    eps: Optional[torch.Tensor] = None,
) -> dict:
    """Compute rectified flow training loss and return intermediate state.

    Same as flow_loss but returns a dict with all intermediate tensors for
    use in gradient-enabled training paths (e.g. TraceFlow).

    Determinism
    -----------
    ``t`` and ``eps`` may be supplied explicitly so that the *same* noise/time
    realisation can be reused across multiple calls.  This is essential for the
    inversion evaluation harness: target and dummy gradients must be computed
    under an identical ``(t, eps)`` realisation, otherwise the gradient-matching
    objective compares apples to oranges.

    - If ``t`` is provided it is used exactly (moved to ``z``'s device).
    - If ``eps`` is provided it is used exactly (moved to ``z``'s device).
    - If either is ``None`` it is freshly sampled (original behaviour).

    Args:
        model: FlowTransformer (or any module with forward(z_t, t, y) -> v).
        z:     Clean latent batch (B, C, H, W).
        y:     Optional class labels (B,).
        t:     Optional fixed time values (B,).  Sampled if ``None``.
        eps:   Optional fixed noise (B, C, H, W).  Sampled if ``None``.

    Returns:
        dict with keys:
            loss:     Scalar MSE loss.
            z_t:      Noisy latent at time t  (B, C, H, W).
            t:        Time values used         (B,).
            eps:      Noise used               (B, C, H, W).
            v_pred:   Predicted velocity       (B, C, H, W).
            v_target: Target velocity          (B, C, H, W).
            z_hat:    Estimated clean latent   (B, C, H, W).
                      z_hat = z_t - t * v_pred  (rectified-flow denoising estimate).
    """
    B = z.shape[0]
    device = z.device

    if eps is None:
        eps = torch.randn_like(z)
    else:
        eps = eps.to(device=device, dtype=z.dtype)

    if t is None:
        t = sample_t(B, device)
    else:
        t = t.to(device=device, dtype=z.dtype)

    z_t = interpolate(z, eps, t)
    v_target = target_velocity(z, eps)

    v_pred = model(z_t, t, y)
    loss = F.mse_loss(v_pred, v_target)

    z_hat = z_t - t.view(B, 1, 1, 1) * v_pred

    return {
        "loss": loss,
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
        v = model(z, t, y)          # predicted dz/dt (forward direction)
        z = z - dt * v              # reverse step: move against forward direction

    return z


@torch.no_grad()
def sample_euler_trajectory(
    model: torch.nn.Module,
    latent_shape: Tuple[int, int, int, int],
    steps: int,
    device: torch.device,
    y: Optional[torch.Tensor] = None,
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
        v = model(z, t, y)
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
) -> torch.Tensor:
    """Generate samples using the Heun (2nd-order Runge-Kutta) integrator.

    Args:
        model:        FlowTransformer.
        latent_shape: (B, C, H, W).
        steps:        Number of Heun steps.
        device:       Target device.
        y:            Optional class labels (B,).

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

        vel_start = model(z, t, y)
        z_pred = z - dt * vel_start
        vel_end = model(z_pred, t_next, y)
        z = z - dt * 0.5 * (vel_start + vel_end)

    return z
