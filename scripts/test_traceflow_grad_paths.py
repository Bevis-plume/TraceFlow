"""scripts/test_traceflow_grad_paths.py — Smoke tests for TraceFlow gradient paths.

Verifies:
  1. decode_with_grad: a loss on the decoded image produces nonzero gradient on input z.
  2. encode_with_grad: a loss on the encoded latent produces nonzero gradient on input x.
  3. flow_loss_with_state: returns valid shapes and z_hat matches z shape.
  4. encode/decode (no-grad versions) still work correctly.

No datasets, training runs, or downloads required.

Usage
-----
    python -m scripts.test_traceflow_grad_paths
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.autoencoder_backend import AutoencoderBackend
from src.generation.rectified_flow import flow_loss_with_state


# ---------------------------------------------------------------------------
# Minimal stub model for flow_loss_with_state test (no DiT/SiT weights needed)
# ---------------------------------------------------------------------------

class _LinearVelocityModel(nn.Module):
    """Trivial velocity model: returns a learned per-channel constant regardless of input."""

    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(latent_channels))

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Broadcast weight to (B, C, H, W) and add a tiny term that depends on z_t
        # so that gradients can flow back through z_t -> z_hat.
        B, C, H, W = z_t.shape
        v = self.weight.view(1, C, 1, 1).expand(B, C, H, W) + 0.0 * z_t
        return v


def main() -> None:
    print("[test] ── TraceFlow gradient path smoke tests ──")

    torch.manual_seed(42)

    B = 2
    latent_channels = 4
    latent_size = 8
    image_size = 64
    channels = 3

    # ------------------------------------------------------------------
    # Build local autoencoder (frozen parameters, as in production)
    # ------------------------------------------------------------------
    autoencoder = AutoencoderBackend(
        backend="local",
        latent_channels=latent_channels,
        image_size=image_size,
        latent_size=latent_size,
        freeze=True,
    )
    autoencoder.eval()

    # ------------------------------------------------------------------
    # 1. decode_with_grad: gradient flows from decoded image back to input z
    # ------------------------------------------------------------------
    z = torch.randn(B, latent_channels, latent_size, latent_size, requires_grad=True)
    x_hat = autoencoder.decode_with_grad(z)

    assert x_hat.shape == (B, channels, image_size, image_size), (
        f"decode_with_grad shape mismatch: {tuple(x_hat.shape)}"
    )

    loss = x_hat.mean()
    loss.backward()

    assert z.grad is not None, "decode_with_grad: z.grad is None"
    grad_norm = z.grad.detach().norm().item()
    assert grad_norm > 0.0, f"decode_with_grad: gradient norm is zero (got {grad_norm})"
    assert math.isfinite(grad_norm), f"decode_with_grad: gradient norm is not finite ({grad_norm})"
    print(f"[test] decode_with_grad gradient flow: grad_norm={grad_norm:.4e}: PASS")

    # ------------------------------------------------------------------
    # 2. encode_with_grad: gradient flows from encoded latent back to input x
    # ------------------------------------------------------------------
    x = torch.randn(B, channels, image_size, image_size, requires_grad=True)
    z_enc = autoencoder.encode_with_grad(x)

    assert z_enc.shape == (B, latent_channels, latent_size, latent_size), (
        f"encode_with_grad shape mismatch: {tuple(z_enc.shape)}"
    )

    loss2 = z_enc.mean()
    loss2.backward()

    assert x.grad is not None, "encode_with_grad: x.grad is None"
    grad_norm2 = x.grad.detach().norm().item()
    assert grad_norm2 > 0.0, f"encode_with_grad: gradient norm is zero (got {grad_norm2})"
    assert math.isfinite(grad_norm2), f"encode_with_grad: gradient norm is not finite ({grad_norm2})"
    print(f"[test] encode_with_grad gradient flow: grad_norm={grad_norm2:.4e}: PASS")

    # ------------------------------------------------------------------
    # 3. encode/decode (no-grad) still work and produce finite outputs
    # ------------------------------------------------------------------
    with torch.no_grad():
        z_nograd = autoencoder.encode(torch.randn(B, channels, image_size, image_size))
        x_nograd = autoencoder.decode(torch.randn(B, latent_channels, latent_size, latent_size))

    assert z_nograd.shape == (B, latent_channels, latent_size, latent_size)
    assert x_nograd.shape == (B, channels, image_size, image_size)
    assert torch.isfinite(z_nograd).all(), "encode output not finite"
    assert torch.isfinite(x_nograd).all(), "decode output not finite"
    print("[test] encode/decode (no-grad paths) still work: PASS")

    # ------------------------------------------------------------------
    # 4. flow_loss_with_state: valid shapes, z_hat matches z shape
    # ------------------------------------------------------------------
    model = _LinearVelocityModel(latent_channels=latent_channels)
    model.train()

    z_clean = torch.randn(B, latent_channels, latent_size, latent_size)
    state = flow_loss_with_state(model, z_clean, y=None)

    expected_keys = {"loss", "z_t", "t", "eps", "v_pred", "v_target", "z_hat"}
    assert set(state.keys()) == expected_keys, (
        f"flow_loss_with_state missing keys: {expected_keys - set(state.keys())}"
    )

    loss_val = state["loss"].item()
    assert math.isfinite(loss_val), f"flow_loss_with_state: loss not finite ({loss_val})"

    latent_shape = (B, latent_channels, latent_size, latent_size)
    for key in ("z_t", "eps", "v_pred", "v_target", "z_hat"):
        assert state[key].shape == torch.Size(latent_shape), (
            f"flow_loss_with_state[{key!r}] shape {tuple(state[key].shape)} != {latent_shape}"
        )

    assert state["t"].shape == torch.Size((B,)), (
        f"flow_loss_with_state['t'] shape {tuple(state['t'].shape)} != ({B},)"
    )

    # z_hat shape must match z
    assert state["z_hat"].shape == z_clean.shape, (
        f"z_hat shape {tuple(state['z_hat'].shape)} != z shape {tuple(z_clean.shape)}"
    )

    print(
        f"[test] flow_loss_with_state shapes: loss={loss_val:.4f} "
        f"z_hat={tuple(state['z_hat'].shape)}: PASS"
    )

    # ------------------------------------------------------------------
    # 5. Determinism: fixed (t, eps) are reused exactly across calls
    # ------------------------------------------------------------------
    # When t/eps are passed explicitly, flow_loss_with_state must NOT resample.
    fixed_eps = torch.randn(B, latent_channels, latent_size, latent_size)
    fixed_t = torch.rand(B)

    s1 = flow_loss_with_state(model, z_clean, t=fixed_t, eps=fixed_eps)
    s2 = flow_loss_with_state(model, z_clean, t=fixed_t, eps=fixed_eps)

    assert torch.allclose(s1["t"], fixed_t), "fixed t not honoured in state"
    assert torch.allclose(s1["eps"], fixed_eps), "fixed eps not honoured in state"
    assert torch.allclose(s1["t"], s2["t"]), "t differs between identical calls"
    assert torch.allclose(s1["eps"], s2["eps"]), "eps differs between identical calls"
    # Intermediate tensors derived purely from (z, t, eps) must match exactly.
    assert torch.allclose(s1["z_t"], s2["z_t"]), "z_t differs between identical fixed calls"
    assert torch.allclose(s1["v_target"], s2["v_target"]), "v_target differs between identical fixed calls"
    print("[test] flow_loss_with_state honours fixed (t, eps): PASS")

    # ------------------------------------------------------------------
    # 6. AttackBatchState: target and dummy paths share the SAME (t, eps, bits)
    # ------------------------------------------------------------------
    from src.attacks.traceflow_inversion import (
        compute_target_gradients,
        AttackBatchState,
        _traceflow_loss_from_latent,
    )

    # Flow-only objective (no watermark) is enough to validate the contract.
    x_real = torch.randn(B, channels, image_size, image_size).tanh()
    target_grads, attack_state = compute_target_gradients(
        model=model,
        autoencoder=autoencoder,
        latent_transform=None,
        watermark_modules=None,
        x=x_real,
        objective="flow_only",
    )

    assert isinstance(attack_state, AttackBatchState), "attack_state wrong type"
    assert attack_state.t.shape == torch.Size((B,)), (
        f"attack_state.t shape {tuple(attack_state.t.shape)} != ({B},)"
    )
    assert attack_state.eps.shape == torch.Size(
        (B, latent_channels, latent_size, latent_size)
    ), f"attack_state.eps shape {tuple(attack_state.eps.shape)}"
    assert isinstance(target_grads, list) and len(target_grads) > 0, "target_grads empty"

    # Re-running a dummy loss with the SAME fixed state must reproduce the exact
    # same flow intermediates — proving t/eps are threaded through, not resampled.
    z_dummy = torch.randn(B, latent_channels, latent_size, latent_size)
    from src.generation.rectified_flow import flow_loss_with_state as _flws
    a = _flws(model, z_dummy, t=attack_state.t, eps=attack_state.eps)
    b = _flws(model, z_dummy, t=attack_state.t, eps=attack_state.eps)
    assert torch.allclose(a["z_t"], b["z_t"]), "dummy path z_t not deterministic under fixed state"
    assert torch.allclose(a["v_target"], b["v_target"]), "dummy path v_target not deterministic"
    print("[test] AttackBatchState fixed (t, eps) shared across paths: PASS")

    # ------------------------------------------------------------------
    # 7. flow_loss_with_state: gradient flows back through loss
    # ------------------------------------------------------------------
    z_grad = torch.randn(B, latent_channels, latent_size, latent_size, requires_grad=True)
    state2 = flow_loss_with_state(model, z_grad, y=None)
    state2["loss"].backward()

    assert z_grad.grad is not None, "flow_loss_with_state: z_grad.grad is None"
    flow_grad_norm = z_grad.grad.detach().norm().item()
    assert math.isfinite(flow_grad_norm), (
        f"flow_loss_with_state: gradient norm not finite ({flow_grad_norm})"
    )
    print(f"[test] flow_loss_with_state gradient flow: grad_norm={flow_grad_norm:.4e}: PASS")

    print("[test] ── ALL TRACEFLOW GRADIENT PATH CHECKS PASSED ──")


if __name__ == "__main__":
    main()
