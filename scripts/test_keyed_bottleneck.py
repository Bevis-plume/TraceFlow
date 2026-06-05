"""scripts/test_keyed_bottleneck.py — Invertibility and correctness tests.

No datasets, training, or downloads required.

Usage
-----
    python -m scripts.test_keyed_bottleneck
"""

from __future__ import annotations

import torch
from src.security.keyed_bottleneck import KeyedLatentBottleneck


def main() -> None:
    print("[test] ── KeyedLatentBottleneck smoke test ──")

    # Smoke-mode latent dimensions: C=4, H=W=8 → D=256; 256/16=16 blocks
    latent_channels = 4
    latent_size     = 8
    block_size      = 16
    bias_scale      = 0.1
    B               = 4

    # ------------------------------------------------------------------
    # 1. Invertibility
    # ------------------------------------------------------------------
    klb   = KeyedLatentBottleneck(
        secret_key="test_key_abc123",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=block_size,
        bias_scale=bias_scale,
    )
    z     = torch.randn(B, latent_channels, latent_size, latent_size)
    z_k   = klb(z)
    z_rec = klb.invert(z_k)

    max_err  = (z - z_rec).abs().max().item()
    mean_err = (z - z_rec).abs().mean().item()
    print(f"[test] max_abs_error:  {max_err:.3e}")
    print(f"[test] mean_abs_error: {mean_err:.3e}")
    assert max_err < 1e-5, (
        f"Reconstruction error too large: {max_err:.3e}  (expected < 1e-5)"
    )
    print("[test] Invertibility: PASS")

    # ------------------------------------------------------------------
    # 2. Different keys → different transforms
    # ------------------------------------------------------------------
    klb2     = KeyedLatentBottleneck(
        secret_key="different_key_xyz789",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=block_size,
        bias_scale=bias_scale,
    )
    z_k2     = klb2(z)
    key_diff = (z_k - z_k2).abs().max().item()
    print(f"[test] Max diff key1 vs key2: {key_diff:.4f}")
    assert key_diff > 0.01, f"Different keys should produce different transforms; got {key_diff:.3e}"
    print("[test] Key sensitivity: PASS")

    # ------------------------------------------------------------------
    # 3. Same key → identical transform (determinism)
    # ------------------------------------------------------------------
    klb3      = KeyedLatentBottleneck(
        secret_key="test_key_abc123",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=block_size,
        bias_scale=bias_scale,
    )
    z_k3      = klb3(z)
    same_diff = (z_k - z_k3).abs().max().item()
    print(f"[test] Max diff same key (must be 0): {same_diff:.3e}")
    assert same_diff == 0.0, f"Same key must produce identical transform; got {same_diff:.3e}"
    print("[test] Key determinism: PASS")

    # ------------------------------------------------------------------
    # 4. Transform is not the identity (z_k ≠ z)
    # ------------------------------------------------------------------
    mag = (z_k - z).abs().mean().item()
    print(f"[test] Mean |z_k - z| (must be > 0): {mag:.4f}")
    assert mag > 0.0, "Transform should not be the identity"
    print("[test] Non-trivial transform: PASS")

    # ------------------------------------------------------------------
    # 5. Bias-free mode (bias_scale=0)
    # ------------------------------------------------------------------
    klb_nb  = KeyedLatentBottleneck(
        secret_key="test_key_abc123",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=block_size,
        bias_scale=0.0,
    )
    z_k_nb  = klb_nb(z)
    z_r_nb  = klb_nb.invert(z_k_nb)
    err_nb  = (z - z_r_nb).abs().max().item()
    print(f"[test] Bias-free max_abs_error: {err_nb:.3e}")
    assert err_nb < 1e-5, f"Bias-free reconstruction error too large: {err_nb:.3e}"
    print("[test] Bias-free invertibility: PASS")

    # ------------------------------------------------------------------
    # 6. Buffers are non-persistent → NOT saved to checkpoint state_dict
    # ------------------------------------------------------------------
    sd = klb.state_dict()
    assert "W"    not in sd, "W should NOT appear in state_dict (persistent=False)"
    assert "beta" not in sd, "beta should NOT appear in state_dict (persistent=False)"
    print("[test] W and beta absent from state_dict: PASS")

    print("\n[test] All 6 tests passed — KeyedLatentBottleneck is correct.")


if __name__ == "__main__":
    main()
