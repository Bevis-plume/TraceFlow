"""scripts/test_keyed_bottleneck.py — Invertibility and correctness tests.

No datasets, training, or downloads required.

Usage
-----
    python -m scripts.test_keyed_bottleneck
"""

from __future__ import annotations

import torch
from src.security.keyed_bottleneck import KeyedLatentBottleneck


def _run_layout_checks(layout: str, latent_size: int) -> None:
    latent_channels = 4
    block_size = 16
    bias_scale = 0.1
    B = 4

    print(f"\n[test] Layout: {layout}  latent={latent_channels}x{latent_size}x{latent_size}")
    klb = KeyedLatentBottleneck(
        secret_key="test_key_abc123",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=block_size,
        block_layout=layout,
        bias_scale=bias_scale,
    )
    z = torch.randn(B, latent_channels, latent_size, latent_size)
    z_k = klb(z)
    z_rec = klb.invert(z_k)

    max_err = (z - z_rec).abs().max().item()
    mean_err = (z - z_rec).abs().mean().item()
    print(f"[test] max_abs_error:  {max_err:.3e}")
    print(f"[test] mean_abs_error: {mean_err:.3e}")
    assert max_err < 1e-5, (
        f"Reconstruction error too large for layout={layout}: {max_err:.3e}"
    )

    klb2 = KeyedLatentBottleneck(
        secret_key="different_key_xyz789",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=block_size,
        block_layout=layout,
        bias_scale=bias_scale,
    )
    z_k2 = klb2(z)
    key_diff = (z_k - z_k2).abs().max().item()
    print(f"[test] Max diff key1 vs key2: {key_diff:.4f}")
    assert key_diff > 0.01, f"Different keys should produce different transforms; got {key_diff:.3e}"

    klb3 = KeyedLatentBottleneck(
        secret_key="test_key_abc123",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=block_size,
        block_layout=layout,
        bias_scale=bias_scale,
    )
    z_k3 = klb3(z)
    same_diff = (z_k - z_k3).abs().max().item()
    print(f"[test] Max diff same key (must be 0): {same_diff:.3e}")
    assert same_diff == 0.0, f"Same key must produce identical transform; got {same_diff:.3e}"

    mag = (z_k - z).abs().mean().item()
    print(f"[test] Mean |z_k - z| (must be > 0): {mag:.4f}")
    assert mag > 0.0, "Transform should not be the identity"

    klb_nb = KeyedLatentBottleneck(
        secret_key="test_key_abc123",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=block_size,
        block_layout=layout,
        bias_scale=0.0,
    )
    z_k_nb = klb_nb(z)
    z_r_nb = klb_nb.invert(z_k_nb)
    err_nb = (z - z_r_nb).abs().max().item()
    print(f"[test] Bias-free max_abs_error: {err_nb:.3e}")
    assert err_nb < 1e-5, f"Bias-free reconstruction error too large: {err_nb:.3e}"

    sd = klb.state_dict()
    assert "W" not in sd, "W should NOT appear in state_dict (persistent=False)"
    assert "beta" not in sd, "beta should NOT appear in state_dict (persistent=False)"

    if layout == "patch":
        assert klb._num_blocks == (latent_size // 2) ** 2
        assert klb._patch_size == 2
        print(f"[test] Patch layout blocks: {klb._num_blocks}  patch_size={klb._patch_size}")


def main() -> None:
    print("[test] ── KeyedLatentBottleneck smoke test ──")
    _run_layout_checks("flat", latent_size=8)
    _run_layout_checks("patch", latent_size=16)
    print("\n[test] All keyed layout tests passed — KeyedLatentBottleneck is correct.")


if __name__ == "__main__":
    main()
