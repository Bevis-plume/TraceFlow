"""
src/security/factory.py
========================
Factory for latent transform modules.

Usage
-----
    from src.security.factory import build_latent_transform

    latent_transform = build_latent_transform(
        cfg=cfg.get("security", {}),
        latent_channels=4,
        latent_size=32,
    ).to(device)

Supported transform types
--------------------------
- ``identity``: No-op.  z_k = z.  Used for baseline experiments.
- ``keyed``:    Block-orthogonal transform from ``KeyedLatentBottleneck``.
                Requires ``secret_key`` in the security config section.

Example config sections
------------------------
  security:
    latent_transform:
      type: identity

  security:
    latent_transform:
      type: keyed
      secret_key: CHANGE_ME_FOR_REAL_RUNS
      block_size: 16
      block_layout: patch
      bias_scale: 0.1
"""

from __future__ import annotations

import torch.nn as nn


def build_latent_transform(
    cfg: dict,
    latent_channels: int,
    latent_size: int,
) -> nn.Module:
    """Instantiate the latent transform from the ``security:`` config section.

    Args:
        cfg:              The ``security:`` sub-dict of the YAML config.
                          Defaults to identity transform if key is missing.
        latent_channels:  C in [B, C, H, W].
        latent_size:      H = W (assumed square).

    Returns:
        An ``nn.Module`` implementing ``forward(z) -> z_k``.
        For invertible transforms, also implements ``invert(z_k) -> z``.

    Raises:
        ValueError:  If ``type`` is not a recognised value.
    """
    lt_cfg = cfg.get("latent_transform", {})
    kind = lt_cfg.get("type", "identity")

    if kind == "identity":
        from src.security.identity_transform import IdentityLatentTransform
        return IdentityLatentTransform()

    elif kind == "keyed":
        from src.security.keyed_bottleneck import KeyedLatentBottleneck

        secret_key = lt_cfg.get("secret_key")
        if not secret_key:
            raise ValueError(
                "latent_transform.type=keyed requires a non-empty 'secret_key' "
                "in the security config.  Set it to a strong random string for real runs."
            )

        return KeyedLatentBottleneck(
            secret_key=secret_key,
            latent_channels=latent_channels,
            latent_size=latent_size,
            block_size=int(lt_cfg.get("block_size", 16)),
            block_layout=str(lt_cfg.get("block_layout", "flat")),
            bias_scale=float(lt_cfg.get("bias_scale", 0.1)),
        )

    else:
        raise ValueError(
            f"Unknown latent_transform type: {kind!r}.  "
            f"Supported types: 'identity', 'keyed'."
        )
