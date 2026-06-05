"""
src/security/identity_transform.py
====================================
Phase-1 identity latent transform hook.

In Phase 1, this is a no-op:  forward(z) = z,  invert(z) = z.

This hook exists as the placeholder for the future keyed latent bottleneck
(Phase 2+).  The training pipeline always calls:

    z_protected = latent_transform(z)

so switching to a keyed transform only requires swapping this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityLatentTransform(nn.Module):
    """Identity (no-op) latent transform.

    Phase 1 placeholder for future keyed latent bottleneck.
    All methods are differentiable pass-throughs.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return z unchanged."""
        return z

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        """Return z unchanged (inversion of identity is identity)."""
        return z
