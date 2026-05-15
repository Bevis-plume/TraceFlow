"""Image range helpers shared by training, attack, and evaluation code.

The VAE in this project consumes and emits image tensors in ``[0, 1]``.
Keeping that contract explicit avoids subtle mismatches between tensors loaded
from datasets, tensors decoded by the VAE, and PNG files reloaded for metrics.
"""

from __future__ import annotations

import torch
import torchvision.transforms as T


def cifar10_transform() -> T.Compose:
    """Return the canonical CIFAR-10 transform for VAE inputs.

    Images are converted to float tensors in ``[0, 1]`` without normalisation.
    """
    return T.Compose([T.ToTensor()])


def to_vae_input(image: torch.Tensor) -> torch.Tensor:
    """Clamp a tensor to the VAE's expected image range ``[0, 1]``."""
    return image.clamp(0.0, 1.0)


def from_vae_output(image: torch.Tensor) -> torch.Tensor:
    """Clamp a decoded image for saving or image-quality metrics."""
    return image.clamp(0.0, 1.0)
