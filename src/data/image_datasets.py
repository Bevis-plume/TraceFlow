"""
src/data/image_datasets.py
============================
Dataset utilities for TraceFlow.

Supports:
    "random"       — random RGB tensors in [-1,1], for smoke tests.
    "cifar10"      — CIFAR-10 (download configurable).
    "imagefolder"  — torchvision ImageFolder from a directory.
    "flat"         — flat directory of images (no subdirs).

All datasets return images normalized to [-1, 1].
Large dataset downloads are NOT triggered silently.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets as dsets


# ---------------------------------------------------------------------------
# Random tensor dataset (smoke mode)
# ---------------------------------------------------------------------------

class RandomTensorDataset(Dataset):
    """Returns random float32 tensors normalized to [-1, 1].

    Useful for smoke tests that must not require real data.
    """

    def __init__(
        self,
        image_size: int = 64,
        num_samples: int = 128,
        channels: int = 3,
    ) -> None:
        self.image_size = image_size
        self.num_samples = num_samples
        self.channels = channels
        self.shape = (channels, image_size, image_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = torch.rand(*self.shape) * 2.0 - 1.0   # uniform in [-1, 1]
        label = 0
        return img, label


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------

def _make_transform(image_size: int) -> Callable:
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),                          # [0, 1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_dataset(
    name: str,
    root: str = "./data",
    image_size: int = 256,
    download: bool = False,
    smoke: bool = False,
    smoke_samples: int = 128,
) -> Dataset:
    """Build a dataset by name.

    Args:
        name:          Dataset name: "random", "cifar10", "imagefolder", "flat".
        root:          Root directory for on-disk datasets.
        image_size:    Spatial size to resize/crop images to.
        download:      Allow downloading CIFAR-10 if missing.
        smoke:         If True, always return a RandomTensorDataset.
        smoke_samples: Number of samples for smoke dataset.

    Returns:
        A torch Dataset yielding (image_tensor, label) pairs.
        image_tensor is float32 in [-1, 1].
    """
    if smoke or name == "random":
        return RandomTensorDataset(image_size=image_size, num_samples=smoke_samples)

    transform = _make_transform(image_size)

    if name == "cifar10":
        if not download:
            # Check if data already exists before erroring
            cifar_path = Path(root) / "cifar-10-batches-py"
            if not cifar_path.exists():
                raise FileNotFoundError(
                    f"CIFAR-10 not found at {cifar_path}. "
                    "Set data.download=true in config to download, or use name=random for smoke mode."
                )
        return dsets.CIFAR10(root=root, train=True, download=download, transform=transform)

    if name == "imagefolder":
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"ImageFolder root not found: {root}. "
                "Update data.root in your config or use name=random for smoke mode."
            )
        return dsets.ImageFolder(root=root, transform=transform)

    if name == "flat":
        # Flat directory of images (no class subdirs) — wrap as unlabeled
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Flat image directory not found: {root}")
        return _FlatImageDataset(root=root, transform=transform)

    raise ValueError(
        f"Unknown dataset name {name!r}. Choose from: 'random', 'cifar10', 'imagefolder', 'flat'."
    )


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader for a given dataset."""
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Flat directory dataset
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class _FlatImageDataset(Dataset):
    """Unlabeled flat directory of image files."""

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        from PIL import Image as _PIL_Image
        self._PIL_Image = _PIL_Image

        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self._PIL_Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
