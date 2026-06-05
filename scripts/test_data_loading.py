"""
scripts/test_data_loading.py
============================
Smoke test for src/data/image_datasets.py.

Usage:
    python -m scripts.test_data_loading
"""

import torch

from src.data.image_datasets import RandomTensorDataset, build_dataset, build_dataloader


def test_random_tensor_dataset() -> None:
    ds = RandomTensorDataset(image_size=32, num_samples=16, channels=3)
    assert len(ds) == 16, f"Expected 16 samples, got {len(ds)}"
    img, label = ds[0]
    assert img.shape == (3, 32, 32), f"Unexpected shape: {img.shape}"
    assert img.dtype == torch.float32, f"Expected float32, got {img.dtype}"
    assert img.min() >= -1.0 and img.max() <= 1.0, "Image values out of [-1, 1] range"
    assert label == 0
    print("  [PASS] RandomTensorDataset: shape, dtype, range OK")


def test_build_dataset_random() -> None:
    ds = build_dataset(name="random", image_size=64, smoke_samples=8)
    assert len(ds) == 8
    img, _ = ds[0]
    assert img.shape == (3, 64, 64)
    print("  [PASS] build_dataset(name='random'): shape OK")


def test_build_dataset_smoke_flag() -> None:
    # smoke=True should override name
    ds = build_dataset(name="cifar10", image_size=32, smoke=True, smoke_samples=4)
    assert isinstance(ds, RandomTensorDataset)
    assert len(ds) == 4
    print("  [PASS] build_dataset(smoke=True): always returns RandomTensorDataset")


def test_build_dataloader() -> None:
    ds = RandomTensorDataset(image_size=16, num_samples=10, channels=3)
    loader = build_dataloader(ds, batch_size=4, num_workers=0, drop_last=False)
    batches = list(loader)
    assert len(batches) == 3, f"Expected 3 batches (10 samples, bs=4), got {len(batches)}"
    imgs, labels = batches[0]
    assert imgs.shape == (4, 3, 16, 16), f"Unexpected batch shape: {imgs.shape}"
    print("  [PASS] build_dataloader: batch shape OK")


def main() -> None:
    print("Running test_data_loading smoke tests...")
    test_random_tensor_dataset()
    test_build_dataset_random()
    test_build_dataset_smoke_flag()
    test_build_dataloader()
    print("All tests passed.")


if __name__ == "__main__":
    main()
