"""
src/watermarking/metrics.py
============================
Watermark evaluation metrics.

All functions accept ``bit_probs`` in [0, 1] of shape ``[B, bit_length]`` and
ground-truth ``bits`` (0/1) broadcastable to the same shape.
"""

from __future__ import annotations

import torch


def _binarize(bit_probs: torch.Tensor, threshold: float) -> torch.Tensor:
    return (bit_probs >= threshold).float()


def bit_accuracy(
    bit_probs: torch.Tensor,
    bits: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Mean fraction of correctly decoded bits over the batch."""
    pred = _binarize(bit_probs, threshold)
    target = bits.to(pred.dtype)
    return (pred == target).float().mean().item()


def bit_error_rate(
    bit_probs: torch.Tensor,
    bits: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Mean fraction of incorrectly decoded bits (1 - bit_accuracy)."""
    return 1.0 - bit_accuracy(bit_probs, bits, threshold)


def detection_passed(
    bit_probs: torch.Tensor,
    bits: torch.Tensor,
    threshold_acc: float = 0.9,
) -> bool:
    """Whether bit accuracy meets/exceeds the detection threshold."""
    return bool(bit_accuracy(bit_probs, bits) >= threshold_acc)


def image_delta_mse(x_w: torch.Tensor, x: torch.Tensor) -> float:
    """Mean-squared distortion between watermarked and original images."""
    return torch.mean((x_w - x) ** 2).item()
