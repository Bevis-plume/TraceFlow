"""
Utilities for TraceFlow forensic trace detection.

The forensic trace objective is different from visible/re-stamped watermarking:
clean CIFAR/AE/generated images should score near chance for the owner code,
while raw no-key inversion artifacts should score high.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def expand_bits(bits: torch.Tensor, batch_size: int, *, device: torch.device | None = None) -> torch.Tensor:
    bits = bits.float()
    if device is not None:
        bits = bits.to(device)
    if bits.ndim == 1:
        bits = bits.unsqueeze(0)
    return bits.expand(batch_size, -1)


def positive_owner_loss(logits: torch.Tensor, owner_bits: torch.Tensor) -> torch.Tensor:
    target = expand_bits(owner_bits, logits.shape[0], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target)


def clean_negative_loss(logits: torch.Tensor) -> torch.Tensor:
    target = torch.full_like(logits, 0.5)
    return F.binary_cross_entropy_with_logits(logits, target)


def bit_accuracy_from_logits(logits: torch.Tensor, owner_bits: torch.Tensor) -> float:
    target = expand_bits(owner_bits, logits.shape[0], device=logits.device).bool()
    pred = torch.sigmoid(logits) >= 0.5
    return (pred == target).float().mean().item()


def owner_match_scores(logits: torch.Tensor, owner_bits: torch.Tensor) -> torch.Tensor:
    """Return per-sample soft agreement with the owner bit code in [0, 1]."""
    probs = torch.sigmoid(logits)
    target = expand_bits(owner_bits, logits.shape[0], device=logits.device)
    return (probs * target + (1.0 - probs) * (1.0 - target)).mean(dim=1)


def binary_auroc(pos_scores: Iterable[float], neg_scores: Iterable[float]) -> float:
    """Pairwise AUROC without sklearn dependency."""
    pos = torch.as_tensor(list(pos_scores), dtype=torch.float64)
    neg = torch.as_tensor(list(neg_scores), dtype=torch.float64)
    if pos.numel() == 0 or neg.numel() == 0:
        return float('nan')
    cmp = pos[:, None] - neg[None, :]
    wins = (cmp > 0).double().sum()
    ties = (cmp == 0).double().sum()
    return ((wins + 0.5 * ties) / (pos.numel() * neg.numel())).item()
