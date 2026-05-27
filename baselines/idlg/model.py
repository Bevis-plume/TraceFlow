from __future__ import annotations

import torch
import torch.nn as nn


class LeNet(nn.Module):
    """LeNet architecture used by the official iDLG implementation."""

    def __init__(self, channel: int = 1, hidden: int = 588, num_classes: int = 10) -> None:
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def weights_init(module: nn.Module) -> None:
    """Match the original iDLG random init behavior."""
    if hasattr(module, "weight") and getattr(module, "weight") is not None:
        try:
            module.weight.data.uniform_(-0.5, 0.5)
        except Exception:
            pass
    if hasattr(module, "bias") and getattr(module, "bias") is not None:
        try:
            module.bias.data.uniform_(-0.5, 0.5)
        except Exception:
            pass