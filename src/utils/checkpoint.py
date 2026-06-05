"""src/utils/checkpoint.py — Checkpoint save/load utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    ema_model: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint to `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if ema_model is not None:
        state["ema_model"] = ema_model.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> int:
    """Load a checkpoint. Returns the step number.

    `ema_model` can be either an ``nn.Module`` or an ``EMAModel`` instance —
    anything that implements ``.load_state_dict()``.
    """
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if ema_model is not None and "ema_model" in state:
        ema_model.load_state_dict(state["ema_model"])
    return int(state.get("step", 0))


class EMAModel:
    """Exponential Moving Average of model parameters.

    Usage::
        ema = EMAModel(model, decay=0.9999)
        # after each optimizer step:
        ema.update(model)
        # to evaluate with EMA weights:
        with ema.average_parameters():
            outputs = model(inputs)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.float()
                )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into model (in-place)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device))
