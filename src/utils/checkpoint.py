"""src/utils/checkpoint.py — Checkpoint save/load utilities."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

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
        self._model = model
        # Number of EMA updates applied. Used for decay warmup so the average is
        # not dominated by the (near-zero, zero-init) model at the start of
        # training. Without this, short/medium runs produce a heavily damped EMA
        # whose velocity field cannot denoise pure noise during sampling.
        self.num_updates = 0
        self._register(model)

    def _register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    def _effective_decay(self) -> float:
        """Warmup-clamped decay: min(decay, (1 + n) / (10 + n)).

        Early on (small n) the effective decay is small so the EMA tracks the
        model closely; it approaches ``self.decay`` as training progresses. This
        is the standard diffusion/DiT EMA warmup and is essential for runs that
        are not millions of steps long.
        """
        warmup = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, warmup)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self._effective_decay()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    decay * self.shadow[name]
                    + (1.0 - decay) * param.data.float()
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

    @contextmanager
    def average_parameters(self, model: Optional[nn.Module] = None) -> Iterator[None]:
        """Temporarily swap EMA parameters into ``model`` and restore raw weights.

        Sampling/evaluation should use this context so short runs do not report
        raw, non-averaged weights while checkpoints still preserve both states.
        """
        target = model or self._model
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in target.named_parameters():
                if param.requires_grad and name in self.shadow:
                    backup[name] = param.data.clone()
            self.copy_to(target)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in target.named_parameters():
                    if name in backup:
                        param.data.copy_(backup[name].to(param.device))
