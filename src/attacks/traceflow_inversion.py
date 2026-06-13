"""
src/attacks/traceflow_inversion.py
=====================================
TraceFlow inversion evaluation harness — gradient matching in latent and pixel space.

Threat model
------------
- Attacker knows : FlowTransformer weights, AutoencoderBackend weights, watermark
  module weights (extractor, decoder_adapter, latent_detector).
- Attacker does NOT know : ``secret_key`` (the keyed latent transform is opaque).
- Defender uses ``secret_key`` only for forensic decode and latent detector checks
  at evaluation time.  The key is never stored in checkpoints or metric JSON.

Two attacks
-----------
``latent_inversion_attack``
    Optimise a dummy protected latent ``z_k_dummy`` (no key) to match gradient
    signals from the real training step.  The attacker decodes ``z_k_dummy``
    directly (no inversion) for a "no-key" reconstruction.  The defender can
    apply the key to get a forensic decode.

``pixel_inversion_attack``
    Optimise a dummy image ``x_dummy`` in pixel space.  The attacker encodes
    ``x_dummy`` with the autoencoder (no key transform) and computes the
    gradient matching loss purely in the unprotected latent space.

Both attacks use iDLG-style second-order gradient matching (grad-of-grad):
    L_attack = Σ_i  1 - cos_sim(∂L/∂θ_i (dummy), ∂L/∂θ_i (target))
where θ are the FlowTransformer parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Deterministic attack batch state
# ---------------------------------------------------------------------------

@dataclass
class AttackBatchState:
    """Fixed ``(t, eps, bits)`` realisation shared by target and dummy paths.

    For the gradient-matching objective to be scientifically valid, the rectified
    flow time ``t``, the noise ``eps``, and the watermark message ``bits`` must be
    **identical** between the defender's target-gradient computation and every
    dummy-gradient computation inside the attack loop.  Sampling fresh noise per
    call would make the two gradient sets incomparable.

    Attributes:
        t:    Fixed time values, shape ``(B,)``.
        eps:  Fixed noise in latent space, shape ``(B, C, H, W)``.
        bits: Fixed watermark message, shape ``(B, bit_length)`` (or ``None`` if
              no watermark is in use).
    """

    t: torch.Tensor
    eps: torch.Tensor
    bits: Optional[torch.Tensor]
    y: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "AttackBatchState":
        return AttackBatchState(
            t=self.t.to(device),
            eps=self.eps.to(device),
            bits=None if self.bits is None else self.bits.to(device),
            y=None if self.y is None else self.y.to(device),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wm_modules_list(wm: Optional[Dict[str, Any]]) -> List[nn.Module]:
    """Return all nn.Module instances stored in a watermark_modules dict."""
    if wm is None:
        return []
    out = []
    for key in ("extractor", "decoder_adapter", "latent_detector"):
        m = wm.get(key)
        if m is not None and isinstance(m, nn.Module):
            out.append(m)
    return out


def _traceflow_loss_from_latent(
    model: nn.Module,
    autoencoder: Any,
    latent_transform: Any,
    watermark_modules: Optional[Dict[str, Any]],
    z_k: torch.Tensor,
    *,
    t: Optional[torch.Tensor] = None,
    eps: Optional[torch.Tensor] = None,
    bits: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the full TraceFlow training loss for a given latent ``z_k``.

    Used both for computing *target* gradients (with the real keyed transform)
    and *dummy* gradients (identity transform for ``no_key``, real transform for
    ``oracle_key``).

    Args:
        z_k:  Protected (or unprotected, in the attacker's case) latent
              ``[B, C, H, W]``.
        t:    Optional fixed rectified-flow time ``(B,)`` (deterministic mode).
        eps:  Optional fixed noise ``[B, C, H, W]`` (deterministic mode).
        bits: Optional fixed watermark message ``[B, bit_length]``.  Falls back
              to the module's stored bits when ``None``.

    Returns:
        Scalar loss with computation graph intact.
    """
    from src.generation.rectified_flow import flow_loss_with_state

    if watermark_modules is None or not watermark_modules.get("enabled", False):
        # Flow-only path — still honour fixed (t, eps) for determinism.
        return flow_loss_with_state(model, z_k, y=y, t=t, eps=eps)["loss"]

    wm_cfg = watermark_modules["config"]
    wm_type = watermark_modules["type"]

    flow_state = flow_loss_with_state(model, z_k, y=y, t=t, eps=eps)
    flow_l = flow_state["loss"]

    if wm_type != "traceflow":
        return flow_l

    z_hat_k = flow_state["z_hat"]
    extractor = watermark_modules["extractor"]
    decoder_adapter = watermark_modules["decoder_adapter"]
    latent_detector = watermark_modules["latent_detector"]

    if bits is None:
        from src.watermarking.message import expand_bits
        B = z_k.shape[0]
        batch_bits = expand_bits(watermark_modules["bits"], B).to(z_k.device)
    else:
        batch_bits = bits.to(z_k.device)

    # --- Decoder-adapter path (z_hat_k → image) ---
    z_hat = latent_transform.invert(z_hat_k)           # identity for no_key attacker
    x_hat = autoencoder.decode_with_grad(z_hat)

    alpha = wm_cfg["alpha"]
    residual = decoder_adapter(x_hat, batch_bits)
    x_w = torch.clamp(x_hat + alpha * residual, -1.0, 1.0)

    bce = nn.BCEWithLogitsLoss()
    bit_logits = extractor.logits(x_w)
    wm_img_l = bce(bit_logits, batch_bits)
    img_l = F.mse_loss(x_w, x_hat.detach())
    residual_l = residual.pow(2).mean()

    # --- Latent cycle path (re-encode → latent detector) ---
    z_re = autoencoder.encode_with_grad(x_w)
    z_re_k = latent_transform(z_re)                    # identity for no_key attacker
    bit_logits_latent = latent_detector.logits(z_re_k)
    wm_latent_l = bce(bit_logits_latent, batch_bits)
    cycle_l = F.mse_loss(z_re_k, z_hat_k.detach())

    wm_robust_l = torch.zeros((), device=z_k.device, dtype=flow_l.dtype)
    if wm_cfg.get("robustness_enabled", False) and wm_cfg.get("lambda_wm_robust", 0.0) > 0:
        from src.watermarking.augment import deterministic_robust_views
        robust_losses = []
        max_views = int(wm_cfg.get("robust_max_views", 2))
        robust_latent_enabled = bool(wm_cfg.get("robust_latent_enabled", False))
        robust_detach_input = bool(wm_cfg.get("robust_detach_input", True))

        # Training usually detaches robust views so they train the detector
        # without pushing second-order gradients through the model/VAE path.
        # This inversion objective matches only FlowTransformer parameter
        # gradients, so a detached robust branch contributes no useful target
        # signal and only explodes memory. Skip it rather than building a graph
        # that cannot affect the matched gradients.
        if not robust_detach_input:
            for x_view in deterministic_robust_views(x_w)[1:1 + max(0, max_views)]:
                robust_losses.append(bce(extractor.logits(x_view), batch_bits))
                if robust_latent_enabled:
                    z_view = autoencoder.encode_with_grad(x_view)
                    z_view_k = latent_transform(z_view)
                    robust_losses.append(bce(latent_detector.logits(z_view_k), batch_bits))
            if robust_losses:
                wm_robust_l = torch.stack(robust_losses).mean()

    loss = (
        flow_l
        + wm_cfg["lambda_wm_img"]     * wm_img_l
        + wm_cfg["lambda_wm_latent"]  * wm_latent_l
        + wm_cfg.get("lambda_wm_robust", 0.0) * wm_robust_l
        + wm_cfg["lambda_img"]        * img_l
        + wm_cfg["lambda_cycle"]      * cycle_l
        + wm_cfg["lambda_residual"]   * residual_l
    )
    return loss


def _model_param_grads(
    loss: torch.Tensor,
    model_params: List[nn.Parameter],
) -> List[Optional[torch.Tensor]]:
    """Return per-parameter gradients of ``loss`` with computation graph retained.

    ``create_graph=True`` allows further differentiation through the returned
    gradients (necessary for the grad-of-grad trick).
    ``allow_unused=True`` returns ``None`` for parameters that did not
    participate in the forward pass.
    """
    grads = torch.autograd.grad(
        loss,
        model_params,
        create_graph=True,
        allow_unused=True,
    )
    return list(grads)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gradient_matching_loss(
    grads_dummy: List[Optional[torch.Tensor]],
    grads_target: List[Optional[torch.Tensor]],
) -> torch.Tensor:
    """Cosine-distance gradient matching loss.

    Sums ``(1 - cosine_similarity)`` over all non-None gradient pairs.
    Handles ``None`` entries robustly (they are skipped).

    Args:
        grads_dummy:  Per-parameter gradients from the dummy input
                      (must carry a computation graph for backprop).
        grads_target: Per-parameter gradients from the real input
                      (treated as constants — no graph required).

    Returns:
        Scalar loss tensor.  Zero when no valid pairs exist.
    """
    total: Optional[torch.Tensor] = None
    device = None

    for gd, gt in zip(grads_dummy, grads_target):
        if gd is None or gt is None:
            continue
        device = gd.device
        gd_flat = gd.flatten()
        gt_flat = gt.detach().flatten()
        cos = F.cosine_similarity(gd_flat.unsqueeze(0), gt_flat.unsqueeze(0))
        term = 1.0 - cos
        total = term if total is None else total + term

    if total is None:
        # No valid gradient pairs — return differentiable zero on correct device
        _d = device or torch.device("cpu")
        return torch.zeros(1, device=_d, requires_grad=True).squeeze()
    return total


def compute_target_gradients(
    model: nn.Module,
    autoencoder: Any,
    latent_transform: Any,
    watermark_modules: Optional[Dict[str, Any]],
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    objective: str = "traceflow",
) -> Tuple[List[Optional[torch.Tensor]], AttackBatchState]:
    """Compute defender-side gradient signals on a real image batch.

    This is the *target* the attacker tries to match.  The real keyed latent
    transform is used here so the gradients reflect true training conditions.

    A deterministic :class:`AttackBatchState` (fixed ``t``, ``eps``, ``bits``) is
    created here and **must** be threaded back into the attack loops so that the
    dummy gradients are computed under the exact same realisation.  Without this,
    the gradient-matching objective is corrupted by random noise differences and
    the reported attack/watermark metrics are scientifically meaningless.

    Args:
        model:             FlowTransformer.
        autoencoder:       AutoencoderBackend.
        latent_transform:  Keyed (or identity) latent transform.
        watermark_modules: Dict from ``build_watermark_modules``, or ``None``.
        x:                 Real image batch ``[B, 3, H, W]`` in ``[-1, 1]``.
        objective:         ``"traceflow"`` (default) or ``"flow_only"``.

    Returns:
        ``(target_grads, attack_state)`` where ``target_grads`` is a list of
        gradient tensors (one per ``model.parameters()``; ``None`` where no
        gradient was received) and ``attack_state`` carries the fixed
        ``(t, eps, bits)`` realisation used.
    """
    from src.generation.rectified_flow import sample_t

    model.zero_grad()

    # Encode + key-transform (defender's view, no grad needed here)
    with torch.no_grad():
        z = autoencoder.encode(x)
        z_k = z if latent_transform is None else latent_transform(z)

    B = z_k.shape[0]
    device = z_k.device

    # --- Build the fixed (t, eps, bits) realisation ---
    fixed_eps = torch.randn_like(z_k)
    fixed_t = sample_t(B, device)
    fixed_bits: Optional[torch.Tensor] = None
    if (
        watermark_modules is not None
        and watermark_modules.get("enabled", False)
        and watermark_modules.get("type") == "traceflow"
    ):
        from src.watermarking.message import expand_bits
        fixed_bits = expand_bits(watermark_modules["bits"], B).to(device)

    fixed_y = None if y is None else y.to(device=device, dtype=torch.long)
    attack_state = AttackBatchState(t=fixed_t, eps=fixed_eps, bits=fixed_bits, y=fixed_y)

    if objective == "flow_only" or watermark_modules is None:
        from src.generation.rectified_flow import flow_loss_with_state
        loss = flow_loss_with_state(
            model, z_k, y=fixed_y, t=fixed_t, eps=fixed_eps
        )["loss"]
    else:
        loss = _traceflow_loss_from_latent(
            model, autoencoder, latent_transform, watermark_modules, z_k,
            t=fixed_t, eps=fixed_eps, bits=fixed_bits, y=fixed_y,
        )

    loss.backward()
    grads = [
        p.grad.detach().clone() if p.grad is not None else None
        for p in model.parameters()
    ]
    model.zero_grad()
    return grads, attack_state


def _resolve_attacker_transform(
    attacker: str,
    latent_transform: Optional[Any],
) -> Any:
    """Select the latent transform used inside the dummy objective.

    - ``"no_key"``     → :class:`IdentityLatentTransform` (attacker has no key).
    - ``"oracle_key"`` → the real ``latent_transform`` (upper-bound oracle).
    """
    if attacker == "oracle_key":
        if latent_transform is None:
            raise ValueError(
                "attacker='oracle_key' requires the real latent_transform."
            )
        return latent_transform
    if attacker == "no_key":
        from src.security.identity_transform import IdentityLatentTransform
        return IdentityLatentTransform()
    raise ValueError(f"Unknown attacker mode: {attacker!r}")


def latent_inversion_attack(
    model: nn.Module,
    autoencoder: Any,
    watermark_modules: Optional[Dict[str, Any]],
    target_grads: List[Optional[torch.Tensor]],
    z_k_shape: Tuple[int, int, int, int],
    attack_state: AttackBatchState,
    steps: int = 300,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
    log_interval: int = 50,
    attacker: str = "no_key",
    latent_transform: Optional[Any] = None,
    snapshot_steps: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Gradient-matching attack in latent space.

    The attacker optimises a dummy latent ``z_k_dummy`` and matches the
    FlowTransformer parameter gradients to ``target_grads``.  Every dummy
    gradient is computed under the **fixed** ``attack_state`` realisation so the
    comparison is noise-free.

    Args:
        model:             FlowTransformer (put in eval mode internally).
        autoencoder:       AutoencoderBackend.
        watermark_modules: Dict from ``build_watermark_modules``, or ``None``.
        target_grads:      Gradient list from :func:`compute_target_gradients`.
        z_k_shape:         ``(B, C, H, W)`` shape for the dummy latent.
        attack_state:      Fixed ``(t, eps, bits)`` realisation (deterministic).
        steps:             Number of Adam optimisation steps.
        lr:                Adam learning rate.
        device:            Target device.
        log_interval:      Log progress every N steps (0 = silent).
        attacker:          ``"no_key"`` (identity dummy transform) or
                           ``"oracle_key"`` (real dummy transform).
        latent_transform:  Real transform, required when ``attacker='oracle_key'``.

    Returns:
        Dict with keys ``z_k_dummy`` (detached), ``gml_history``, ``final_gml``,
        and ``attacker``.
    """
    dummy_transform = _resolve_attacker_transform(attacker, latent_transform)
    attack_state = attack_state.to(device)

    z_k_dummy = torch.randn(*z_k_shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z_k_dummy], lr=lr)

    model_params = list(model.parameters())
    gml_history: List[float] = []
    snapshot_set = {int(s) for s in (snapshot_steps or []) if int(s) > 0}
    snapshots: Dict[int, torch.Tensor] = {}

    was_training = model.training
    model.eval()
    for m in _wm_modules_list(watermark_modules):
        m.eval()

    try:
        for step in range(steps):
            optimizer.zero_grad()

            # Dummy loss uses the SAME fixed (t, eps, bits) as the target.
            loss = _traceflow_loss_from_latent(
                model, autoencoder, dummy_transform, watermark_modules, z_k_dummy,
                t=attack_state.t, eps=attack_state.eps, bits=attack_state.bits, y=attack_state.y,
            )
            dummy_grads = _model_param_grads(loss, model_params)

            gml = gradient_matching_loss(dummy_grads, target_grads)
            gml.backward()
            optimizer.step()

            gml_val = gml.item()
            gml_history.append(gml_val)
            if (step + 1) in snapshot_set:
                snapshots[step + 1] = z_k_dummy.detach().cpu()
            if log_interval > 0 and (step + 1) % log_interval == 0:
                print(f"  [latent_inv:{attacker}] step={step+1:4d}  gml={gml_val:.6f}")
    finally:
        if was_training:
            model.train()

    return {
        "z_k_dummy": z_k_dummy.detach(),
        "gml_history": gml_history,
        "final_gml": gml_history[-1] if gml_history else float("nan"),
        "attacker": attacker,
        "snapshots": snapshots,
    }


def pixel_inversion_attack(
    model: nn.Module,
    autoencoder: Any,
    watermark_modules: Optional[Dict[str, Any]],
    target_grads: List[Optional[torch.Tensor]],
    x_shape: Tuple[int, int, int, int],
    attack_state: AttackBatchState,
    steps: int = 300,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
    log_interval: int = 50,
    attacker: str = "no_key",
    latent_transform: Optional[Any] = None,
    snapshot_steps: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Gradient-matching attack in pixel space.

    Optimises a dummy image ``x_dummy`` by matching FlowTransformer parameter
    gradients.  The attacker encodes ``x_dummy`` with the AE and computes the
    dummy loss under the **fixed** ``attack_state`` realisation.

    Args:
        model:             FlowTransformer (put in eval mode internally).
        autoencoder:       AutoencoderBackend (must support ``encode_with_grad``).
        watermark_modules: Dict from ``build_watermark_modules``, or ``None``.
        target_grads:      Gradient list from :func:`compute_target_gradients`.
        x_shape:           ``(B, 3, H, W)`` shape for the dummy image.
        attack_state:      Fixed ``(t, eps, bits)`` realisation (deterministic).
        steps:             Number of Adam optimisation steps.
        lr:                Adam learning rate.
        device:            Target device.
        log_interval:      Log progress every N steps (0 = silent).
        attacker:          ``"no_key"`` (identity dummy transform) or
                           ``"oracle_key"`` (real dummy transform).
        latent_transform:  Real transform, required when ``attacker='oracle_key'``.

    Returns:
        Dict with keys ``x_dummy`` (detached), ``gml_history``, ``final_gml``,
        and ``attacker``.
    """
    dummy_transform = _resolve_attacker_transform(attacker, latent_transform)
    attack_state = attack_state.to(device)

    x_dummy = torch.empty(*x_shape, device=device, requires_grad=True)
    with torch.no_grad():
        x_dummy.data.uniform_(-1.0, 1.0)
    optimizer = torch.optim.Adam([x_dummy], lr=lr)

    model_params = list(model.parameters())
    gml_history: List[float] = []
    snapshot_set = {int(s) for s in (snapshot_steps or []) if int(s) > 0}
    snapshots: Dict[int, torch.Tensor] = {}

    was_training = model.training
    model.eval()
    for m in _wm_modules_list(watermark_modules):
        m.eval()

    try:
        for step in range(steps):
            optimizer.zero_grad()

            # Encode with gradient tracking; dummy loss uses fixed (t, eps, bits).
            z_dummy = autoencoder.encode_with_grad(x_dummy)
            loss = _traceflow_loss_from_latent(
                model, autoencoder, dummy_transform, watermark_modules, z_dummy,
                t=attack_state.t, eps=attack_state.eps, bits=attack_state.bits, y=attack_state.y,
            )
            dummy_grads = _model_param_grads(loss, model_params)

            gml = gradient_matching_loss(dummy_grads, target_grads)
            gml.backward()
            optimizer.step()

            # Keep dummy image in valid range
            with torch.no_grad():
                x_dummy.data.clamp_(-1.0, 1.0)

            gml_val = gml.item()
            gml_history.append(gml_val)
            if (step + 1) in snapshot_set:
                snapshots[step + 1] = x_dummy.detach().cpu()
            if log_interval > 0 and (step + 1) % log_interval == 0:
                print(f"  [pixel_inv:{attacker}] step={step+1:4d}  gml={gml_val:.6f}")
    finally:
        if was_training:
            model.train()

    return {
        "x_dummy": x_dummy.detach(),
        "gml_history": gml_history,
        "final_gml": gml_history[-1] if gml_history else float("nan"),
        "attacker": attacker,
        "snapshots": snapshots,
    }
