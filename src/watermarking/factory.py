"""
src/watermarking/factory.py
============================
Config-driven factory for the final TraceFlow watermark modules.

When ``watermark.enabled`` is true, the only supported type is ``traceflow``.
The factory returns the image detector, decoder adapter, latent detector, owner
bits, and resolved scalar config used by training/sampling/evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.watermarking.image_watermark import ImageWatermarkDetector
from src.watermarking.decoder_watermark import TraceDecoderAdapter
from src.watermarking.latent_watermark import TraceLatentDetector
from src.watermarking.message import generate_watermark_bits


def build_watermark_modules(
    cfg: dict,
    image_size: int,
    device: Optional[torch.device] = None,
    channels: int = 3,
) -> Optional[Dict[str, Any]]:
    """Build final TraceFlow watermark modules from config.

    Returns ``None`` when watermarking is disabled.  Otherwise returns a dict
    with stable keys used by the rest of the project: ``extractor``,
    ``decoder_adapter``, ``latent_detector``, ``bits``, ``config``, and
    ``type='traceflow'``.
    """
    wm_cfg = cfg.get("watermark", cfg) if isinstance(cfg, dict) else {}
    if not wm_cfg or not wm_cfg.get("enabled", False):
        return None

    wm_type = wm_cfg.get("type", "traceflow")
    if wm_type != "traceflow":
        raise ValueError(
            f"Unsupported watermark.type: {wm_type!r}. The active paper codebase "
            "supports only 'traceflow'."
        )

    bit_length = int(wm_cfg.get("bit_length", 64))
    seed = int(wm_cfg.get("seed", 1234))
    alpha = float(wm_cfg.get("alpha", 0.02))
    hidden_dim = int(wm_cfg.get("extractor_hidden_dim", 256))
    latent_channels = int(wm_cfg.get("latent_channels", 4))
    latent_detector_hidden_dim = int(
        wm_cfg.get("latent_detector_hidden_dim", max(32, hidden_dim // 2))
    )

    extractor = ImageWatermarkDetector(
        bit_length=bit_length,
        image_size=image_size,
        channels=channels,
        hidden_dim=hidden_dim,
    )
    decoder_adapter = TraceDecoderAdapter(
        bit_length=bit_length,
        channels=channels,
        hidden_dim=hidden_dim,
        image_size=image_size,
    )
    latent_detector = TraceLatentDetector(
        bit_length=bit_length,
        latent_channels=latent_channels,
        hidden_dim=latent_detector_hidden_dim,
    )
    bits = generate_watermark_bits(bit_length, seed, device=device)

    if device is not None:
        extractor = extractor.to(device)
        decoder_adapter = decoder_adapter.to(device)
        latent_detector = latent_detector.to(device)

    resolved: Dict[str, Any] = {
        "enabled": True,
        "type": "traceflow",
        "bit_length": bit_length,
        "seed": seed,
        "alpha": alpha,
        "extractor_hidden_dim": hidden_dim,
        "latent_channels": latent_channels,
        "latent_detector_hidden_dim": latent_detector_hidden_dim,
        "lambda_wm_img": float(wm_cfg.get("lambda_wm_img", 1.0)),
        "lambda_wm_latent": float(wm_cfg.get("lambda_wm_latent", 1.0)),
        "lambda_img": float(wm_cfg.get("lambda_img", 0.1)),
        "lambda_cycle": float(wm_cfg.get("lambda_cycle", 0.1)),
        "lambda_residual": float(wm_cfg.get("lambda_residual", 0.01)),
        "cycle_target": str(wm_cfg.get("cycle_target", "protected_latent")),
        "message_mode": str(wm_cfg.get("message_mode", "random_per_sample")),
        "save_bits": bool(wm_cfg.get("save_bits", True)),
        "detection_threshold_acc": float(wm_cfg.get("detection_threshold_acc", 0.9)),
    }

    return {
        "enabled": True,
        "type": "traceflow",
        "bits": bits,
        "extractor": extractor,
        "decoder_adapter": decoder_adapter,
        "latent_detector": latent_detector,
        "config": resolved,
    }
