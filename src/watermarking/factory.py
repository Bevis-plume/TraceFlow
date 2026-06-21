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
    adapter_base_channels = int(wm_cfg.get("adapter_base_channels", 96))
    adapter_num_blocks = int(wm_cfg.get("adapter_num_blocks", 3))
    adapter_max_channels = int(wm_cfg.get("adapter_max_channels", 512))
    adapter_carrier_strength = float(wm_cfg.get("adapter_carrier_strength", 1.0))
    adapter_carrier_grid_size = int(wm_cfg.get("adapter_carrier_grid_size", 32))
    detector_base_channels = int(wm_cfg.get("detector_base_channels", 96))
    detector_num_scales = int(wm_cfg.get("detector_num_scales", 5))
    detector_num_blocks = int(wm_cfg.get("detector_num_blocks", 3))
    detector_max_channels = int(wm_cfg.get("detector_max_channels", 768))
    detector_carrier_grid_size = int(wm_cfg.get("detector_carrier_grid_size", 32))
    detector_carrier_weight = float(wm_cfg.get("detector_carrier_weight", 1.0))
    latent_channels = int(wm_cfg.get("latent_channels", 4))
    latent_detector_hidden_dim = int(
        wm_cfg.get("latent_detector_hidden_dim", max(32, hidden_dim // 2))
    )
    latent_detector_base_channels = int(
        wm_cfg.get("latent_detector_base_channels", max(64, detector_base_channels))
    )
    latent_detector_num_blocks = int(wm_cfg.get("latent_detector_num_blocks", 4))
    latent_detector_max_channels = int(wm_cfg.get("latent_detector_max_channels", 512))

    extractor = ImageWatermarkDetector(
        bit_length=bit_length,
        image_size=image_size,
        channels=channels,
        hidden_dim=hidden_dim,
        base_channels=detector_base_channels,
        num_scales=detector_num_scales,
        num_blocks=detector_num_blocks,
        max_channels=detector_max_channels,
        carrier_grid_size=detector_carrier_grid_size,
        carrier_weight=detector_carrier_weight,
    )
    decoder_adapter = TraceDecoderAdapter(
        bit_length=bit_length,
        channels=channels,
        hidden_dim=hidden_dim,
        image_size=image_size,
        base_channels=adapter_base_channels,
        num_blocks=adapter_num_blocks,
        max_channels=adapter_max_channels,
        carrier_strength=adapter_carrier_strength,
        carrier_grid_size=adapter_carrier_grid_size,
    )
    latent_detector = TraceLatentDetector(
        bit_length=bit_length,
        latent_channels=latent_channels,
        hidden_dim=latent_detector_hidden_dim,
        base_channels=latent_detector_base_channels,
        num_blocks=latent_detector_num_blocks,
        max_channels=latent_detector_max_channels,
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
        "adapter_base_channels": adapter_base_channels,
        "adapter_num_blocks": adapter_num_blocks,
        "adapter_max_channels": adapter_max_channels,
        "adapter_carrier_strength": adapter_carrier_strength,
        "adapter_carrier_grid_size": adapter_carrier_grid_size,
        "detector_base_channels": detector_base_channels,
        "detector_num_scales": detector_num_scales,
        "detector_num_blocks": detector_num_blocks,
        "detector_max_channels": detector_max_channels,
        "detector_carrier_grid_size": detector_carrier_grid_size,
        "detector_carrier_weight": detector_carrier_weight,
        "latent_channels": latent_channels,
        "latent_detector_hidden_dim": latent_detector_hidden_dim,
        "latent_detector_base_channels": latent_detector_base_channels,
        "latent_detector_num_blocks": latent_detector_num_blocks,
        "latent_detector_max_channels": latent_detector_max_channels,
        "lambda_wm_img": float(wm_cfg.get("lambda_wm_img", 1.0)),
        "lambda_wm_latent": float(wm_cfg.get("lambda_wm_latent", 1.0)),
        "lambda_img": float(wm_cfg.get("lambda_img", 0.1)),
        "lambda_cycle": float(wm_cfg.get("lambda_cycle", 0.1)),
        "lambda_residual": float(wm_cfg.get("lambda_residual", 0.01)),
        "lambda_wm_robust": float(wm_cfg.get("lambda_wm_robust", 0.0)),
        "lambda_clean_negative": float(wm_cfg.get("lambda_clean_negative", 0.0)),
        "clean_negative_batch_size": int(wm_cfg.get("clean_negative_batch_size", 0)),
        "clean_negative_chunk_size": int(wm_cfg.get("clean_negative_chunk_size", 0)),
        "lambda_perceptual": float(wm_cfg.get("lambda_perceptual", 0.0)),
        "lambda_frequency": float(wm_cfg.get("lambda_frequency", 0.0)),
        "robustness_enabled": bool(wm_cfg.get("robustness_enabled", False)),
        "robust_max_views": int(wm_cfg.get("robust_max_views", 2)),
        "robust_batch_size": int(wm_cfg.get("robust_batch_size", 0)),
        "robust_chunk_size": int(wm_cfg.get("robust_chunk_size", 0)),
        "robust_detach_input": bool(wm_cfg.get("robust_detach_input", True)),
        "robust_latent_enabled": bool(wm_cfg.get("robust_latent_enabled", False)),
        "image_warmup_detach_until": int(wm_cfg.get("image_warmup_detach_until", 0)),
        "carrier_schedule_enabled": bool(wm_cfg.get("carrier_schedule_enabled", False)),
        "carrier_schedule_start": int(wm_cfg.get("carrier_schedule_start", 0)),
        "carrier_schedule_steps": int(wm_cfg.get("carrier_schedule_steps", 1)),
        "carrier_schedule_min_scale": float(wm_cfg.get("carrier_schedule_min_scale", 0.0)),
        "schedule_warmup_steps": int(wm_cfg.get("schedule_warmup_steps", 0)),
        "schedule_main_steps": int(wm_cfg.get("schedule_main_steps", 5000)),
        "schedule_robust_start": int(wm_cfg.get("schedule_robust_start", 5000)),
        "schedule_robust_steps": int(wm_cfg.get("schedule_robust_steps", 30000)),
        "schedule_polish_start": int(wm_cfg.get("schedule_polish_start", 30000)),
        "schedule_polish_steps": int(wm_cfg.get("schedule_polish_steps", 5000)),
        "cycle_target": str(wm_cfg.get("cycle_target", "protected_latent")),
        "objective": str(wm_cfg.get("objective", "joint_watermark")),
        "trace_message_mode": str(wm_cfg.get("trace_message_mode", "fixed_owner")),
        "lambda_trace_img": float(wm_cfg.get("lambda_trace_img", 1.0)),
        "lambda_trace_latent": float(wm_cfg.get("lambda_trace_latent", 1.0)),
        "lambda_trace_clean_negative": float(wm_cfg.get("lambda_trace_clean_negative", 2.0)),
        "trace_positive_source": str(wm_cfg.get("trace_positive_source", "inversion_outputs")),
        "trace_negative_sources": list(wm_cfg.get("trace_negative_sources", [])),
        "trace_clean_fp_target": float(wm_cfg.get("trace_clean_fp_target", 0.55)),
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
