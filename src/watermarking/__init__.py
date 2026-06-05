"""
src/watermarking
================
Final TraceFlow watermarking modules.

The active paper method uses a bit-conditioned decoder adapter, an image-domain
detector, and a latent-domain detector tied together by a re-encode cycle.
"""

from __future__ import annotations

from src.watermarking.message import generate_watermark_bits, expand_bits
from src.watermarking.image_watermark import ImageWatermarkDetector
from src.watermarking.decoder_watermark import TraceDecoderAdapter, TraceDecoderWrapper
from src.watermarking.latent_watermark import TraceLatentDetector
from src.watermarking.metrics import (
    bit_accuracy,
    bit_error_rate,
    detection_passed,
    image_delta_mse,
)
from src.watermarking.factory import build_watermark_modules

__all__ = [
    "generate_watermark_bits",
    "expand_bits",
    "ImageWatermarkDetector",
    "TraceDecoderAdapter",
    "TraceDecoderWrapper",
    "TraceLatentDetector",
    "bit_accuracy",
    "bit_error_rate",
    "detection_passed",
    "image_delta_mse",
    "build_watermark_modules",
]
