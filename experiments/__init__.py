"""
experiments/
=============
TraceFlow experiment modules. Each module exposes ``run(args) -> dict`` and can
also be launched directly with ``python -m experiments.expXX_name``.

Available experiments
---------------------
exp01  generation_baseline          Baseline generation model
exp02  keyed_semantic_failure       Keyed latent transform without watermark
exp03  traceflow_identity           TraceFlow watermark with identity transform
exp04  traceflow_inversion          Full TraceFlow + inversion evaluation
exp05  robustness                   Latent and pixel attack robustness
"""

from __future__ import annotations

REGISTRY: dict[str, str] = {
    "exp01": "experiments.exp01_generation_baseline",
    "exp02": "experiments.exp02_keyed_semantic_failure",
    "exp03": "experiments.exp03_decoder_watermark",
    "exp04": "experiments.exp04_traceflow_inversion",
    "exp05": "experiments.exp05_robustness",
}

__all__ = ["REGISTRY"]
