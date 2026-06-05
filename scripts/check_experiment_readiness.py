"""
scripts/check_experiment_readiness.py
=====================================
Lightweight readiness checker for TraceFlow experiment outputs.

The checker is intentionally advisory by default: it prints warnings for missing
metrics or figures and exits zero unless ``--strict`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_EXPERIMENTS = ["exp01", "exp02", "exp03", "exp04", "exp05"]
REQUIRED_FIGURES = [
    "fig1_pipeline.png",
    "fig2_ablation.png",
    "fig3_attack_grid.png",
    "fig4_curves.png",
    "fig5_robustness.png",
    "fig6_training_dashboard.png",
    "summary.csv",
    "summary.md",
]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _mode_dir(exp_dir: Path) -> Path | None:
    for mode in ("full", "smoke"):
        candidate = exp_dir / mode
        if (candidate / "metrics.json").exists():
            return candidate
    return None


def _warn(warnings: List[str], msg: str) -> None:
    warnings.append(msg)
    print(f"[readiness] warning: {msg}")


def check(results_dir: Path) -> List[str]:
    warnings: List[str] = []
    print(f"[readiness] results_dir={results_dir}")

    if not results_dir.exists():
        _warn(warnings, f"results directory does not exist: {results_dir}")
        return warnings

    for exp_id in REQUIRED_EXPERIMENTS:
        exp_dir = results_dir / exp_id
        mode_dir = _mode_dir(exp_dir)
        if mode_dir is None:
            _warn(warnings, f"missing metrics for {exp_id}")
            continue
        metrics = _load_json(mode_dir / "metrics.json")
        if not metrics:
            _warn(warnings, f"invalid metrics JSON for {exp_id}")
            continue
        print(f"[readiness] {exp_id}: found {mode_dir / 'metrics.json'}")

        if exp_id in {"exp03", "exp04", "exp05"}:
            if metrics.get("generated_image_bit_acc") is None:
                _warn(warnings, f"{exp_id} missing generated_image_bit_acc")
            if metrics.get("generated_latent_bit_acc") is None:
                _warn(warnings, f"{exp_id} missing generated_latent_bit_acc")
        if exp_id == "exp04":
            for key in ("latent_final_gml", "latent_raw_no_key_image_bit_acc", "latent_raw_no_key_latent_bit_acc"):
                if metrics.get(key) is None:
                    _warn(warnings, f"exp04 missing inversion metric: {key}")
        if exp_id == "exp05":
            robustness = metrics.get("robustness")
            if not isinstance(robustness, dict) or not robustness:
                _warn(warnings, "exp05 missing robustness block")

        config_text = ""
        cfg_path = mode_dir / "run_config.yml"
        if cfg_path.exists():
            config_text = cfg_path.read_text()
        if "CHANGE_ME" in config_text or "DEV_ONLY" in config_text:
            _warn(warnings, f"{exp_id} run_config still contains placeholder key text")

    figures_dir = results_dir / "figures"
    for fig in REQUIRED_FIGURES:
        if not (figures_dir / fig).exists():
            _warn(warnings, f"missing figure artifact: figures/{fig}")

    if warnings:
        print(f"[readiness] completed with {len(warnings)} warning(s)")
    else:
        print("[readiness] all required experiment outputs and figures are present")
    return warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TraceFlow experiment readiness.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on warnings.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    warnings = check(args.results_dir)
    if args.strict and warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()
