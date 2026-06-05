"""
experiments/exp03_decoder_watermark.py
========================================
Experiment 03 — Decoder Watermark (identity transform + traceflow)

Config:  configs/defaults/exp03_traceflow_identity.yml
         (identity latent transform, TraceFlow dual-head watermark)

This ablation isolates the TraceFlow watermark contribution without the keyed
latent transform.  Compare with exp04 (full pipeline, keyed transform)
to measure the effect of the key on forensic attribution.

Steps
-----
1. Train with traceflow_identity config.
2. Sample from trained checkpoint (watermark evaluation included in report).
3. Collect watermark metrics from the training report.

Key metrics
-----------
  generated_image_bit_acc, generated_latent_bit_acc,
  clean_false_positive_img, clean_false_positive_latent,
  image_delta_mse

CLI
---
    python -m experiments.exp03_decoder_watermark --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from experiments.common import (
    append_csv_row,
    checkpoint_path,
    copy_config,
    load_json,
    make_exp_dir,
    print_validation_report,
    read_last_jsonl,
    run_command,
    select_default_config,
    train_output_dir,
    validate_experiment_config,
    write_json,
)

EXP_ID = "exp03"
SMOKE_CONFIG = "configs/defaults/exp03_traceflow_identity.yml"
FULL_CONFIG = "configs/defaults/exp03_traceflow_identity.yml"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    smoke: bool = args.smoke
    dry_run: bool = args.dry_run
    cfg_path = select_default_config(
        smoke=smoke,
        override=getattr(args, "config", None),
        smoke_config=SMOKE_CONFIG,
        full_config=FULL_CONFIG,
    )
    run_name = getattr(args, "run_name", None) or f"{EXP_ID}_{'smoke' if smoke else 'full'}"

    output_base = Path(args.output_dir)
    exp_dir = make_exp_dir(output_base, EXP_ID, smoke)
    copy_config(Path(cfg_path), exp_dir)

    print(f"\n[{EXP_ID}] Config:   {cfg_path}")
    print(f"\n[{EXP_ID}] Note: identity transform + TraceFlow dual-head watermark")

    validation = validate_experiment_config(
        config_path=cfg_path,
        run_name=run_name,
        resume=getattr(args, "resume", None),
    )
    if dry_run:
        print_validation_report(EXP_ID, validation)

    # ------------------------------------------------------------------ #
    # 1. Train                                                             #
    # ------------------------------------------------------------------ #
    train_cmd = [
        sys.executable, "-m", "scripts.train_flow_transformer",
        "--config", cfg_path,
        "--run-name", run_name,
    ]
    if getattr(args, "resume", None):
        train_cmd.extend(["--resume", args.resume])
    if smoke:
        train_cmd.append("--smoke")

    print(f"\n[{EXP_ID}] Step 1/2: train with TraceFlow watermark")
    result = run_command(train_cmd, dry_run=dry_run)
    if not dry_run and result["returncode"] != 0:
        return {"status": "error", "step": "train", "exp_id": EXP_ID}

    # ------------------------------------------------------------------ #
    # 2. Sample                                                            #
    # ------------------------------------------------------------------ #
    ckpt = checkpoint_path(run_name, config_path=cfg_path)
    train_out = train_output_dir(run_name, config_path=cfg_path)
    sample_out = train_out / "exp_samples"

    sample_cmd = [
        sys.executable, "-m", "scripts.sample_flow_transformer",
        "--config", cfg_path,
        "--checkpoint", str(ckpt),
        "--num-samples", "4" if smoke else "16",
        "--steps", "2" if smoke else "50",
        "--output-dir", str(sample_out),
        "--seed", "42",
    ]

    print(f"\n[{EXP_ID}] Step 2/2: sample + watermark eval")
    run_command(sample_cmd, dry_run=dry_run)

    # ------------------------------------------------------------------ #
    # 3. Collect metrics                                                   #
    # ------------------------------------------------------------------ #
    report_name = "smoke_report.json" if smoke else "train_report.json"
    report = load_json(train_out / report_name)
    last_step = read_last_jsonl(train_out / "train_log.jsonl")
    wm_report = report.get("watermark", {})

    metrics: Dict[str, Any] = {
        "exp_id": EXP_ID,
        "config": cfg_path,
        "smoke": smoke,
        "run_name": run_name,
        "status": "dry_run" if dry_run else report.get("status", "ok"),
        "steps_completed": report.get("steps_completed"),
        "model_params_M": report.get("model_params_M"),
        "device": report.get("device"),
        "final_loss": last_step.get("loss"),
        "final_bit_acc_img": last_step.get("bit_acc_img"),
        "final_bit_acc_latent": last_step.get("bit_acc_latent"),
        # From the training final eval (generated samples, not training batch):
        "generated_image_bit_acc": wm_report.get("generated_image_bit_acc"),
        "generated_latent_bit_acc": wm_report.get("generated_latent_bit_acc"),
        "clean_false_positive_img": wm_report.get("clean_false_positive_img"),
        "clean_false_positive_latent": wm_report.get("clean_false_positive_latent"),
        "image_delta_mse": wm_report.get("image_delta_mse"),
        "wm_type": wm_report.get("type"),
        "checkpoint": str(ckpt),
        "output_dir": str(train_out),
        "validation_warnings": validation["warnings"],
    }

    write_json(exp_dir / "metrics.json", metrics)
    append_csv_row(output_base / "all_metrics.csv", metrics)

    print(f"\n[{EXP_ID}] Metrics written: {exp_dir / 'metrics.json'}")
    if not dry_run and wm_report:
        print(
            f"[{EXP_ID}] generated_image_bit_acc={metrics['generated_image_bit_acc']}  "
            f"generated_latent_bit_acc={metrics['generated_latent_bit_acc']}  "
            f"clean_false_positive_img={metrics['clean_false_positive_img']}"
        )

    return {
        "status": metrics["status"],
        "exp_id": EXP_ID,
        "metrics": metrics,
        "output_dir": str(exp_dir),
    }


# ---------------------------------------------------------------------------
# Direct CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Run experiment {EXP_ID}.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--output-dir", dest="output_dir", default="results/traceflow")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


if __name__ == "__main__":
    result = run(_parse_args())
    print(f"\n[{EXP_ID}] Done — status={result['status']}")
