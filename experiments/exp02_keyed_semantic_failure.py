"""
experiments/exp02_keyed_semantic_failure.py
============================================
Experiment 02 — Keyed Semantic Failure

Config:  configs/defaults/exp02_keyed.yml
         (keyed block-orthogonal latent transform, no watermark)

Demonstrates that the keyed latent transform prevents an attacker without the
secret key from obtaining meaningful images from the model's sample outputs:

  • WITH key  → defender decodes via latent_transform.invert(z0_k) → coherent
  • WITHOUT key → attacker decodes z0_k directly → noise

Steps
-----
1. Train FlowTransformer with keyed latent transform.
2. Sample WITH key (full config) → saves to ``with_key/``.
3. Sample WITHOUT key (temp config, secret_key stripped) → saves to ``no_key/``.

Key metrics
-----------
  steps_completed, model_params_M, final_loss, with_key_sample_ok, no_key_sample_ok

CLI
---
    python -m experiments.exp02_keyed_semantic_failure --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from experiments.common import (
    append_csv_row,
    checkpoint_exists,
    copy_config,
    infer_run_dir_from_checkpoint,
    load_json,
    make_exp_dir,
    print_validation_report,
    read_last_jsonl,
    resolve_checkpoint,
    run_command,
    select_default_config,
    should_train_for_policy,
    train_output_dir,
    validate_experiment_config,
    write_json,
    write_skipped_metrics,
)

EXP_ID = "exp02"
SMOKE_CONFIG = "configs/defaults/exp02_keyed.yml"
FULL_CONFIG = "configs/defaults/exp02_keyed.yml"


def _write_no_key_config(config_path: str) -> str:
    """Write a temp YAML identical to *config_path* but with ``secret_key`` removed.

    Returns the path to the temp file (caller is responsible for deletion).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    lt = cfg.get("security", {}).get("latent_transform", {})
    lt.pop("secret_key", None)
    fd, tmp_path = tempfile.mkstemp(suffix="_no_key.yml")
    os.close(fd)
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return tmp_path


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
    print(f"[{EXP_ID}] Run name: {run_name}")
    print(f"[{EXP_ID}] Exp dir:  {exp_dir}")

    validation = validate_experiment_config(
        config_path=cfg_path,
        run_name=run_name,
        resume=getattr(args, "resume", None),
    )
    if dry_run:
        print_validation_report(EXP_ID, validation)

    ckpt = resolve_checkpoint(
        run_name=run_name,
        config_path=cfg_path,
        explicit=getattr(args, "checkpoint", None) or getattr(args, "checkpoint_override", None),
    )
    train_policy = getattr(args, "train_policy", "always")
    train_out = infer_run_dir_from_checkpoint(ckpt, run_name=run_name, config_path=cfg_path)

    if train_policy == "never" and not checkpoint_exists(ckpt, dry_run=dry_run):
        return write_skipped_metrics(
            exp_id=EXP_ID,
            exp_dir=exp_dir,
            output_base=output_base,
            config_path=cfg_path,
            run_name=run_name,
            smoke=smoke,
            checkpoint=ckpt,
            reason="eval-only run has no checkpoint for this experiment",
            validation_warnings=validation["warnings"],
        )

    # ------------------------------------------------------------------ #
    # 1. Train or reuse checkpoint                                        #
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

    if should_train_for_policy(train_policy, ckpt, dry_run=dry_run):
        print(f"\n[{EXP_ID}] Step 1/3: train keyed model")
        result = run_command(train_cmd, dry_run=dry_run)
        if not dry_run and result["returncode"] != 0:
            return {"status": "error", "step": "train", "exp_id": EXP_ID}
        ckpt = resolve_checkpoint(run_name=run_name, config_path=cfg_path)
        train_out = train_output_dir(run_name, config_path=cfg_path)
    else:
        print(f"\n[{EXP_ID}] Step 1/3: reuse checkpoint -> {ckpt}")

    # ------------------------------------------------------------------ #
    # 2. Sample WITH key                                                   #
    # ------------------------------------------------------------------ #
    with_key_out = train_out / "samples_with_key"
    sample_with_key_cmd = [
        sys.executable, "-m", "scripts.sample_flow_transformer",
        "--config", cfg_path,
        "--checkpoint", str(ckpt),
        "--num-samples", "4" if smoke else "16",
        "--steps", "2" if smoke else "50",
        "--output-dir", str(with_key_out),
        "--seed", "42",
    ]

    print(f"\n[{EXP_ID}] Step 2/3: sample WITH key → {with_key_out}")
    run_command(sample_with_key_cmd, dry_run=dry_run)

    # ------------------------------------------------------------------ #
    # 3. Sample WITHOUT key                                                #
    # ------------------------------------------------------------------ #
    no_key_cfg = None
    try:
        if not dry_run:
            no_key_cfg = _write_no_key_config(cfg_path)
        else:
            no_key_cfg = cfg_path  # in dry-run, use original (won't execute)

        no_key_out = train_out / "samples_no_key"
        sample_no_key_cmd = [
            sys.executable, "-m", "scripts.sample_flow_transformer",
            "--config", no_key_cfg,
            "--checkpoint", str(ckpt),
            "--num-samples", "4" if smoke else "16",
            "--steps", "2" if smoke else "50",
            "--output-dir", str(no_key_out),
            "--seed", "42",
        ]

        print(f"\n[{EXP_ID}] Step 3/3: sample WITHOUT key → {no_key_out}")
        print(f"[{EXP_ID}] Note: no-key samples should be noise (protected latent space)")
        run_command(sample_no_key_cmd, dry_run=dry_run)
    finally:
        if no_key_cfg and no_key_cfg != cfg_path and os.path.exists(no_key_cfg):
            os.unlink(no_key_cfg)

    # ------------------------------------------------------------------ #
    # 4. Collect metrics                                                   #
    # ------------------------------------------------------------------ #
    report_name = "smoke_report.json" if smoke else "train_report.json"
    report = load_json(train_out / report_name)
    last_step = read_last_jsonl(train_out / "train_log.jsonl")

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
        "final_loss_flow": last_step.get("loss_flow"),
        "with_key_sample_dir": str(with_key_out),
        "no_key_sample_dir": str(no_key_out),
        "checkpoint": str(ckpt),
        "output_dir": str(train_out),
        "note": "no_key samples expected to decode as noise (protected latent space)",
        "validation_warnings": validation["warnings"],
    }

    write_json(exp_dir / "metrics.json", metrics)
    append_csv_row(output_base / "all_metrics.csv", metrics)

    print(f"\n[{EXP_ID}] Metrics written: {exp_dir / 'metrics.json'}")
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
