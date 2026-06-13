"""
experiments/exp05_robustness.py
=================================
Experiment 05 — Watermark Robustness under Inversion

Config:  configs/defaults/exp05_robustness.yml  (keyed + TraceFlow watermark)

Runs the inversion eval across multiple attack types (latent and pixel) to
compare how each attack mode affects watermark detection.  Produces a single
metrics table that can be used to support the robustness claim in Section 5
of the experiment plan.

Steps
-----
1. Train full TraceFlow (smoke: 3 steps, DiT-XS).
2. Run latent inversion attack (no_key attacker).
3. Run pixel inversion attack (no_key attacker).
4. Aggregate and compare metrics.

Key output
----------
  ``metrics.json`` with side-by-side latent_attack and pixel_attack metrics.
  ``inversion/`` with full eval outputs from each attack.

CLI
---
    python -m experiments.exp05_robustness --smoke
    python -m experiments.exp05_robustness --smoke --steps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

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

EXP_ID = "exp05"
SMOKE_CONFIG = "configs/defaults/exp05_robustness.yml"
FULL_CONFIG = "configs/defaults/exp05_robustness.yml"


def _run_attack(
    cfg_path: str,
    ckpt: Path,
    attack_type: str,
    steps: int,
    smoke: bool,
    dry_run: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run a single inversion attack and return parsed metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_source = "random" if smoke else "config"
    cmd = [
        sys.executable, "-m", "scripts.eval_traceflow_inversion",
        "--config", cfg_path,
        "--checkpoint", str(ckpt),
        "--attack", attack_type,
        "--attacker", "no_key",
        "--steps", str(steps),
        "--batch-size", "1" if smoke else "2",
        "--data-source", data_source,
        "--output-dir", str(out_dir),
    ]
    result = run_command(cmd, dry_run=dry_run)
    ok = dry_run or result["returncode"] == 0
    data = load_json(out_dir / "metrics.json")
    return {"ok": ok, "data": data}


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
    steps: int = getattr(args, "steps", None) or (3 if smoke else 50)

    output_base = Path(args.output_dir)
    exp_dir = make_exp_dir(output_base, EXP_ID, smoke)
    copy_config(Path(cfg_path), exp_dir)

    print(f"\n[{EXP_ID}] Config:   {cfg_path}")
    print(f"[{EXP_ID}] Run name: {run_name}  |  attack_steps={steps}")

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
            reason="eval-only run has no TraceFlow checkpoint",
            validation_warnings=validation["warnings"],
        )

    # ------------------------------------------------------------------ #
    # 1. Train full TraceFlow or reuse checkpoint                         #
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
        print(f"\n[{EXP_ID}] Step 1/3: train")
        result = run_command(train_cmd, dry_run=dry_run)
        if not dry_run and result["returncode"] != 0:
            return {"status": "error", "step": "train", "exp_id": EXP_ID}
        ckpt = resolve_checkpoint(run_name=run_name, config_path=cfg_path)
        train_out = train_output_dir(run_name, config_path=cfg_path)
    else:
        print(f"\n[{EXP_ID}] Step 1/3: reuse checkpoint -> {ckpt}")

    # ------------------------------------------------------------------ #
    # 2. Latent inversion attack                                           #
    # ------------------------------------------------------------------ #
    print(f"\n[{EXP_ID}] Step 2/3: latent inversion attack ({steps} steps)")
    latent_out = exp_dir / "inversion_latent"
    latent_result = _run_attack(
        cfg_path, ckpt, "latent", steps, smoke, dry_run, latent_out
    )

    # ------------------------------------------------------------------ #
    # 3. Pixel inversion attack                                            #
    # ------------------------------------------------------------------ #
    print(f"\n[{EXP_ID}] Step 3/3: pixel inversion attack ({steps} steps)")
    pixel_out = exp_dir / "inversion_pixel"
    pixel_result = _run_attack(
        cfg_path, ckpt, "pixel", steps, smoke, dry_run, pixel_out
    )

    # ------------------------------------------------------------------ #
    # 4. Collect and aggregate metrics                                     #
    # ------------------------------------------------------------------ #
    report_name = "smoke_report.json" if smoke else "train_report.json"
    report = load_json(train_out / report_name)
    last_step = read_last_jsonl(train_out / "train_log.jsonl")
    wm_report = report.get("watermark", {})

    def _extract_attack(data: Dict[str, Any], attack_key: str) -> Dict[str, Any]:
        runs = data.get("attacker_runs", {}).get("no_key", {})
        return runs.get(attack_key, {})

    lat_m = _extract_attack(latent_result["data"], "latent_attack")
    pix_m = _extract_attack(pixel_result["data"], "pixel_attack")

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
        "generated_image_bit_acc": wm_report.get("generated_image_bit_acc"),
        "generated_latent_bit_acc": wm_report.get("generated_latent_bit_acc"),
        "attack_steps": steps,
        # Latent attack headline metrics:
        "latent_final_gml": lat_m.get("final_gml"),
        "latent_no_key_psnr": lat_m.get("no_key_psnr"),
        "latent_raw_no_key_image_bit_acc": lat_m.get("raw_no_key_image_bit_acc"),
        "latent_raw_no_key_latent_bit_acc": lat_m.get("raw_no_key_latent_bit_acc"),
        # Pixel attack headline metrics:
        "pixel_final_gml": pix_m.get("final_gml"),
        "pixel_psnr": pix_m.get("psnr"),
        "pixel_raw_image_bit_acc": pix_m.get("raw_pixel_image_bit_acc"),
        "pixel_raw_latent_bit_acc": pix_m.get("raw_pixel_latent_bit_acc"),
        "robustness": {
            "latent_attack": lat_m.get("robustness", {}),
            "pixel_attack": pix_m.get("robustness", {}),
        },
        "validation_warnings": validation["warnings"],
        # Paths:
        "checkpoint": str(ckpt),
        "output_dir": str(train_out),
        "latent_eval_dir": str(latent_out),
        "pixel_eval_dir": str(pixel_out),
    }

    write_json(exp_dir / "metrics.json", metrics)
    append_csv_row(output_base / "all_metrics.csv", metrics)

    print(f"\n[{EXP_ID}] Metrics written: {exp_dir / 'metrics.json'}")
    if not dry_run and lat_m:
        print(
            f"[{EXP_ID}] Latent: final_gml={lat_m.get('final_gml')}  "
            f"no_key_psnr={lat_m.get('no_key_psnr')}  "
            f"raw_img_acc={lat_m.get('raw_no_key_image_bit_acc')}"
        )
    if not dry_run and pix_m:
        print(
            f"[{EXP_ID}] Pixel:  final_gml={pix_m.get('final_gml')}  "
            f"psnr={pix_m.get('psnr')}  "
            f"raw_img_acc={pix_m.get('raw_pixel_image_bit_acc')}"
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
    p.add_argument(
        "--steps", type=int, default=None,
        help="Inversion attack optimisation steps (default: 3 smoke, 50 full)",
    )
    return p.parse_args()


if __name__ == "__main__":
    result = run(_parse_args())
    print(f"\n[{EXP_ID}] Done — status={result['status']}")
