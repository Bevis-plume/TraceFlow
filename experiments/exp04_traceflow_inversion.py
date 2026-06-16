"""
experiments/exp04_traceflow_inversion.py
=============================================
Experiment 04 — Full TraceFlow + Inversion Evaluation

Config:  configs/defaults/exp04_traceflow.yml
         (keyed latent transform + TraceFlow dual-head watermark)

Steps
-----
1. Train full TraceFlow (keyed + dual-head watermark).
2. Run gradient-matching inversion attack (eval_traceflow_inversion.py).
3. Collect training + inversion metrics.

Key metrics
-----------
  Training:   generated_image_bit_acc, generated_latent_bit_acc,
              clean_false_positive_img
  Inversion:  final_gml, no_key_psnr, defender_psnr,
              raw_no_key_image_bit_acc, raw_defender_image_bit_acc,
              raw_no_key_latent_bit_acc

Threat model
------------
  The inversion attack uses ``attacker=no_key`` (realistic attacker without the
  secret key).  The defender uses the key for forensic latent inversion and
  dual-detector verification.  The secret_key is never written to output JSON.

CLI
---
    python -m experiments.exp04_traceflow_inversion --smoke
    python -m experiments.exp04_traceflow_inversion --smoke --attack latent --steps 3
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

EXP_ID = "exp04"
SMOKE_CONFIG = "configs/defaults/exp04_traceflow.yml"
FULL_CONFIG = "configs/defaults/exp04_traceflow.yml"


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

    # Attack settings: prefer explicit CLI value, fall back to smoke-safe defaults
    attack_type: str = getattr(args, "attack", "latent") or "latent"
    attack_steps: int = getattr(args, "steps", None) or (3 if smoke else 50)
    attacker: str = getattr(args, "attacker", "no_key") or "no_key"
    data_source = "random" if smoke else "config"

    output_base = Path(args.output_dir)
    exp_dir = make_exp_dir(output_base, EXP_ID, smoke)
    copy_config(Path(cfg_path), exp_dir)

    print(f"\n[{EXP_ID}] Config:        {cfg_path}")
    print(f"[{EXP_ID}] Run name:      {run_name}")
    print(f"[{EXP_ID}] Attack:        {attack_type} | steps={attack_steps} | attacker={attacker}")
    print(f"[{EXP_ID}] Exp dir:       {exp_dir}")

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
        print(f"\n[{EXP_ID}] Step 1/2: train full TraceFlow")
        result = run_command(train_cmd, dry_run=dry_run)
        if not dry_run and result["returncode"] != 0:
            return {"status": "error", "step": "train", "exp_id": EXP_ID}
        ckpt = resolve_checkpoint(run_name=run_name, config_path=cfg_path)
        train_out = train_output_dir(run_name, config_path=cfg_path)
    else:
        print(f"\n[{EXP_ID}] Step 1/2: reuse checkpoint -> {ckpt}")

    # ------------------------------------------------------------------ #
    # 2. Inversion evaluation                                              #
    # ------------------------------------------------------------------ #
    inversion_dir_name = "strong_inversion_geiping" if attack_type == "geiping_pixel" else "inversion"
    inversion_out = exp_dir / inversion_dir_name
    inversion_out.mkdir(parents=True, exist_ok=True)

    eval_cmd = [
        sys.executable, "-m", "scripts.eval_traceflow_inversion",
        "--config", cfg_path,
        "--checkpoint", str(ckpt),
        "--attack", attack_type,
        "--attacker", attacker,
        "--steps", str(attack_steps),
        # Inversion eval runs at batch size 1 to stay within 32GB VRAM (RTX 5090).
        "--batch-size", "1",
        "--data-source", data_source,
        "--output-dir", str(inversion_out),
    ]

    print(f"\n[{EXP_ID}] Step 2/2: inversion eval ({attack_type} attack, {attack_steps} steps)")
    eval_result = run_command(eval_cmd, dry_run=dry_run)
    eval_ok = dry_run or eval_result["returncode"] == 0

    # ------------------------------------------------------------------ #
    # 3. Collect metrics                                                   #
    # ------------------------------------------------------------------ #
    report_name = "smoke_report.json" if smoke else "train_report.json"
    report = load_json(train_out / report_name)
    last_step = read_last_jsonl(train_out / "train_log.jsonl")
    wm_report = report.get("watermark", {})

    # Parse inversion metrics
    inv_metrics = load_json(inversion_out / "metrics.json")
    attacker_runs = inv_metrics.get("attacker_runs", {})
    att_run = attacker_runs.get(attacker, {})
    latent_att = att_run.get("latent_attack", {})
    pixel_att = att_run.get("pixel_attack", {})

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
        # Training final eval (generated samples):
        "generated_image_bit_acc": wm_report.get("generated_image_bit_acc"),
        "generated_latent_bit_acc": wm_report.get("generated_latent_bit_acc"),
        "clean_false_positive_img": wm_report.get("clean_false_positive_img"),
        "clean_false_positive_latent": wm_report.get("clean_false_positive_latent"),
        "image_delta_mse": wm_report.get("image_delta_mse"),
        # Inversion eval:
        "attack_type": attack_type,
        "attack_steps": attack_steps,
        "attacker": attacker,
        "eval_ok": eval_ok,
        # Latent attack (headline metrics):
        "latent_final_gml": latent_att.get("final_gml"),
        "latent_no_key_psnr": latent_att.get("no_key_psnr"),
        "latent_no_key_ssim": latent_att.get("no_key_ssim"),
        "latent_defender_psnr": latent_att.get("defender_psnr"),
        "latent_defender_ssim": latent_att.get("defender_ssim"),
        "latent_raw_no_key_image_bit_acc": latent_att.get("raw_no_key_image_bit_acc"),
        "latent_raw_defender_image_bit_acc": latent_att.get("raw_defender_image_bit_acc"),
        "latent_raw_no_key_latent_bit_acc": latent_att.get("raw_no_key_latent_bit_acc"),
        "latent_raw_defender_latent_bit_acc": latent_att.get("raw_defender_latent_bit_acc"),
        # Pixel attack (if run):
        "pixel_final_gml": pixel_att.get("final_gml"),
        "pixel_psnr": pixel_att.get("psnr"),
        "pixel_ssim": pixel_att.get("ssim"),
        "pixel_raw_image_bit_acc": pixel_att.get("raw_pixel_image_bit_acc"),
        # Paths:
        "checkpoint": str(ckpt),
        "output_dir": str(train_out),
        "inversion_output_dir": str(inversion_out),
        "validation_warnings": validation["warnings"],
    }

    write_json(exp_dir / "metrics.json", metrics)
    append_csv_row(output_base / "all_metrics.csv", metrics)

    print(f"\n[{EXP_ID}] Metrics written: {exp_dir / 'metrics.json'}")
    if not dry_run:
        gml = metrics.get("latent_final_gml")
        nk_psnr = metrics.get("latent_no_key_psnr")
        def_psnr = metrics.get("latent_defender_psnr")
        img_acc = metrics.get("latent_raw_no_key_image_bit_acc")
        print(
            f"[{EXP_ID}] latent attack: final_gml={gml}  "
            f"no_key_psnr={nk_psnr}  defender_psnr={def_psnr}  "
            f"raw_no_key_image_bit_acc={img_acc}"
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
        "--attack", default="latent",
        choices=["latent", "pixel", "both"],
        help="Inversion attack type (default: latent)",
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help="Inversion attack optimisation steps (default: 3 smoke, 50 full)",
    )
    p.add_argument(
        "--attacker", default="no_key",
        choices=["no_key", "oracle_key", "both"],
        help="Attacker knowledge level (default: no_key)",
    )
    return p.parse_args()


if __name__ == "__main__":
    result = run(_parse_args())
    print(f"\n[{EXP_ID}] Done — status={result['status']}")
