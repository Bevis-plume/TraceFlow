"""
scripts.analyze_training_loss
=============================
Diagnose TraceFlow training curves, especially apparent total-loss plateaus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def _find_run_dir(root: Path) -> Optional[Path]:
    if (root / "train_log.jsonl").exists():
        return root
    candidates = sorted((root / "outputs").glob("*/train_log.jsonl"))
    if candidates:
        return candidates[-1].parent
    candidates = sorted(root.glob("outputs/*/train_log.jsonl"))
    if candidates:
        return candidates[-1].parent
    return None


def _tail_mean(rows: List[Dict[str, Any]], key: str, frac: float = 0.2) -> Optional[float]:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    if not values:
        return None
    n = max(1, int(len(values) * frac))
    return mean(values[-n:])


def _slope(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    pts = [(float(r["step"]), float(r[key])) for r in rows if r.get("step") is not None and r.get(key) is not None]
    if len(pts) < 2:
        return None
    n = max(2, int(len(pts) * 0.3))
    pts = pts[-n:]
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    if x1 == x0:
        return None
    return (y1 - y0) / (x1 - x0)


def analyze(run_dir: Path, output_dir: Path) -> Dict[str, Any]:
    actual_run_dir = _find_run_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if actual_run_dir is None:
        result = {
            "status": "missing_log",
            "summary": "No train_log.jsonl was found. Run training before diagnosis.",
        }
        _write_outputs(result, output_dir)
        return result

    rows = _read_jsonl(actual_run_dir / "train_log.jsonl")
    report = _read_json(actual_run_dir / "train_report.json") or _read_json(actual_run_dir / "smoke_report.json")
    last = rows[-1] if rows else {}

    keys = [
        "loss", "loss_flow", "loss_wm_img", "loss_wm_latent", "loss_img",
        "loss_cycle", "loss_residual", "bit_acc_img", "bit_acc_latent",
        "ber_img", "ber_latent", "clean_false_positive_img",
        "clean_false_positive_latent", "learning_rate",
    ]
    tail = {k: _tail_mean(rows, k) for k in keys}
    slopes = {k: _slope(rows, k) for k in ("loss", "loss_flow", "loss_wm_img", "loss_wm_latent")}

    warnings: List[str] = []
    notes: List[str] = []
    if tail.get("loss") is not None and tail["loss"] >= 0.4:
        notes.append(
            "Total loss above 0.4 is not automatically a bug: TraceFlow optimizes a composite "
            "rectified-flow + watermark + reconstruction objective, not a classifier cross-entropy."
        )
    if tail.get("bit_acc_img") is not None and tail["bit_acc_img"] < 0.65:
        warnings.append("Image watermark bit accuracy is close to random; check lambda_wm_img, alpha, and detector capacity.")
    if tail.get("bit_acc_latent") is not None and tail["bit_acc_latent"] < 0.65:
        warnings.append("Latent watermark bit accuracy is close to random; check re-encode cycle and latent detector learning.")
    if tail.get("loss_wm_img") is not None and tail["loss_wm_img"] > 0.65:
        warnings.append("Image watermark BCE remains near 0.69; the image detector may not be learning.")
    if tail.get("loss_wm_latent") is not None and tail["loss_wm_latent"] > 0.65:
        warnings.append("Latent watermark BCE remains near 0.69; the latent detector may not be learning.")
    if tail.get("learning_rate") is not None and tail["learning_rate"] <= 0:
        warnings.append("Learning rate is zero in the logged tail.")
    if slopes.get("loss_flow") is not None and abs(slopes["loss_flow"]) < 1e-7:
        notes.append("Flow loss is nearly flat in the latest window; judge by samples and validation metrics, not total loss alone.")

    if warnings:
        verdict = "needs_attention"
    elif tail.get("loss") is None:
        verdict = "insufficient_data"
    else:
        verdict = "plausibly_normal_plateau"

    result = {
        "status": verdict,
        "run_dir": str(actual_run_dir),
        "run_name": report.get("run_name"),
        "steps_completed": report.get("steps_completed") or last.get("step"),
        "last": last,
        "tail_mean": tail,
        "tail_slope_per_step": slopes,
        "warnings": warnings,
        "notes": notes,
    }
    _write_outputs(result, output_dir)
    return result


def _write_outputs(result: Dict[str, Any], output_dir: Path) -> None:
    with open(output_dir / "loss_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    lines = [
        "# TraceFlow Training Loss Diagnosis",
        "",
        f"Status: `{result.get('status')}`",
        f"Run: `{result.get('run_name', '')}`",
        f"Steps: `{result.get('steps_completed', '')}`",
        "",
        "## Interpretation",
    ]
    notes = result.get("notes") or []
    warnings = result.get("warnings") or []
    if not notes and not warnings:
        lines.append("No obvious curve-level issue was detected. Inspect sample grids and final metrics before claiming convergence.")
    for note in notes:
        lines.append(f"- {note}")
    for warning in warnings:
        lines.append(f"- WARNING: {warning}")
    lines.extend(["", "## Tail Means", "", "| Metric | Value |", "|---|---:|"])
    for key, value in (result.get("tail_mean") or {}).items():
        lines.append(f"| {key} | {'' if value is None else value:.6g} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines.extend(["", "## Why 0.48 Can Be Normal", ""])
    lines.append(
        "TraceFlow's logged total loss combines rectified-flow denoising, image/latent watermark BCE, image reconstruction, "
        "cycle consistency, and residual regularization. It is therefore not comparable to examples where a single supervised "
        "loss drops to 0.0x. A plateau is suspicious mainly when detector accuracies stay near 0.5, BER stays high, or samples "
        "do not improve."
    )
    (output_dir / "loss_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[loss-diagnosis] wrote {output_dir / 'loss_diagnosis.md'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose TraceFlow training loss curves.")
    p.add_argument("--run-dir", required=True, type=Path, help="Bundle root or one training run directory.")
    p.add_argument("--output-dir", type=Path, default=None, help="Report directory; defaults to <run-dir>/reports.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(args.run_dir, args.output_dir or (args.run_dir / "reports"))
