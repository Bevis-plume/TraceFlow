"""
scripts/evaluate_paper_metrics
==============================
Bundle-level paper metrics for TraceFlow.

This script aggregates generated samples, watermark metrics, inversion metrics,
and optional distribution metrics (FID/KID/Inception Score) into reports that can
be downloaded directly with the run bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from src.utils.quality_metrics import distribution_metrics


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_sample_dirs(bundle: Path) -> List[Path]:
    out_root = bundle / "outputs"
    if not out_root.exists():
        return []
    dirs = []
    for path in out_root.rglob("sample_0000.png"):
        dirs.append(path.parent)
    return sorted(set(dirs))


def _flatten(prefix: str, data: Dict[str, Any], out: Dict[str, Any]) -> None:
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(key, v, out)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v


def evaluate(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle_dir)
    config_path = Path(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else bundle / "reports" / "paper_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assets = cfg.get("assets", {}) or {}
    metric_weights = assets.get("metric_weights", {}) or {}
    if metric_weights.get("torch_home"):
        os.environ.setdefault("TORCH_HOME", str(Path(metric_weights["torch_home"]).resolve()))
    if metric_weights.get("hf_home"):
        os.environ.setdefault("HF_HOME", str(Path(metric_weights["hf_home"]).resolve()))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(metric_weights["hf_home"]).resolve()))
    data_cfg = cfg.get("data", {})
    real_root = Path(str(data_cfg.get("root", "")))
    image_size = int(data_cfg.get("image_size", 256))
    max_images = int(args.max_images)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    summary: Dict[str, Any] = {
        "bundle": str(bundle),
        "config": str(config_path),
        "data_root": str(real_root),
        "image_size": image_size,
        "max_images": max_images,
    }
    warnings: List[str] = []

    # Distribution metrics for any generated sample dirs.
    sample_dirs = _find_sample_dirs(bundle)
    dist_rows = []
    for sample_dir in sample_dirs:
        if not real_root.exists():
            warnings.append(f"real data root missing; skipped distribution metrics for {sample_dir}")
            continue
        try:
            metrics = distribution_metrics(real_root, sample_dir, image_size=image_size, max_images=max_images, device=device)
        except Exception as exc:
            metrics = {"warning": str(exc)}
        row = {"sample_dir": str(sample_dir), **metrics}
        dist_rows.append(row)
    summary["distribution_metrics"] = dist_rows

    # Collect watermark metrics emitted by sample_flow_transformer.
    wm_rows = []
    for wm_path in sorted((bundle / "outputs").rglob("watermark_metrics.json")) if (bundle / "outputs").exists() else []:
        row = {"path": str(wm_path)}
        _flatten("", _load_json(wm_path), row)
        wm_rows.append(row)
    summary["watermark_metrics"] = wm_rows

    # Collect exp metrics.
    exp_rows = []
    for metrics_path in sorted((bundle / "results").glob("exp*/metrics.json")) if (bundle / "results").exists() else []:
        data = _load_json(metrics_path)
        row = {"path": str(metrics_path), "exp_id": metrics_path.parent.name}
        headline_keys = [
            "status", "generated_image_bit_acc", "generated_latent_bit_acc",
            "clean_false_positive_img", "image_delta_mse",
            "latent_no_key_psnr", "latent_no_key_ssim", "latent_no_key_lpips",
            "latent_defender_psnr", "latent_defender_ssim", "latent_defender_lpips",
            "latent_raw_no_key_image_bit_acc", "latent_raw_no_key_latent_bit_acc",
            "pixel_psnr", "pixel_ssim", "pixel_lpips",
            "pixel_raw_image_bit_acc", "pixel_raw_latent_bit_acc",
        ]
        for key in headline_keys:
            if key in data:
                row[key] = data.get(key)
        exp_rows.append(row)
    summary["experiment_metrics"] = exp_rows
    summary["warnings"] = warnings

    (out_dir / "paper_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # CSVs for quick paper tables.
    for name, rows in (("distribution_metrics.csv", dist_rows), ("watermark_metrics.csv", wm_rows), ("experiment_metrics.csv", exp_rows)):
        if not rows:
            continue
        keys = sorted({k for row in rows for k in row.keys()})
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# TraceFlow Paper Metrics", "", f"Bundle: `{bundle}`", f"Data root: `{real_root}`", ""]
    lines.append("## Distribution Metrics")
    if dist_rows:
        for row in dist_rows:
            lines.append(f"- `{row['sample_dir']}`: FID={row.get('fid')} KID={row.get('kid_mean')} IS={row.get('inception_score_mean')} warning={row.get('warning')}")
    else:
        lines.append("- No generated sample directories found or distribution metrics skipped.")
    lines.append("\n## Watermark Metrics")
    if wm_rows:
        for row in wm_rows:
            lines.append(f"- `{row['path']}`: image_acc={row.get('generated_image_bit_acc')} latent_acc={row.get('generated_latent_bit_acc')} psnr={row.get('watermarked_vs_clean_psnr')} ssim={row.get('watermarked_vs_clean_ssim')} lpips={row.get('watermarked_vs_clean_lpips')}")
    else:
        lines.append("- No watermark_metrics.json files found.")
    lines.append("\n## Experiment Metrics")
    if exp_rows:
        for row in exp_rows:
            lines.append(f"- {row['exp_id']}: status={row.get('status')} no_key_psnr={row.get('latent_no_key_psnr')} raw_img_acc={row.get('latent_raw_no_key_image_bit_acc')} pixel_psnr={row.get('pixel_psnr')}")
    else:
        lines.append("- No exp metrics found.")
    if warnings:
        lines.append("\n## Warnings")
        lines.extend(f"- {w}" for w in warnings)
    (out_dir / "paper_metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[paper-metrics] wrote {out_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute TraceFlow paper-level quality metrics for a bundle.")
    p.add_argument("--config", required=True)
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-images", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(evaluate(_parse_args()))
