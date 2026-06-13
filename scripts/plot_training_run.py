"""
scripts/plot_training_run.py
============================
Generate per-run TraceFlow training visualisations from a run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection

from src.utils.plotting import METRIC_COLORS, load_image, placeholder_axis, save_figure, setup_style


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
                    continue
    except FileNotFoundError:
        pass
    return rows


def _series(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for r in rows:
        if r.get(key) is not None and r.get("step") is not None:
            xs.append(int(r["step"]))
            ys.append(float(r[key]))
    return xs, ys


def _plot_lines(rows: Sequence[Dict[str, Any]], specs: Sequence[Tuple[str, str, str]], title: str, ylabel: str, out_stem: Path, chance: bool = False) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    any_line = False
    for key, label, color_key in specs:
        xs, ys = _series(rows, key)
        if xs:
            any_line = True
            ax.plot(xs, ys, label=label, color=METRIC_COLORS.get(color_key, None), marker="o", markersize=3)
    if not any_line:
        placeholder_axis(ax, "no train_log.jsonl data")
    else:
        if chance:
            ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2, label="random chance")
            ax.set_ylim(0, 1.05)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, out_stem)


def _runtime_plot(rows: Sequence[Dict[str, Any]], report: Dict[str, Any], out_stem: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    xs, elapsed = _series(rows, "elapsed_s")
    if xs:
        axes[0].plot(xs, elapsed, color="#4C72B0", marker="o", markersize=3)
        axes[0].set_title("Elapsed wall time")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Seconds")
    else:
        placeholder_axis(axes[0], "no elapsed data")

    axes[1].axis("off")
    text = (
        f"Run: {report.get('run_name', '(unknown)')}\n"
        f"Steps: {report.get('steps_completed', '')}\n"
        f"Device: {report.get('device', '')}\n"
        f"Params (M): {report.get('model_params_M', '')}\n"
        f"Total time (s): {report.get('total_train_wall_time_s', '')}\n"
        f"Avg step (s): {report.get('avg_step_time_s', '')}\n"
        f"CUDA max memory (MB): {report.get('cuda_max_memory_allocated_MB', '')}"
    )
    axes[1].text(0.02, 0.95, text, ha="left", va="top", family="monospace", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#cccccc"))
    fig.suptitle("Runtime and resource summary", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_stem)


def _sample_timeline(run_dir: Path, out_path: Path) -> None:
    setup_style()
    samples = sorted(run_dir.glob("samples_step*.png"))
    samples = [p for p in samples if "watermarked" not in p.name][:8]
    wm_samples = sorted(run_dir.glob("samples_step*_watermarked.png"))[:8]
    rows = []
    if samples:
        rows.append(("Generated", samples))
    if wm_samples:
        rows.append(("Watermarked", wm_samples))
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        placeholder_axis(ax, "no sample grids")
        save_figure(fig, out_path.with_suffix(""), formats=("png",))
        return
    ncols = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(max(4, 2.2 * ncols), 2.6 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for row_idx, (label, paths) in enumerate(rows):
        row_axes = axes[row_idx]
        if ncols == 1:
            row_axes = [row_axes]
        for col_idx in range(ncols):
            ax = row_axes[col_idx]
            ax.axis("off")
            if col_idx < len(paths):
                img = load_image(paths[col_idx])
                if img is not None:
                    ax.imshow(img)
                ax.set_title(paths[col_idx].stem.replace("samples_", ""), fontsize=9)
            if col_idx == 0:
                ax.text(-0.05, 0.5, label, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontweight="bold")
    fig.suptitle("Sample timeline", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _latent_trajectory_3d(run_dir: Path, out_stem: Path) -> None:
    setup_style()
    paths = sorted(run_dir.glob("latent_trajectory_step*.json"))
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    if not paths:
        ax.text2D(0.5, 0.5, "no latent trajectory JSON found", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        save_figure(fig, out_stem)
        return
    data = _read_json(paths[-1])
    points = data.get("points", [])
    by_sample: Dict[int, List[Dict[str, Any]]] = {}
    for p in points:
        by_sample.setdefault(int(p.get("sample", 0)), []).append(p)
    for sample, pts in sorted(by_sample.items()):
        pts = sorted(pts, key=lambda x: x.get("step_index", 0))
        xs = [p.get("pc1", 0.0) for p in pts]
        ys = [p.get("pc2", 0.0) for p in pts]
        zs = [p.get("pc3", 0.0) for p in pts]
        ax.plot(xs, ys, zs, marker="o", markersize=2.5, linewidth=1.4, label=f"sample {sample}")
        if xs:
            ax.scatter([xs[0]], [ys[0]], [zs[0]], marker="x", s=45, color="black")
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], marker="o", s=45)
    ax.set_title("Latent reverse-flow trajectory (PCA 3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_stem)


def _write_summary(out_dir: Path, rows: Sequence[Dict[str, Any]], report: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    last = rows[-1] if rows else {}
    summary = {
        "run_name": report.get("run_name"),
        "steps_completed": report.get("steps_completed"),
        "device": report.get("device"),
        "model_params_M": report.get("model_params_M"),
        "final_loss": last.get("loss"),
        "final_loss_flow": last.get("loss_flow"),
        "final_bit_acc_img": last.get("bit_acc_img"),
        "final_bit_acc_latent": last.get("bit_acc_latent"),
        "final_loss_wm_robust": last.get("loss_wm_robust"),
        "final_loss_clean_negative": last.get("loss_clean_negative"),
        "final_loss_perceptual": last.get("loss_perceptual"),
        "final_loss_frequency": last.get("loss_frequency"),
        "total_train_wall_time_s": report.get("total_train_wall_time_s"),
        "avg_step_time_s": report.get("avg_step_time_s"),
        "cuda_max_memory_allocated_MB": report.get("cuda_max_memory_allocated_MB"),
    }
    with open(out_dir / "training_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    lines = ["| Metric | Value |", "|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {'' if v is None else v} |")
    (out_dir / "training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_training_run(run_dir: Path, output_dir: Path) -> None:
    rows = _read_jsonl(run_dir / "train_log.jsonl")
    report = _read_json(run_dir / "train_report.json") or _read_json(run_dir / "smoke_report.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_lines(
        rows,
        [
            ("loss", "total", "loss"),
            ("loss_flow", "flow", "loss_flow"),
            ("loss_wm_img", "image wm", "image_detector"),
            ("loss_wm_latent", "latent wm", "latent_detector"),
            ("loss_wm_robust", "robust wm", "image_bit_acc"),
            ("loss_clean_negative", "clean negative", "clean_false_positive"),
            ("loss_perceptual", "perceptual", "gml"),
            ("loss_frequency", "frequency", "loss"),
            ("loss_img", "image recon", "gml"),
            ("loss_cycle", "cycle", "clean_false_positive"),
            ("loss_residual", "residual", "loss"),
        ],
        "TraceFlow loss curves",
        "Loss",
        output_dir / "loss_curves",
    )
    _plot_lines(
        rows,
        [
            ("bit_acc_img", "image bit acc", "image_bit_acc"),
            ("bit_acc_latent", "latent bit acc", "latent_bit_acc"),
            ("ber_img", "image BER", "loss"),
            ("ber_latent", "latent BER", "gml"),
            ("clean_false_positive_img", "clean FP image", "clean_false_positive"),
            ("clean_false_positive_latent", "clean FP latent", "latent_detector"),
            ("wm_schedule_main", "schedule main", "image_detector"),
            ("wm_schedule_robust", "schedule robust", "image_bit_acc"),
            ("wm_schedule_polish", "schedule polish", "clean_false_positive"),
        ],
        "Watermark detector curves",
        "Accuracy / BER",
        output_dir / "watermark_curves",
        chance=True,
    )
    _runtime_plot(rows, report, output_dir / "runtime_memory")
    _sample_timeline(run_dir, output_dir / "sample_timeline.png")
    _latent_trajectory_3d(run_dir, output_dir / "latent_trajectory_3d")
    _write_summary(output_dir, rows, report)
    print(f"[plot-training] wrote figures and summaries to {output_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-run TraceFlow training plots.")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_training_run(args.run_dir, args.output_dir)
