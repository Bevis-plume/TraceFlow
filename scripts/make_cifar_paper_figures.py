"""
scripts/make_cifar_paper_figures.py
===================================
Curated, method-named paper visualisation for the CIFAR-10 32x32 TraceFlow run.

Unlike ``scripts/make_traceflow_figures.py`` (which labels things exp01-exp05),
this pipeline emits **method names only** in every generated paper figure:

    Baseline Generator | Keyed Latent | TraceFlow Identity | Full TraceFlow
    No-Key Inversion   | Defender Decode | Clean Images | Watermarked Samples

All final assets are written into a single curated folder (default
``PAPER_CIFAR32_RESULTS/``) instead of being scattered across experiment dirs.

Design rules honoured here
--------------------------
* Watermark detectability is shown **only** for watermarked methods. Baseline /
  Keyed are reported as ``not_applicable`` (not as missing / failed data).
* A metric that is genuinely absent because a run failed is ``missing`` and is
  flagged in the readiness section.
* Training logs are read robustly: every ``train_log*.jsonl`` segment in a run
  directory is merged, deduplicated by ``step`` (last record wins), and sorted,
  so resumed / segmented training does not create misleading duplicate-step
  charts.
* Summary tables clearly separate generation quality, watermark traceability,
  clean false positive, inversion resistance, and robustness.

Usage
-----
    python -m scripts.make_cifar_paper_figures \\
        --bundle-dir runs/traceflow-cifar32_lat16_vae/traceflow-cifar32_lat16_vae-paper-all \\
        --results-dir <bundle>/results \\
        --output-dir PAPER_CIFAR32_RESULTS

Or via the CLI:
    python -m scripts.traceflow paper-figures --config configs/traceflow_cifar32.yml
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils.plotting import (
    METHOD_COLORS,
    METRIC_COLORS,
    grouped_bar,
    load_image,
    placeholder_axis,
    save_figure,
    setup_style,
)

NOT_APPLICABLE = "not_applicable"
MISSING = "missing"


# ---------------------------------------------------------------------------
# Method registry — the single source of truth for labels and watermark status.
# (exp ids are an internal implementation detail and never shown in figures.)
# ---------------------------------------------------------------------------

class Method:
    def __init__(self, label: str, color_key: str, exp_id: str, has_watermark: bool):
        self.label = label
        self.color_key = color_key
        self.exp_id = exp_id
        self.has_watermark = has_watermark


METHODS: List[Method] = [
    Method("Baseline Generator", "baseline", "exp01", False),
    Method("Keyed Latent", "keyed", "exp02", False),
    Method("TraceFlow Identity", "traceflow_identity", "exp03", True),
    Method("Full TraceFlow", "traceflow", "exp04", True),
]

# The robustness / inversion attack run that backs the Full TraceFlow method.
INVERSION_EXP = "exp04"
ROBUSTNESS_EXP = "exp05"

ROBUSTNESS_TRANSFORMS = ["clean", "jpeg", "resize", "blur", "gaussian_noise", "crop_resize"]
CURRENT_BUNDLE_DIR: Optional[Path] = None


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _resolve_existing_path(path_like: Any) -> Optional[Path]:
    """Resolve server-exported absolute paths back into the local bundle."""
    if not path_like:
        return None
    raw = Path(str(path_like))
    if raw.exists():
        return raw
    candidates: List[Path] = []
    if CURRENT_BUNDLE_DIR is not None:
        parts = raw.parts
        if "traceflow_runs" in parts:
            idx = parts.index("traceflow_runs")
            # /root/autodl-fs/traceflow_runs/<bundle>/outputs/run -> <local-bundle>/outputs/run
            if len(parts) > idx + 2:
                candidates.append(CURRENT_BUNDLE_DIR.joinpath(*parts[idx + 2:]))
        if not raw.is_absolute():
            candidates.append(CURRENT_BUNDLE_DIR / raw)
    if not raw.is_absolute():
        candidates.append(raw)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _run_dir(metrics: Dict[str, Any]) -> Optional[Path]:
    out_dir = _resolve_existing_path(metrics.get("output_dir"))
    if out_dir is not None and (out_dir / "train_report.json").exists():
        return out_dir
    run_name = str(metrics.get("run_name") or "")
    candidate_names: List[str] = []
    if run_name:
        candidate_names.append(run_name)
        replacements = {
            "-exp01-baseline": "-generator",
            "-exp02-keyed": "-keyed",
            "-exp03-identity": "-identity",
            "-exp04-traceflow": "-final",
            "-exp05-robustness": "-final",
        }
        for src, dst in replacements.items():
            if src in run_name:
                candidate_names.append(run_name.replace(src, dst))
    checkpoint = str(metrics.get("checkpoint") or "")
    if checkpoint:
        parts = Path(checkpoint).parts
        if "checkpoints" in parts:
            idx = parts.index("checkpoints")
            if len(parts) > idx + 1:
                ckpt_name = parts[idx + 1]
                if ckpt_name not in {"generator", "keyed", "identity", "traceflow"}:
                    candidate_names.append(ckpt_name)
    for name in candidate_names:
        candidates: List[Path] = []
        if CURRENT_BUNDLE_DIR is not None:
            candidates.append(CURRENT_BUNDLE_DIR / "outputs" / name)
        candidates.extend([
            Path("outputs/flow_transformer_cifar32_lat16_vae") / name,
            Path("outputs/flow_transformer_cifar32_lat16") / name,
            Path("outputs/flow_transformer_cifar32") / name,
            Path("outputs/flow_transformer") / name,
            Path("outputs") / name,
        ])
        for cand in candidates:
            if cand.exists() and ((cand / "train_report.json").exists() or (cand / "train_log.jsonl").exists()):
                return cand
    if out_dir is not None:
        return out_dir
    return None


def _train_report_for_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = _run_dir(metrics)
    if run_dir is None:
        return {}
    for name in ("train_report.json", "smoke_report.json"):
        path = run_dir / name
        if path.exists():
            return _load_json(path)
    return {}


def _last_log_value(metrics: Dict[str, Any], key: str) -> Any:
    rows = read_merged_log(metrics)
    for row in reversed(rows):
        if row.get(key) is not None:
            return row[key]
    return None


def _flatten_inversion_metrics(metrics: Dict[str, Any], inversion: Dict[str, Any]) -> None:
    """Copy nested eval_traceflow_inversion metrics into summary-friendly keys."""
    if not inversion:
        return
    runs = inversion.get("attacker_runs")
    if isinstance(runs, dict):
        run = runs.get("no_key") or next(iter(runs.values()), {})
        latent = run.get("latent_attack") if isinstance(run, dict) else None
        if isinstance(latent, dict):
            mapping = {
                "final_gml": "latent_final_gml",
                "no_key_psnr": "latent_no_key_psnr",
                "raw_no_key_image_bit_acc": "latent_raw_no_key_image_bit_acc",
                "raw_no_key_latent_bit_acc": "latent_raw_no_key_latent_bit_acc",
            }
            for src, dst in mapping.items():
                if metrics.get(dst) is None and latent.get(src) is not None:
                    metrics[dst] = latent[src]
            if metrics.get("robustness") is None and latent.get("robustness") is not None:
                metrics["robustness"] = latent["robustness"]
        for key, block in (run or {}).items():
            if isinstance(block, dict) and key.endswith("pixel_attack"):
                for src, dst in (("final_gml", "pixel_final_gml"),
                                 ("psnr", "pixel_psnr"),
                                 ("raw_pixel_image_bit_acc", "pixel_raw_image_bit_acc"),
                                 ("raw_pixel_latent_bit_acc", "pixel_raw_latent_bit_acc")):
                    if metrics.get(dst) is None and block.get(src) is not None:
                        metrics[dst] = block[src]
                if block.get("attack_method") == "geiping_pixel":
                    for src, dst in (("final_gml", "strong_pixel_final_gml"),
                                     ("psnr", "strong_pixel_psnr"),
                                     ("raw_pixel_image_bit_acc", "strong_pixel_raw_image_bit_acc"),
                                     ("raw_pixel_latent_bit_acc", "strong_pixel_raw_latent_bit_acc")):
                        if metrics.get(dst) is None and block.get(src) is not None:
                            metrics[dst] = block[src]
    elif inversion.get("final_gml") is not None and metrics.get("latent_final_gml") is None:
        metrics["latent_final_gml"] = inversion["final_gml"]


def _enrich_metrics(metrics: Dict[str, Any], inversion: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(metrics)
    run_dir = _run_dir(enriched)
    if run_dir is not None:
        enriched["output_dir"] = str(run_dir)
    report = _train_report_for_metrics(enriched)
    if report:
        for key in ("steps_completed", "total_train_wall_time_s",
                    "avg_step_time_s", "cuda_max_memory_allocated_MB"):
            if enriched.get(key) is None and report.get(key) is not None:
                enriched[key] = report[key]
        for src, dst in (("loss", "final_loss"), ("loss_flow", "final_flow_loss")):
            val = _last_log_value(enriched, src)
            if enriched.get(dst) is None and val is not None:
                enriched[dst] = val
        wm = report.get("watermark", {}) if isinstance(report.get("watermark"), dict) else {}
        for key in ("generated_image_bit_acc", "generated_latent_bit_acc",
                    "generated_image_ber", "generated_latent_ber",
                    "clean_false_positive_img", "clean_false_positive_latent",
                    "image_delta_mse"):
            if enriched.get(key) is None and wm.get(key) is not None:
                enriched[key] = wm[key]
    _flatten_inversion_metrics(enriched, inversion)
    return enriched


def discover_experiments(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return ``{exp_id: {"mode", "dir", "metrics", "inversion"}}`` (prefer full)."""
    found: Dict[str, Dict[str, Any]] = {}
    if not results_dir.exists():
        return found
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp"):
            continue
        chosen: Optional[Path] = None
        chosen_mode: Optional[str] = None
        for m in ("full", "smoke"):
            if (exp_dir / m / "metrics.json").exists():
                chosen, chosen_mode = exp_dir / m, m
                break
        if chosen is None:
            continue
        inversion: Dict[str, Any] = {}
        for inv_name in ("strong_inversion_geiping", "inversion", "inversion_latent"):
            inv_path = chosen / inv_name / "metrics.json"
            if inv_path.exists():
                inversion = _load_json(inv_path)
                break
        raw_metrics = _load_json(chosen / "metrics.json")
        found[exp_dir.name] = {
            "mode": chosen_mode,
            "dir": chosen,
            "metrics": _enrich_metrics(raw_metrics, inversion),
            "inversion": inversion,
        }
    return found


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
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


def read_merged_log(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge all train_log*.jsonl segments, dedupe by step (last wins), sort.

    This handles resumed / segmented training cleanly and avoids misleading
    charts caused by duplicate or out-of-order steps.
    """
    run_dir = _run_dir(metrics)
    if run_dir is None:
        return []
    by_step: Dict[int, Dict[str, Any]] = {}
    no_step: List[Dict[str, Any]] = []
    for seg in sorted(run_dir.glob("train_log*.jsonl")):
        for row in _read_jsonl(seg):
            step = row.get("step")
            if step is None:
                no_step.append(row)
            else:
                by_step[int(step)] = row  # last record per step wins
    merged = [by_step[s] for s in sorted(by_step)]
    return merged or no_step


# ---------------------------------------------------------------------------
# Metric resolution with not_applicable vs missing semantics
# ---------------------------------------------------------------------------

def _method_info(experiments: Dict[str, Dict[str, Any]], method: Method) -> Optional[Dict[str, Any]]:
    return experiments.get(method.exp_id)


def metric_value(
    experiments: Dict[str, Dict[str, Any]],
    method: Method,
    key: str,
    *,
    requires_watermark: bool = False,
) -> Any:
    """Resolve a metric for a method, returning a numeric value or a status string.

    - ``not_applicable`` when the metric only makes sense for watermarked methods
      and this method has no watermark.
    - ``missing`` when the run/metric is absent for an applicable method.
    """
    if requires_watermark and not method.has_watermark:
        return NOT_APPLICABLE
    info = _method_info(experiments, method)
    if info is None:
        return MISSING
    value = info["metrics"].get(key)
    return value if value is not None else MISSING


def _numeric(value: Any) -> Optional[float]:
    return value if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# Figure 1 — Method overview / pipeline
# ---------------------------------------------------------------------------

def fig_method_overview(out_stem: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    main_chain = [
        (2, 16, 13, 8, "Encoder", "#E8EEF7"),
        (18, 16, 13, 8, "Keyed Latent\n$z_k$", "#D6E4F5"),
        (34, 16, 14, 8, "Flow\nTransformer", "#C5D9F1"),
        (51, 16, 13, 8, "Inverse Key\n$\\hat z$", "#D6E4F5"),
        (67, 16, 15, 8, "Watermarked\nDecoder", "#F7E6D6"),
    ]
    boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for x, y, w, h, label, fc in main_chain:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.4, edgecolor="#33476b", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        boxes[label.split("\n")[0]] = (x, y, w, h)

    for x, y, w, h, label, fc in [
        (67, 2, 15, 8, "Re-Encoder\n$z_{re,k}$", "#F7E6D6"),
        (86, 24, 12, 8, "Image\nDetector", "#E6F2E6"),
        (86, 4, 12, 8, "Latent\nDetector", "#E6F2E6"),
    ]:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.4, edgecolor="#2f6b3f", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        boxes[label.split("\n")[0]] = (x, y, w, h)

    def arrow(p0, p1, color="#33476b"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.6, color=color, shrinkA=0, shrinkB=0))

    chain = ["Encoder", "Keyed Latent", "Flow", "Inverse Key", "Watermarked"]
    for a, b in zip(chain[:-1], chain[1:]):
        xa, ya, wa, ha = boxes[a]
        xb, yb, wb, hb = boxes[b]
        arrow((xa + wa, ya + ha / 2), (xb, yb + hb / 2))
    xw, yw, ww, hw = boxes["Watermarked"]
    xi, yi, wi, hi = boxes["Image"]
    arrow((xw + ww, yw + hw * 0.7), (xi, yi + hi / 2), color="#2f6b3f")
    xr, yr, wr, hr = boxes["Re-Encoder"]
    arrow((xw + ww / 2, yw), (xr + wr / 2, yr + hr), color="#2f6b3f")
    xl, yl, wl, hl = boxes["Latent"]
    arrow((xr + wr, yr + hr / 2), (xl, yl + hl / 2), color="#2f6b3f")

    ax.set_title("TraceFlow (CIFAR-10 32x32): keyed rectified flow with a dual-head watermark",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=[
        mpatches.Patch(facecolor="#C5D9F1", edgecolor="#33476b", label="Generative path"),
        mpatches.Patch(facecolor="#F7E6D6", edgecolor="#33476b", label="Watermark embedding"),
        mpatches.Patch(facecolor="#E6F2E6", edgecolor="#2f6b3f", label="Forensic detectors"),
    ], loc="upper left", ncol=3, bbox_to_anchor=(0.0, 0.04), fontsize=9)
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 2 — Generation quality comparison by method (sample grids)
# ---------------------------------------------------------------------------

def _verified_post_training_sample_grid(run_dir: Path) -> Optional[Path]:
    """Return a post-training sample grid only when metadata proves it is safe.

    Old CIFAR runs used ``sample_flow_transformer`` without loading the local AE
    checkpoint, producing grids decoded by a random autoencoder.  Only trust the
    post-training sample directory after the sampler records both EMA use and a
    loaded AE checkpoint.
    """
    sample_dir = run_dir / "exp_samples"
    grid = sample_dir / "sample_grid.png"
    meta_path = sample_dir / "sampling_config.json"
    if not grid.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    if meta.get("use_ema") is True and meta.get("autoencoder_checkpoint_loaded") is True:
        return grid
    return None


def _sample_grid_choice(metrics: Dict[str, Any]) -> Tuple[Optional[Path], str]:
    run_dir = _run_dir(metrics)
    if run_dir is None:
        return None, "missing_run_dir"
    verified = _verified_post_training_sample_grid(run_dir)
    if verified is not None:
        return verified, "verified_post_training_ema"
    # Prefer plain (un-watermarked) generated samples for a fair quality view,
    # but mark this as a fallback because these training-time grids may predate
    # the EMA/AE-checkpoint sampling fix.
    plain = sorted(p for p in run_dir.glob("samples_step*.png")
                   if "watermarked" not in p.name)
    if plain:
        return plain[-1], "fallback_to_raw_training_sample"
    any_sample = sorted(run_dir.glob("sample_*.png"))
    if any_sample:
        return any_sample[-1], "fallback_to_unverified_sample"
    return None, "missing_sample_grid"


def _latest_sample_grid(metrics: Dict[str, Any]) -> Optional[Path]:
    grid, _ = _sample_grid_choice(metrics)
    return grid


def fig_generation_quality(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, len(METHODS), figsize=(3.0 * len(METHODS), 3.6))
    if len(METHODS) == 1:
        axes = [axes]
    for ax, method in zip(axes, METHODS):
        ax.set_title(method.label, fontsize=11,
                     color=METHOD_COLORS.get(method.color_key, "#333333"))
        ax.set_xticks([])
        ax.set_yticks([])
        info = _method_info(experiments, method)
        img = None
        subtitle = MISSING
        if info is not None:
            grid = _latest_sample_grid(info["metrics"])
            if grid is not None:
                img = load_image(grid)
            flow = _numeric(_final_log_value(info["metrics"], "loss_flow"))
            if flow is not None:
                subtitle = f"flow loss {flow:.3f}"
        if img is not None:
            ax.imshow(img)
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
            ax.set_xlabel(subtitle, fontsize=9)
        else:
            placeholder_axis(ax, f"no samples\n({subtitle})")
    fig.suptitle("Generation quality by method (generated CIFAR-10 32x32 samples)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.2)
    save_figure(fig, out_stem)


def _final_log_value(metrics: Dict[str, Any], key: str) -> Any:
    rows = read_merged_log(metrics)
    for row in reversed(rows):
        if key in row and row[key] is not None:
            return row[key]
    return None


# ---------------------------------------------------------------------------
# Figure 3 — Autoencoder reconstruction diagnostic
# ---------------------------------------------------------------------------

def _ae_grid_and_metrics(bundle_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    grid_path = None
    metrics_path = None
    for cand in [
        bundle_dir / "reports" / "ae_recon_grid.png",
        bundle_dir / "reports" / "ae_diagnosis" / "ae_recon_grid.png",
    ]:
        if cand.exists():
            grid_path = cand
            break
    for cand in [
        bundle_dir / "reports" / "ae_metrics.json",
        bundle_dir / "reports" / "ae_diagnosis" / "ae_metrics.json",
    ]:
        if cand.exists():
            metrics_path = cand
            break
    return grid_path, metrics_path


def fig_autoencoder_diagnostics(bundle_dir: Path, out_stem: Path) -> None:
    setup_style()
    grid_path, metrics_path = _ae_grid_and_metrics(bundle_dir)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_xticks([])
    ax.set_yticks([])
    img = load_image(grid_path) if grid_path else None
    if img is not None:
        ax.imshow(img)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        caption = "top: originals   bottom: reconstructions"
        if metrics_path:
            rec = _load_json(metrics_path).get("reconstruction", {})
            psnr = rec.get("psnr")
            ssim = rec.get("ssim")
            l1 = rec.get("l1")
            parts = []
            if psnr is not None:
                parts.append(f"PSNR {psnr:.2f} dB")
            if ssim is not None:
                parts.append(f"SSIM {ssim:.3f}")
            if l1 is not None:
                parts.append(f"L1 {l1:.4f}")
            if parts:
                caption += "   |   " + "   ".join(parts)
        ax.set_xlabel(caption, fontsize=10)
    else:
        placeholder_axis(ax, "AE reconstruction grid not found.\n"
                             "Run `traceflow train-autoencoder` first.")
    ax.set_title("Local autoencoder reconstruction diagnostic",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 4 — Watermark traceability (watermarked methods only)
# ---------------------------------------------------------------------------

def fig_watermark_traceability(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    wm_methods = [m for m in METHODS if m.has_watermark]

    labels: List[str] = []
    img_vals: List[Optional[float]] = []
    lat_vals: List[Optional[float]] = []
    for method in wm_methods:
        labels.append(method.label)
        img_vals.append(_numeric(metric_value(experiments, method, "generated_image_bit_acc",
                                               requires_watermark=True)))
        lat_vals.append(_numeric(metric_value(experiments, method, "generated_latent_bit_acc",
                                               requires_watermark=True)))

    # Strong-traceability target band (0.80-0.90): clearly above chance is the
    # claim; perfect 1.0 is not required for the feasibility story.
    ax.axhspan(0.80, 0.90, color="#55A868", alpha=0.12, zorder=0,
               label="Target band (0.80-0.90)")
    grouped_bar(
        ax, labels,
        {"Image detector": img_vals, "Latent detector": lat_vals},
        colors={"Image detector": METRIC_COLORS["image_detector"],
                "Latent detector": METRIC_COLORS["latent_detector"]},
        value_fmt="{:.2f}",
    )
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.4, label="Random chance (0.50)")
    ax.set_ylabel("Generated-sample bit accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Watermark traceability (watermarked methods only)")
    ax.legend(loc="upper left", ncol=2)
    ax.text(0.5, 0.02,
            "Baseline Generator and Keyed Latent carry no watermark -> not applicable.",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=9,
            color="#666666", style="italic")
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 5 — Clean false-positive
# ---------------------------------------------------------------------------

def fig_clean_false_positive(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    wm_methods = [m for m in METHODS if m.has_watermark]
    labels = [m.label for m in wm_methods]
    img_fp: List[Optional[float]] = []
    lat_fp: List[Optional[float]] = []
    for method in wm_methods:
        img_fp.append(_numeric(metric_value(experiments, method, "clean_false_positive_img",
                                            requires_watermark=True)))
        lat_fp.append(_numeric(metric_value(experiments, method, "clean_false_positive_latent",
                                            requires_watermark=True)))
    grouped_bar(
        ax, labels,
        {"Image detector": img_fp, "Latent detector": lat_fp},
        colors={"Image detector": METRIC_COLORS["image_detector"],
                "Latent detector": METRIC_COLORS["latent_detector"]},
        value_fmt="{:.2f}",
    )
    ax.axhline(0.5, color="#C44E52", linestyle="--", linewidth=1.4,
               label="Target (~0.5, no false detection)")
    ax.set_ylabel("Clean-image bit accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Clean false-positive (should stay near 0.5)")
    ax.legend(loc="upper left")
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 6 — Inversion attack grid
# ---------------------------------------------------------------------------

def fig_inversion_attack(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    img_dir: Optional[Path] = None
    for exp_id in (INVERSION_EXP, ROBUSTNESS_EXP):
        info = experiments.get(exp_id)
        if not info:
            continue
        for inv_name in ("strong_inversion_geiping", "inversion", "inversion_latent"):
            cand = info["dir"] / inv_name / "images"
            if cand.exists():
                img_dir = cand
                break
        if img_dir:
            break

    panels = [
        ("original_grid.png", "Clean Images"),
        ("latent_*_raw_nokey_grid.png", "No-Key Inversion"),
        ("latent_*_raw_defender_grid.png", "Defender Decode"),
        ("latent_*_post_watermark_defender_grid.png", "Watermarked Forensic"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    for ax, (pattern, title) in zip(axes, panels):
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        img = None
        if img_dir is not None:
            matches = sorted(img_dir.glob(pattern))
            if matches:
                img = load_image(matches[0])
        if img is not None:
            ax.imshow(img)
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
        else:
            placeholder_axis(ax, "not available")
    fig.suptitle("No-key inversion vs. defender decode (Full TraceFlow)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.5)
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 7 — Robustness
# ---------------------------------------------------------------------------

def _collect_robustness(experiments: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, List[Optional[float]]], bool]:
    series: Dict[str, List[Optional[float]]] = {
        "Image detector": [None] * len(ROBUSTNESS_TRANSFORMS),
        "Latent detector": [None] * len(ROBUSTNESS_TRANSFORMS),
    }
    has_real = False
    info = experiments.get(ROBUSTNESS_EXP) or experiments.get(INVERSION_EXP)
    if info is None:
        return series, has_real
    rob = info["metrics"].get("robustness", {})
    block = rob.get("latent_attack", rob) if isinstance(rob, dict) else {}
    for i, t in enumerate(ROBUSTNESS_TRANSFORMS):
        entry = block.get(t) if isinstance(block, dict) else None
        if isinstance(entry, dict):
            ib = entry.get("image_bit_acc")
            lb = entry.get("latent_bit_acc")
            if ib is not None:
                series["Image detector"][i] = ib
                has_real = True
            if lb is not None:
                series["Latent detector"][i] = lb
                has_real = True
    # Back-fill clean reference from headline raw metrics.
    clean_idx = ROBUSTNESS_TRANSFORMS.index("clean")
    m = info["metrics"]
    if series["Image detector"][clean_idx] is None and m.get("latent_raw_no_key_image_bit_acc") is not None:
        series["Image detector"][clean_idx] = m["latent_raw_no_key_image_bit_acc"]
    if series["Latent detector"][clean_idx] is None and m.get("latent_raw_no_key_latent_bit_acc") is not None:
        series["Latent detector"][clean_idx] = m["latent_raw_no_key_latent_bit_acc"]
    return series, has_real


def fig_robustness(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    series, has_real = _collect_robustness(experiments)
    grouped_bar(
        ax, [t.upper() for t in ROBUSTNESS_TRANSFORMS], series,
        colors={"Image detector": METRIC_COLORS["image_detector"],
                "Latent detector": METRIC_COLORS["latent_detector"]},
        value_fmt="{:.2f}",
    )
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.4, label="Random chance (0.5)")
    ax.axhline(0.8, color="#55A868", linestyle="--", linewidth=1.2, label="Acceptable (0.80)")
    ax.set_ylabel("Recovered bit accuracy (Full TraceFlow)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Transform robustness on raw inversion outputs")
    ax.legend(loc="upper right", ncol=2)
    if not has_real:
        ax.text(0.5, 0.92,
                "Robustness was disabled for this paper run (robustness_enabled: false).\n"
                "Only the clean reference is populated; re-run exp05 with robustness on to fill this.",
                ha="center", va="top", transform=ax.transAxes, fontsize=9,
                color="#a05050", style="italic",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf3f3", ec="#e3c2c2"))
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 8 — Training dashboard (Full TraceFlow, merged/deduped log)
# ---------------------------------------------------------------------------

def _series(rows: List[Dict[str, Any]], key: str) -> Tuple[List[Any], List[Any]]:
    xs, ys = [], []
    for row in rows:
        s = row.get("step")
        y = row.get(key)
        if s is not None and y is not None:
            xs.append(s)
            ys.append(y)
    return xs, ys


def fig_training_dashboard(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    info = experiments.get("exp04") or experiments.get("exp03")
    rows = read_merged_log(info["metrics"]) if info else []

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    panels = [
        (axes[0][0], "Total & flow loss", [("loss", "total"), ("loss_flow", "flow")], None),
        (axes[0][1], "Watermark losses", [("loss_wm_img", "img wm"), ("loss_wm_latent", "latent wm")], None),
        (axes[0][2], "Preservation losses",
         [("loss_img", "image"), ("loss_residual", "residual"),
          ("loss_perceptual", "perceptual"), ("loss_frequency", "frequency")], None),
        (axes[1][0], "Image detector accuracy", [("bit_acc_img", "image bit acc")], 0.5),
        (axes[1][1], "Latent detector accuracy", [("bit_acc_latent", "latent bit acc")], 0.5),
        (axes[1][2], "Clean false positive",
         [("clean_false_positive_img", "clean FP img"),
          ("clean_false_positive_latent", "clean FP latent")], 0.5),
    ]
    any_data = False
    for ax, title, keys, ref in panels:
        plotted = False
        for key, label in keys:
            xs, ys = _series(rows, key)
            if xs:
                ax.plot(xs, ys, label=label)
                plotted = True
                any_data = True
        if ref is not None:
            ax.axhline(ref, color="#888888", linestyle=":", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        if plotted:
            ax.legend(loc="best", fontsize=9)
        else:
            placeholder_axis(ax, "no data")
    if not any_data:
        for row in axes:
            for ax in row:
                placeholder_axis(ax, "no training log found")
    fig.suptitle("Full TraceFlow training dashboard (CIFAR-10 32x32)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Summary tables (CSV + Markdown), separated by category
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return MISSING
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


# Each category maps to a list of (metric_key, column_label, requires_watermark,
# applies_only_to_inversion).
SECTIONS: List[Tuple[str, List[Tuple[str, str, bool, bool]]]] = [
    ("Generation quality", [
        ("steps_completed", "Steps", False, False),
        ("final_loss", "Final loss", False, False),
    ]),
    ("Watermark traceability", [
        ("generated_image_bit_acc", "Gen img acc", True, False),
        ("generated_latent_bit_acc", "Gen lat acc", True, False),
    ]),
    ("Clean false positive", [
        ("clean_false_positive_img", "Clean FP img", True, False),
        ("clean_false_positive_latent", "Clean FP latent", True, False),
    ]),
    ("Inversion resistance", [
        ("latent_final_gml", "Latent GML", False, True),
        ("latent_no_key_psnr", "No-key PSNR", False, True),
        ("latent_raw_no_key_image_bit_acc", "No-key img acc", True, True),
        ("latent_raw_no_key_latent_bit_acc", "No-key lat acc", True, True),
        ("pixel_final_gml", "Pixel/strong GML", False, True),
        ("pixel_psnr", "Pixel/strong PSNR", False, True),
        ("pixel_raw_image_bit_acc", "Pixel/strong img acc", True, True),
    ]),
]


def _section_value(experiments, method, key, requires_wm, inversion_only) -> Any:
    if inversion_only and method.exp_id not in (INVERSION_EXP,):
        # Inversion metrics are only produced for the Full TraceFlow attack run.
        return NOT_APPLICABLE
    return metric_value(experiments, method, key, requires_watermark=requires_wm)


def _runtime_for_method(
    experiments: Dict[str, Dict[str, Any]], method: Method
) -> Tuple[Any, Any, Any]:
    """Return (steps_completed, wall_time_s, avg_step_s) for a method.

    Wall time comes from the final merged training-log ``elapsed_s`` so resumed
    runs still report a sensible figure. Returns ``missing`` when unavailable.
    """
    info = _method_info(experiments, method)
    if info is None:
        return MISSING, MISSING, MISSING
    steps = info["metrics"].get("steps_completed")
    rows = read_merged_log(info["metrics"])
    elapsed: Any = MISSING
    avg: Any = MISSING
    for row in reversed(rows):
        if row.get("elapsed_s") is not None:
            elapsed = row["elapsed_s"]
            last_step = row.get("step")
            if isinstance(elapsed, (int, float)) and isinstance(last_step, (int, float)) and last_step:
                avg = elapsed / last_step
            break
    return (steps if steps is not None else MISSING, elapsed, avg)


def _ae_reconstruction_metrics(bundle_dir: Optional[Path]) -> Dict[str, Any]:
    if bundle_dir is None:
        return {}
    _, metrics_path = _ae_grid_and_metrics(bundle_dir)
    if metrics_path is None:
        return {}
    return _load_json(metrics_path).get("reconstruction", {})


def write_method_metrics_csv(experiments: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Long-format CSV of every metric per method (status-aware).

    One row per (method, metric). ``not_applicable`` / ``missing`` are recorded
    in a dedicated ``status`` column so downstream tooling never confuses an
    absent value with a genuine zero.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for method in METHODS:
        info = _method_info(experiments, method)
        metrics = info["metrics"] if info else {}
        keys = sorted(k for k, v in metrics.items() if isinstance(v, (int, float, bool)))
        if not keys and info is None:
            rows.append({
                "method": method.label,
                "has_watermark": "yes" if method.has_watermark else "no",
                "metric": "(run)",
                "value": "",
                "status": MISSING,
            })
            continue
        for key in keys:
            value = metrics.get(key)
            rows.append({
                "method": method.label,
                "has_watermark": "yes" if method.has_watermark else "no",
                "metric": key,
                "value": _fmt(value),
                "status": "ok",
            })
    with open(out_dir / "method_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "has_watermark", "metric", "value", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    experiments: Dict[str, Dict[str, Any]],
    out_dir: Path,
    bundle_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write a combined summary.csv and a sectioned summary.md.

    Returns a readiness dict describing missing metrics for the readiness report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined CSV: one row per method, all columns.
    all_cols: List[Tuple[str, str, bool, bool]] = []
    for _, cols in SECTIONS:
        all_cols.extend(cols)

    csv_fields = ["method", "has_watermark"] + [label for _, label, _, _ in all_cols]
    missing_report: Dict[str, List[str]] = {}
    sample_grid_status: Dict[str, str] = {}
    rows_out: List[Dict[str, str]] = []
    for method in METHODS:
        info = _method_info(experiments, method)
        row = {
            "method": method.label,
            "has_watermark": "yes" if method.has_watermark else "no",
        }
        method_missing: List[str] = []
        for key, label, req_wm, inv_only in all_cols:
            val = _section_value(experiments, method, key, req_wm, inv_only)
            row[label] = _fmt(val)
            if val == MISSING:
                method_missing.append(label)
        if info is None:
            method_missing.append("run/metrics.json absent")
        else:
            _grid, sample_status = _sample_grid_choice(info["metrics"])
            if sample_status != "verified_post_training_ema":
                sample_grid_status[method.label] = sample_status
        if method_missing:
            missing_report[method.label] = method_missing
        rows_out.append(row)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    # Sectioned Markdown.
    lines: List[str] = ["# TraceFlow CIFAR-10 32x32 — Paper Summary", ""]
    lines.append("Metrics are method-named. `not_applicable` marks methods for which a "
                 "metric is meaningless (e.g. watermark accuracy for a no-watermark "
                 "baseline). `missing` marks an applicable metric that is absent because "
                 "a run did not produce it.")
    lines.append("")
    for section_name, cols in SECTIONS:
        lines.append(f"## {section_name}")
        lines.append("")
        headers = ["Method"] + [label for _, label, _, _ in cols]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for method in METHODS:
            cells = [method.label]
            for key, _label, req_wm, inv_only in cols:
                cells.append(_fmt(_section_value(experiments, method, key, req_wm, inv_only)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Robustness section (Full TraceFlow only).
    lines.append("## Robustness (Full TraceFlow, raw inversion outputs)")
    lines.append("")
    series, has_real = _collect_robustness(experiments)
    headers = ["Transform", "Image detector", "Latent detector"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for i, t in enumerate(ROBUSTNESS_TRANSFORMS):
        ib = series["Image detector"][i]
        lb = series["Latent detector"][i]
        lines.append(f"| {t} | {_fmt(ib)} | {_fmt(lb)} |")
    if not has_real:
        lines.append("")
        lines.append("_Robustness was disabled for this paper run; only the clean "
                     "reference is populated._")
    lines.append("")

    # AE reconstruction quality (single shared local autoencoder).
    lines.append("## Autoencoder reconstruction quality")
    lines.append("")
    rec = _ae_reconstruction_metrics(bundle_dir)
    if rec:
        headers = ["Metric", "Value"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for label, key in (("PSNR (dB)", "psnr"), ("SSIM", "ssim"),
                           ("L1", "l1"), ("MSE", "mse")):
            lines.append(f"| {label} | {_fmt(rec.get(key))} |")
    else:
        lines.append("_AE reconstruction metrics not found in the bundle "
                     "(run `traceflow train-autoencoder`)._")
    lines.append("")

    # Training cost / runtime.
    lines.append("## Training cost / runtime")
    lines.append("")
    headers = ["Method", "Steps", "Wall time (s)", "Avg step (s)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for method in METHODS:
        steps, elapsed, avg = _runtime_for_method(experiments, method)
        lines.append(f"| {method.label} | {_fmt(steps)} | {_fmt(elapsed)} | {_fmt(avg)} |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    return {"missing": missing_report, "sample_grid_status": sample_grid_status}


def write_readiness(out_dir: Path, experiments: Dict[str, Dict[str, Any]], readiness: Dict[str, Any]) -> None:
    lines = ["# CIFAR-32 Paper Readiness", ""]
    present = [m.label for m in METHODS if _method_info(experiments, m) is not None]
    absent = [m.label for m in METHODS if _method_info(experiments, m) is None]
    lines.append(f"Methods with metrics: {', '.join(present) if present else '(none)'}")
    if absent:
        lines.append("")
        lines.append("## Missing runs")
        for label in absent:
            lines.append(f"- {label}: run/metrics.json absent (status `missing`).")
    miss = readiness.get("missing", {})
    detail = {k: v for k, v in miss.items() if v}
    sample_status = readiness.get("sample_grid_status", {})
    if detail:
        lines.append("")
        lines.append("## Missing metrics (applicable but absent)")
        for label, cols in detail.items():
            lines.append(f"- {label}: {', '.join(cols)}")
    if sample_status:
        lines.append("")
        lines.append("## Sample grid source warnings")
        for label, status in sample_status.items():
            lines.append(f"- {label}: {status}")
    if not absent and not detail and not sample_status:
        lines.append("")
        lines.append("All applicable metrics are present.")
    (out_dir / "readiness.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Curated-folder asset copying
# ---------------------------------------------------------------------------

def _copy_sample_grids(experiments: Dict[str, Dict[str, Any]], samples_dir: Path) -> List[str]:
    """Copy the latest generated sample grid for each method into ``samples/``."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for method in METHODS:
        info = _method_info(experiments, method)
        if info is None:
            continue
        grid = _latest_sample_grid(info["metrics"])
        if grid is None:
            continue
        slug = method.color_key
        dst = samples_dir / f"{slug}_samples.png"
        try:
            shutil.copy2(grid, dst)
            copied.append(dst.name)
        except OSError:
            continue
    return copied


def _copy_ae_diagnostics(bundle_dir: Optional[Path], diagnostics_dir: Path) -> List[str]:
    """Copy AE reconstruction grid + metrics into ``diagnostics/``."""
    if bundle_dir is None:
        return []
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    grid_path, metrics_path = _ae_grid_and_metrics(bundle_dir)
    copied: List[str] = []
    for src in (grid_path, metrics_path):
        if src is None:
            continue
        dst = diagnostics_dir / src.name
        try:
            shutil.copy2(src, dst)
            copied.append(dst.name)
        except OSError:
            continue
    return copied


def write_readme(
    output_dir: Path,
    experiments: Dict[str, Dict[str, Any]],
    results_dir: Path,
    bundle_dir: Optional[Path],
    copied_samples: List[str],
) -> None:
    present = [m.label for m in METHODS if _method_info(experiments, m) is not None]
    lines = [
        "# TraceFlow CIFAR-10 32x32 — Curated Paper Results",
        "",
        "Method-named, publication-ready assets for the TraceFlow feasibility study.",
        "Every figure and table uses method names only — no internal run IDs appear.",
        "",
        "## Story (in priority order)",
        "",
        "1. **Image quality** — generated samples are coherent (fig2, samples/).",
        "2. **Watermark traceability** — watermarked methods are detectable well "
        "above chance (fig4).",
        "3. **No-key inversion resistance** — an attacker without the key cannot "
        "recover clean content; the defender can (fig6).",
        "4. **Robustness** — supportive evidence under transforms (fig7).",
        "",
        "## Layout",
        "",
        "- `figures/` — fig1..fig8 as PNG (300 dpi) and vector PDF.",
        "- `tables/` — `summary.csv`, `summary.md`, `method_metrics.csv`.",
        "- `diagnostics/` — `readiness.md` plus copied autoencoder diagnostics.",
        "- `samples/` — latest generated sample grid per method.",
        "",
        "## Figures",
        "",
        "| File | Content |",
        "|---|---|",
        "| fig1_method_overview | Keyed rectified-flow pipeline with dual-head watermark |",
        "| fig2_generation_quality | Generated samples by method |",
        "| fig3_autoencoder_diagnostics | Original vs. AE reconstruction (PSNR/SSIM/L1) |",
        "| fig4_watermark_traceability | Image + latent detector accuracy (target 0.80-0.90) |",
        "| fig5_clean_false_positive | Clean false-positive calibration (~0.50) |",
        "| fig6_inversion_attack | No-key inversion vs. defender decode |",
        "| fig7_robustness | Transform robustness (clean/jpeg/resize/blur/noise/crop) |",
        "| fig8_training_dashboard | Full TraceFlow training curves |",
        "",
        "## Methods present",
        "",
        f"{', '.join(present) if present else '(none — re-run training/eval)'}",
        "",
        "## Provenance",
        "",
        f"- results-dir: `{results_dir}`",
        f"- bundle-dir: `{bundle_dir if bundle_dir else '(not provided)'}`",
        f"- sample grids copied: {', '.join(copied_samples) if copied_samples else '(none)'}",
        "",
        "See `diagnostics/readiness.md` for any missing or not-applicable metrics.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    global CURRENT_BUNDLE_DIR
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else None
    CURRENT_BUNDLE_DIR = bundle_dir
    results_dir = Path(args.results_dir) if args.results_dir else (
        bundle_dir / "results" if bundle_dir else Path("results/traceflow_cifar32_lat16_vae"))
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    diagnostics_dir = output_dir / "diagnostics"
    samples_dir = output_dir / "samples"
    for d in (output_dir, figures_dir, tables_dir, diagnostics_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(results_dir)
    print(f"[cifar-figures] results-dir: {results_dir}")
    print(f"[cifar-figures] output-dir:  {output_dir}")
    for method in METHODS:
        present = _method_info(experiments, method) is not None
        print(f"[cifar-figures]   {method.label:20s} -> {'found' if present else 'MISSING'}")

    fig_method_overview(figures_dir / "fig1_method_overview")
    print("[cifar-figures] wrote fig1_method_overview")
    fig_generation_quality(experiments, figures_dir / "fig2_generation_quality")
    print("[cifar-figures] wrote fig2_generation_quality")
    if bundle_dir is not None:
        fig_autoencoder_diagnostics(bundle_dir, figures_dir / "fig3_autoencoder_diagnostics")
        print("[cifar-figures] wrote fig3_autoencoder_diagnostics")
    fig_watermark_traceability(experiments, figures_dir / "fig4_watermark_traceability")
    print("[cifar-figures] wrote fig4_watermark_traceability")
    fig_clean_false_positive(experiments, figures_dir / "fig5_clean_false_positive")
    print("[cifar-figures] wrote fig5_clean_false_positive")
    fig_inversion_attack(experiments, figures_dir / "fig6_inversion_attack")
    print("[cifar-figures] wrote fig6_inversion_attack")
    fig_robustness(experiments, figures_dir / "fig7_robustness")
    print("[cifar-figures] wrote fig7_robustness")
    fig_training_dashboard(experiments, figures_dir / "fig8_training_dashboard")
    print("[cifar-figures] wrote fig8_training_dashboard")

    readiness = write_summary(experiments, tables_dir, bundle_dir)
    write_method_metrics_csv(experiments, tables_dir)
    write_readiness(diagnostics_dir, experiments, readiness)
    copied_samples = _copy_sample_grids(experiments, samples_dir)
    _copy_ae_diagnostics(bundle_dir, diagnostics_dir)
    write_readme(output_dir, experiments, results_dir, bundle_dir, copied_samples)
    print("[cifar-figures] wrote tables/summary.csv, tables/summary.md, "
          "tables/method_metrics.csv")
    print("[cifar-figures] wrote diagnostics/readiness.md, README.md")
    print(f"[cifar-figures] curated paper assets: {output_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curated method-named CIFAR-32 paper figures.")
    p.add_argument("--bundle-dir", default=None, help="Bundle root (for AE diagnostics + sample images).")
    p.add_argument("--results-dir", default=None, help="Directory with <exp>/<full|smoke>/metrics.json.")
    p.add_argument("--output-dir", default="PAPER_CIFAR32_RESULTS", help="Curated output folder.")
    p.add_argument("--config", default=None, help="Optional config (unused; accepted for CLI symmetry).")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))
