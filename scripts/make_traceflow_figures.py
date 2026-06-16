"""
scripts/make_traceflow_figures.py
=================================
Paper-level visualisation and result aggregation for TraceFlow.

Reads the per-experiment metrics written by the experiment framework
(``results/traceflow/<exp>/<smoke|full>/metrics.json`` and the nested
inversion / robustness JSON) and produces a set of publication-style figures
plus a summary CSV and Markdown table.

Figures (each saved as PNG **and** PDF)
---------------------------------------
1. ``fig1_pipeline``      — Method pipeline diagram.
2. ``fig2_ablation``      — Ablation bar chart (baseline / keyed / identity TraceFlow / full TraceFlow).
3. ``fig3_attack_grid``   — Attack reconstruction grid (original, no-key latent
                            decode, defender decode, pixel attack, post-watermark
                            sanity).
4. ``fig4_curves``        — Training / attack curves (generation loss, image bit
                            accuracy, latent bit accuracy, clean false-positive,
                            gradient-matching loss).
5. ``fig5_robustness``    — Robustness of the image and latent detectors under
                            JPEG / resize / blur / noise / crop.

Aggregation outputs
-------------------
* ``summary.csv``  — one row per experiment with headline metrics.
* ``summary.md``   — the same table rendered as Markdown.

Dependencies
------------
Only ``matplotlib`` is required.  ``pandas`` is used if available (purely for
convenience) but the script falls back to the standard-library ``csv``/``json``
modules otherwise.

Usage
-----
    python -m scripts.make_traceflow_figures \\
        --results-dir results/traceflow \\
        --output-dir  results/traceflow/figures
"""

from __future__ import annotations

import argparse
import csv
import json
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


# ---------------------------------------------------------------------------
# Experiment registry for figure labelling
# ---------------------------------------------------------------------------

# Ordered ablation lineup for Figure 2.  Each entry maps a display label and a
# stable colour key to the experiment id that supplies its metrics.
ABLATION_LINEUP: List[Tuple[str, str, str]] = [
    ("Baseline\n(identity)", "baseline", "exp01"),
    ("Keyed\nlatent", "keyed", "exp02"),
    ("TraceFlow\n(identity)", "traceflow_identity", "exp03"),
    ("Full\nTraceFlow", "traceflow", "exp04"),
]

# Experiment preferred for the training/attack curves (Figure 4).
CURVES_EXP_PRIORITY = ["exp04", "exp05", "exp03"]

# Experiment preferred for the attack reconstruction grid (Figure 3).
GRID_EXP_PRIORITY = ["exp04", "exp05"]

ROBUSTNESS_TRANSFORMS = [
    "clean",
    "jpeg",
    "resize",
    "blur",
    "gaussian_noise",
    "crop_resize",
]

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


def discover_experiments(results_dir: Path, mode: str) -> Dict[str, Dict[str, Any]]:
    """Discover experiment metrics under *results_dir*.

    Returns ``{exp_id: {"mode", "dir", "metrics", "inversion"}}``.

    ``mode`` is one of ``smoke``, ``full`` or ``auto`` (prefer full, else smoke).
    """
    found: Dict[str, Dict[str, Any]] = {}
    if not results_dir.exists():
        return found

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp"):
            continue
        exp_id = exp_dir.name

        candidate_modes = (
            [mode] if mode in ("smoke", "full") else ["full", "smoke"]
        )
        chosen: Optional[Path] = None
        chosen_mode: Optional[str] = None
        for m in candidate_modes:
            mdir = exp_dir / m
            if (mdir / "metrics.json").exists():
                chosen, chosen_mode = mdir, m
                break
        if chosen is None:
            continue

        metrics = _load_json(chosen / "metrics.json")
        inversion = {}
        for inv_name in ("strong_inversion_geiping", "inversion", "inversion_latent"):
            inv_path = chosen / inv_name / "metrics.json"
            if inv_path.exists():
                inversion = _load_json(inv_path)
                break

        found[exp_id] = {
            "mode": chosen_mode,
            "dir": chosen,
            "metrics": metrics,
            "inversion": inversion,
        }
    return found


def _train_log_path(metrics: Dict[str, Any]) -> Optional[Path]:
    """Resolve the train_log.jsonl path for an experiment from its metrics."""
    out_dir = _resolve_existing_path(metrics.get("output_dir"))
    if out_dir is not None:
        for p in sorted(out_dir.glob("train_log*.jsonl")):
            if p.exists():
                return p
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
                candidate_names.append(parts[idx + 1])
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
        for run_dir in candidates:
            for p in sorted(run_dir.glob("train_log*.jsonl")):
                if p.exists():
                    return p
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return rows


# ---------------------------------------------------------------------------
# Figure 1 — Method pipeline diagram
# ---------------------------------------------------------------------------

def fig_pipeline(out_stem: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    # (x, y, w, h, label, facecolor)
    main_chain = [
        (2,  16, 13, 8, "Encoder",            "#E8EEF7"),
        (18, 16, 13, 8, "Keyed Latent\n$z_k$", "#D6E4F5"),
        (34, 16, 14, 8, "Flow\nTransformer",   "#C5D9F1"),
        (51, 16, 13, 8, "Inverse Key\n$\\hat z$", "#D6E4F5"),
        (67, 16, 15, 8, "Watermarked\nDecoder", "#F7E6D6"),
    ]
    boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for x, y, w, h, label, fc in main_chain:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.4, edgecolor="#33476b", facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        boxes[label.split("\n")[0]] = (x, y, w, h)

    # Branch boxes (Re-Encoder + two detectors)
    branch = [
        (67, 2,  15, 8, "Re-Encoder\n$z_{re,k}$", "#F7E6D6"),
        (86, 24, 12, 8, "Image\nDetector",        "#E6F2E6"),
        (86, 4,  12, 8, "Latent\nDetector",       "#E6F2E6"),
    ]
    for x, y, w, h, label, fc in branch:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.4, edgecolor="#2f6b3f", facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        boxes[label.split("\n")[0]] = (x, y, w, h)

    def arrow(p0: Tuple[float, float], p1: Tuple[float, float],
              color: str = "#33476b") -> None:
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=16,
            linewidth=1.6, color=color, shrinkA=0, shrinkB=0,
        ))

    # Forward chain arrows (left → right along the main row)
    chain_order = ["Encoder", "Keyed Latent", "Flow", "Inverse Key", "Watermarked"]
    for a, b in zip(chain_order[:-1], chain_order[1:]):
        xa, ya, wa, ha = boxes[a]
        xb, yb, wb, hb = boxes[b]
        arrow((xa + wa, ya + ha / 2), (xb, yb + hb / 2))

    # Watermarked Decoder → Image Detector  (x_w)
    xw, yw, ww, hw = boxes["Watermarked"]
    xi, yi, wi, hi = boxes["Image"]
    arrow((xw + ww, yw + hw * 0.7), (xi, yi + hi / 2), color="#2f6b3f")
    # Watermarked Decoder → Re-Encoder (x_w)
    xr, yr, wr, hr = boxes["Re-Encoder"]
    arrow((xw + ww / 2, yw), (xr + wr / 2, yr + hr), color="#2f6b3f")
    # Re-Encoder → Latent Detector (z_re_k)
    xl, yl, wl, hl = boxes["Latent"]
    arrow((xr + wr, yr + hr / 2), (xl, yl + hl / 2), color="#2f6b3f")

    # Edge labels
    ax.text(16.5, 25.5, "$z$", fontsize=10, color="#33476b")
    ax.text(83.5, 22.0, "$x_w$", fontsize=10, color="#2f6b3f")
    ax.text(74.0, 11.5, "$x_w$", fontsize=10, color="#2f6b3f")

    ax.set_title(
        "TraceFlow pipeline: keyed rectified flow with a dual-head watermark",
        fontsize=13, fontweight="bold", pad=10,
    )
    # Legend for the two stages
    legend_handles = [
        mpatches.Patch(facecolor="#C5D9F1", edgecolor="#33476b", label="Generative path"),
        mpatches.Patch(facecolor="#F7E6D6", edgecolor="#33476b", label="Watermark embedding"),
        mpatches.Patch(facecolor="#E6F2E6", edgecolor="#2f6b3f", label="Forensic detectors"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", ncol=3,
              bbox_to_anchor=(0.0, 0.04), fontsize=9)

    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 2 — Ablation bar chart
# ---------------------------------------------------------------------------

def fig_ablation(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5.2))

    labels: List[str] = []
    img_vals: List[Optional[float]] = []
    lat_vals: List[Optional[float]] = []
    bar_colors: List[str] = []

    for label, color_key, exp_id in ABLATION_LINEUP:
        labels.append(label)
        bar_colors.append(METHOD_COLORS.get(color_key, "#8C8C8C"))
        m = experiments.get(exp_id, {}).get("metrics", {})
        img_vals.append(m.get("generated_image_bit_acc"))
        lat_vals.append(m.get("generated_latent_bit_acc"))

    grouped_bar(
        ax,
        labels,
        {"Image detector": img_vals, "Latent detector": lat_vals},
        colors={"Image detector": METRIC_COLORS["image_detector"],
                "Latent detector": METRIC_COLORS["latent_detector"]},
        value_fmt="{:.2f}",
    )

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.4,
               label="Random chance (0.5)")
    ax.axhline(0.9, color="#C44E52", linestyle="--", linewidth=1.2,
               label="Traceability target (0.9)")
    ax.set_ylabel("Generated-sample bit accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Watermark detectability across ablations")
    ax.legend(loc="upper left", ncol=2)

    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 3 — Attack reconstruction grid
# ---------------------------------------------------------------------------

def fig_attack_grid(
    experiments: Dict[str, Dict[str, Any]], out_stem: Path
) -> None:
    setup_style()

    # Pick the first available experiment with an images directory.
    img_dir: Optional[Path] = None
    chosen_exp = None
    for exp_id in GRID_EXP_PRIORITY:
        info = experiments.get(exp_id)
        if not info:
            continue
        for inv_name in ("strong_inversion_geiping", "inversion", "inversion_latent"):
            cand = info["dir"] / inv_name / "images"
            if cand.exists():
                img_dir, chosen_exp = cand, exp_id
                break
        if img_dir:
            break

    # (filename glob, panel title)
    panels = [
        ("original_grid.png",                      "Original"),
        ("latent_*_raw_nokey_grid.png",            "No-key latent decode"),
        ("latent_*_raw_defender_grid.png",         "Defender decode"),
        ("pixel_*_raw_recon_grid.png",             "Pixel attack"),
        ("latent_*_post_watermark_defender_grid.png", "Post-watermark sanity"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]

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

    src_note = f"source: {chosen_exp}" if chosen_exp else "source: (none found)"
    fig.suptitle(
        "Inversion-attack reconstructions vs. forensic decode  —  " + src_note,
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.5)
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 4 — Training / attack curves
# ---------------------------------------------------------------------------

def _gml_history(inversion: Dict[str, Any]) -> List[float]:
    runs = inversion.get("attacker_runs", {})
    for _attacker, run in runs.items():
        lat = run.get("latent_attack", {})
        if "gml_history" in lat:
            return list(lat["gml_history"])
        pix = run.get("pixel_attack", {})
        if "gml_history" in pix:
            return list(pix["gml_history"])
    return []



def fig_curves(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()

    # Choose an experiment that has a train log.
    log_rows: List[Dict[str, Any]] = []
    inversion: Dict[str, Any] = {}
    chosen_exp = None
    for exp_id in CURVES_EXP_PRIORITY:
        info = experiments.get(exp_id)
        if not info:
            continue
        log_path = _train_log_path(info["metrics"])
        if log_path is not None:
            log_rows = _read_jsonl(log_path)
            inversion = info.get("inversion", {})
            chosen_exp = exp_id
            break

    steps = [r.get("step") for r in log_rows]

    def valid_series(key: str) -> Tuple[List[Any], List[Any]]:
        xs, ys = [], []
        for s, row in zip(steps, log_rows):
            y = row.get(key)
            if s is not None and y is not None:
                xs.append(s)
                ys.append(y)
        return xs, ys

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    axes = axes.flatten()

    # Panel 1: all loss components, not just flow loss.
    ax = axes[0]
    ax.set_title("Training loss components")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    loss_specs = [
        ("loss", "total", "loss"),
        ("loss_flow", "flow", "loss_flow"),
        ("loss_wm_img", "wm image", "image_detector"),
        ("loss_wm_latent", "wm latent", "latent_detector"),
        ("loss_wm_robust", "wm robust", "image_bit_acc"),
        ("loss_clean_negative", "clean negative", "clean_false_positive"),
        ("loss_img", "image recon", "gml"),
        ("loss_cycle", "cycle", "clean_false_positive"),
        ("loss_residual", "residual", "loss"),
    ]
    has_loss = False
    for key, label, color_key in loss_specs:
        xs, ys = valid_series(key)
        if xs:
            has_loss = True
            ax.plot(xs, ys, marker="o", markersize=3, label=label,
                    color=METRIC_COLORS.get(color_key, None))
    if has_loss:
        ax.legend(loc="best", fontsize=8)
    else:
        placeholder_axis(ax, "no train log")

    # Panel 2: bit accuracy.
    ax = axes[1]
    ax.set_title("Watermark bit accuracy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Bit accuracy")
    for key, label, color_key in [
        ("bit_acc_img", "image detector", "image_bit_acc"),
        ("bit_acc_latent", "latent detector", "latent_bit_acc"),
    ]:
        xs, ys = valid_series(key)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=label,
                    color=METRIC_COLORS.get(color_key, None))
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)

    # Panel 3: BER.
    ax = axes[2]
    ax.set_title("Bit error rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("BER")
    has_ber = False
    for key, label, color_key in [
        ("ber_img", "image BER", "image_bit_acc"),
        ("ber_latent", "latent BER", "latent_bit_acc"),
    ]:
        xs, ys = valid_series(key)
        if xs:
            has_ber = True
            ax.plot(xs, ys, marker="o", markersize=3, label=label,
                    color=METRIC_COLORS.get(color_key, None))
    if has_ber:
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=8)
    else:
        placeholder_axis(ax, "no BER data")

    # Panel 4: clean false positives.
    ax = axes[3]
    ax.set_title("Clean false-positive signal")
    ax.set_xlabel("Step")
    ax.set_ylabel("Bit accuracy on clean images")
    has_fp = False
    for key, label, color_key in [
        ("clean_false_positive_img", "image detector", "clean_false_positive"),
        ("clean_false_positive_latent", "latent detector", "latent_detector"),
    ]:
        xs, ys = valid_series(key)
        if xs:
            has_fp = True
            ax.plot(xs, ys, marker="o", markersize=3, label=label,
                    color=METRIC_COLORS.get(color_key, None))
    if has_fp:
        ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=8)
    else:
        placeholder_axis(ax, "no clean FP data")

    # Panel 5: gradient-matching loss from the inversion run.
    ax = axes[4]
    ax.set_title("Gradient-matching loss")
    ax.set_xlabel("Attack step")
    ax.set_ylabel("GML")
    gml = _gml_history(inversion)
    if gml:
        ax.plot(range(1, len(gml) + 1), gml, marker="s", color=METRIC_COLORS["gml"])
    else:
        placeholder_axis(ax, "no inversion history")

    # Panel 6: run metadata.
    ax = axes[5]
    ax.axis("off")
    src = chosen_exp or "(none)"
    n_steps = len(log_rows)
    ax.text(
        0.02, 0.95,
        f"Source experiment: {src}\n"
        f"Train-log rows:    {n_steps}\n"
        f"Inversion steps:   {len(gml)}",
        ha="left", va="top", transform=ax.transAxes,
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="#f5f5f5", ec="#cccccc"),
    )

    fig.suptitle("TraceFlow training & attack curves", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0, w_pad=2.0)
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 5 — Robustness chart
# ---------------------------------------------------------------------------

def _collect_robustness(
    experiments: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, List[Optional[float]]], bool]:
    """Collect transform robustness, preferring the nested exp05 block.

    New format::

        "robustness": {
          "latent_attack": {"jpeg": {"image_bit_acc": .., "latent_bit_acc": ..}, ...},
          "pixel_attack":  {"jpeg": {"image_bit_acc": .., "latent_bit_acc": ..}, ...}
        }

    Older flat blocks are still accepted and mapped onto the latent-attack series.
    """
    series: Dict[str, List[Optional[float]]] = {
        "Latent attack / Image detector": [None] * len(ROBUSTNESS_TRANSFORMS),
        "Latent attack / Latent detector": [None] * len(ROBUSTNESS_TRANSFORMS),
        "Pixel attack / Image detector": [None] * len(ROBUSTNESS_TRANSFORMS),
        "Pixel attack / Latent detector": [None] * len(ROBUSTNESS_TRANSFORMS),
    }
    has_real = False

    def _assign_nested(attack_blob: Dict[str, Any], img_key: str, lat_key: str) -> None:
        nonlocal has_real
        for i, t in enumerate(ROBUSTNESS_TRANSFORMS):
            entry = attack_blob.get(t)
            if not isinstance(entry, dict):
                continue
            if entry.get("image_bit_acc") is not None:
                series[img_key][i] = entry["image_bit_acc"]
                has_real = True
            if entry.get("latent_bit_acc") is not None:
                series[lat_key][i] = entry["latent_bit_acc"]
                has_real = True

    for info in experiments.values():
        for blob in (info.get("metrics", {}), info.get("inversion", {})):
            rob = blob.get("robustness")
            if not isinstance(rob, dict):
                continue
            if "latent_attack" in rob or "pixel_attack" in rob:
                _assign_nested(
                    rob.get("latent_attack", {}),
                    "Latent attack / Image detector",
                    "Latent attack / Latent detector",
                )
                _assign_nested(
                    rob.get("pixel_attack", {}),
                    "Pixel attack / Image detector",
                    "Pixel attack / Latent detector",
                )
            else:
                _assign_nested(
                    rob,
                    "Latent attack / Image detector",
                    "Latent attack / Latent detector",
                )

    # Back-fill the clean reference from exp05 headline raw metrics.
    clean_idx = ROBUSTNESS_TRANSFORMS.index("clean")
    m = experiments.get("exp05", {}).get("metrics", {})
    if series["Latent attack / Image detector"][clean_idx] is None:
        if m.get("latent_raw_no_key_image_bit_acc") is not None:
            series["Latent attack / Image detector"][clean_idx] = m["latent_raw_no_key_image_bit_acc"]
        if m.get("latent_raw_no_key_latent_bit_acc") is not None:
            series["Latent attack / Latent detector"][clean_idx] = m["latent_raw_no_key_latent_bit_acc"]
    if series["Pixel attack / Image detector"][clean_idx] is None:
        if m.get("pixel_raw_image_bit_acc") is not None:
            series["Pixel attack / Image detector"][clean_idx] = m["pixel_raw_image_bit_acc"]
        if m.get("pixel_raw_latent_bit_acc") is not None:
            series["Pixel attack / Latent detector"][clean_idx] = m["pixel_raw_latent_bit_acc"]

    return series, has_real


def fig_robustness(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5.2))

    series, has_real = _collect_robustness(experiments)

    grouped_bar(
        ax,
        [t.upper() for t in ROBUSTNESS_TRANSFORMS],
        series,
        colors={
            "Latent attack / Image detector": METRIC_COLORS["image_detector"],
            "Latent attack / Latent detector": METRIC_COLORS["latent_detector"],
            "Pixel attack / Image detector": "#DD8452",
            "Pixel attack / Latent detector": "#8172B3",
        },
        value_fmt="{:.2f}",
    )

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.4,
               label="Random chance (0.5)")
    ax.axhline(0.9, color="#C44E52", linestyle="--", linewidth=1.2,
               label="Traceability target (0.9)")
    ax.set_ylabel("Recovered bit accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Transform robustness on raw inversion outputs")
    ax.legend(loc="upper right", ncol=2)

    if not has_real:
        ax.text(
            0.5, 0.92,
            "Transform robustness not yet run — only clean references are populated.\n"
            "Run exp05 to fill the nested robustness block.",
            ha="center", va="top", transform=ax.transAxes, fontsize=9,
            color="#a05050", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf3f3", ec="#e3c2c2"),
        )

    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Aggregation: summary CSV + Markdown
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    ("exp_id", "Experiment"),
    ("config", "Config"),
    ("mode", "Mode"),
    ("steps_completed", "Steps"),
    ("final_loss", "Final loss"),
    ("generated_image_bit_acc", "Gen img acc"),
    ("generated_latent_bit_acc", "Gen lat acc"),
    ("clean_false_positive_img", "Clean FP img"),
    ("latent_final_gml", "Latent GML"),
    ("latent_no_key_psnr", "No-key PSNR"),
    ("latent_raw_no_key_image_bit_acc", "Attack img acc"),
    ("latent_raw_no_key_latent_bit_acc", "Attack lat acc"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_summary(
    experiments: Dict[str, Dict[str, Any]],
    csv_path: Path,
    md_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    for exp_id in sorted(experiments):
        info = experiments[exp_id]
        m = info["metrics"]
        row = {"exp_id": exp_id, "mode": info["mode"]}
        for key, _ in SUMMARY_FIELDS:
            if key in ("exp_id", "mode"):
                continue
            row[key] = m.get(key)
        rows.append(row)

    # CSV (stdlib — pandas not required)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[k for k, _ in SUMMARY_FIELDS], extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k, _ in SUMMARY_FIELDS})

    # Markdown table
    headers = [label for _, label in SUMMARY_FIELDS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [_fmt(row.get(key)) for key, _ in SUMMARY_FIELDS]
        lines.append("| " + " | ".join(cells) + " |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")



# ---------------------------------------------------------------------------
# Figure 6 — Training dashboard across experiments
# ---------------------------------------------------------------------------

def fig_training_dashboard(experiments: Dict[str, Dict[str, Any]], out_stem: Path) -> None:
    setup_style()
    labels: List[str] = []
    final_loss: List[Optional[float]] = []
    final_flow: List[Optional[float]] = []
    final_img_acc: List[Optional[float]] = []
    final_lat_acc: List[Optional[float]] = []

    for exp_id in ["exp01", "exp02", "exp03", "exp04", "exp05"]:
        info = experiments.get(exp_id)
        if not info:
            continue
        labels.append(exp_id)
        rows: List[Dict[str, Any]] = []
        log_path = _train_log_path(info["metrics"])
        if log_path is not None:
            rows = _read_jsonl(log_path)
        last = rows[-1] if rows else {}
        m = info["metrics"]
        final_loss.append(last.get("loss", m.get("final_loss")))
        final_flow.append(last.get("loss_flow", m.get("final_loss_flow")))
        final_img_acc.append(last.get("bit_acc_img", m.get("generated_image_bit_acc")))
        final_lat_acc.append(last.get("bit_acc_latent", m.get("generated_latent_bit_acc")))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    if not labels:
        placeholder_axis(axes[0], "no experiments")
        placeholder_axis(axes[1], "no experiments")
        save_figure(fig, out_stem)
        return

    x = list(range(len(labels)))
    width = 0.35
    ax = axes[0]
    loss_vals = [v if v is not None else 0 for v in final_loss]
    flow_vals = [v if v is not None else 0 for v in final_flow]
    ax.bar([i - width / 2 for i in x], loss_vals, width, label="total loss", color=METRIC_COLORS["loss"])
    ax.bar([i + width / 2 for i in x], flow_vals, width, label="flow loss", color=METRIC_COLORS["loss_flow"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final logged loss")
    ax.set_title("Final training losses")
    ax.legend(loc="best")

    ax = axes[1]
    img_vals = [v if v is not None else 0 for v in final_img_acc]
    lat_vals = [v if v is not None else 0 for v in final_lat_acc]
    ax.bar([i - width / 2 for i in x], img_vals, width, label="image detector", color=METRIC_COLORS["image_bit_acc"])
    ax.bar([i + width / 2 for i in x], lat_vals, width, label="latent detector", color=METRIC_COLORS["latent_bit_acc"])
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bit accuracy")
    ax.set_title("Final watermark accuracy")
    ax.legend(loc="best")

    fig.suptitle("TraceFlow experiment training dashboard", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    global CURRENT_BUNDLE_DIR
    results_dir = Path(args.results_dir)
    CURRENT_BUNDLE_DIR = results_dir.parent if results_dir.name == "results" else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(results_dir, args.mode)
    print(f"[figures] results-dir: {results_dir}")
    print(f"[figures] output-dir:  {output_dir}")
    if experiments:
        for exp_id, info in sorted(experiments.items()):
            print(f"[figures]   found {exp_id} ({info['mode']}) -> {info['dir']}")
    else:
        print("[figures]   WARNING: no experiment metrics found "
              "(figures will use placeholders).")

    # Figures (each writes .png + .pdf)
    fig_pipeline(output_dir / "fig1_pipeline")
    print("[figures] wrote fig1_pipeline.{png,pdf}")
    fig_ablation(experiments, output_dir / "fig2_ablation")
    print("[figures] wrote fig2_ablation.{png,pdf}")
    fig_attack_grid(experiments, output_dir / "fig3_attack_grid")
    print("[figures] wrote fig3_attack_grid.{png,pdf}")
    fig_curves(experiments, output_dir / "fig4_curves")
    print("[figures] wrote fig4_curves.{png,pdf}")
    fig_robustness(experiments, output_dir / "fig5_robustness")
    print("[figures] wrote fig5_robustness.{png,pdf}")
    fig_training_dashboard(experiments, output_dir / "fig6_training_dashboard")
    print("[figures] wrote fig6_training_dashboard.{png,pdf}")

    # Aggregation
    write_summary(
        experiments,
        output_dir / "summary.csv",
        output_dir / "summary.md",
    )
    print("[figures] wrote summary.csv and summary.md")

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate publication-style TraceFlow figures and tables."
    )
    p.add_argument(
        "--results-dir", dest="results_dir", default="results/traceflow",
        help="Directory containing <exp>/<smoke|full>/metrics.json.",
    )
    p.add_argument(
        "--output-dir", dest="output_dir",
        default="results/traceflow/figures",
        help="Directory for generated figures and summary tables.",
    )
    p.add_argument(
        "--mode", default="auto", choices=["auto", "smoke", "full"],
        help="Which run mode to read (auto = prefer full, else smoke).",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))
