"""
src/utils/plotting.py
=====================
Publication-style plotting helpers for TraceFlow figures.

Design goals
------------
* Minimal dependencies: only ``matplotlib`` is required.  ``numpy`` is used if
  present (it ships with matplotlib) but never assumed beyond that.
* Consistent house style: a single colour palette, readable fonts, vector-ready
  output.  Every figure is saved as **both** PNG (raster preview) and PDF
  (vector, for LaTeX inclusion).
* No "toy screenshot" output — axes are labelled, legends are placed, and
  colours are stable across figures so the same method always reads the same.

Public API
----------
``setup_style()``           — apply the house matplotlib rcParams.
``save_figure(fig, stem)``  — write ``<stem>.png`` and ``<stem>.pdf``.
``METHOD_COLORS``           — stable method → colour map.
``METRIC_COLORS``           — stable metric → colour map.
``color_for(name, table)``  — palette lookup with deterministic fallback.
``load_image(path)``        — read a PNG into an array for imshow (or ``None``).
``placeholder_axis(ax, msg)`` — draw a "data not available" panel.
``grouped_bar(ax, ...)``    — grouped bar chart primitive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

# A colour-blind-friendly qualitative palette (Okabe–Ito derived).
PALETTE: List[str] = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
    "#CCB974",  # gold
    "#64B5CD",  # cyan
]

# Stable method → colour map so the same ablation always reads the same colour.
METHOD_COLORS: Dict[str, str] = {
    "baseline":    "#8C8C8C",  # grey  — lower bound, no protection
    "keyed":       "#4C72B0",  # blue  — key only
    "traceflow_identity": "#55A868",
    "traceflow":          "#C44E52",
}

# Stable metric → colour map for curves / bars.
METRIC_COLORS: Dict[str, str] = {
    "loss":               "#C44E52",
    "loss_flow":          "#C44E52",
    "image_bit_acc":      "#4C72B0",
    "latent_bit_acc":     "#55A868",
    "clean_false_positive": "#937860",
    "gml":                "#8172B3",
    "image_detector":     "#4C72B0",
    "latent_detector":    "#55A868",
}


def setup_style() -> None:
    """Apply the TraceFlow house matplotlib style (idempotent)."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.autolayout": False,
    })


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_figure(
    fig: "plt.Figure",
    stem: Union[str, Path],
    *,
    formats: Sequence[str] = ("png", "pdf"),
    close: bool = True,
) -> List[Path]:
    """Save *fig* to ``<stem>.<fmt>`` for each format in *formats*.

    Parent directories are created.  Returns the list of written paths.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        written.append(out)
    if close:
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Colour lookup
# ---------------------------------------------------------------------------

def color_for(name: str, table: Optional[Dict[str, str]] = None) -> str:
    """Look up *name* in *table*; fall back to a deterministic palette slot."""
    table = table if table is not None else METHOD_COLORS
    if name in table:
        return table[name]
    idx = (abs(hash(name)) % len(PALETTE))
    return PALETTE[idx]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: Union[str, Path]):
    """Read a PNG image into an array for ``imshow``.  Returns ``None`` on error."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return plt.imread(str(path))
    except Exception:  # noqa: BLE001 — figure code should never crash on a bad PNG
        return None


# ---------------------------------------------------------------------------
# Placeholder panel
# ---------------------------------------------------------------------------

def placeholder_axis(ax: "plt.Axes", message: str) -> None:
    """Render a clean 'data not available' panel into *ax*."""
    ax.axis("off")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=10, color="#888888", style="italic",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#cccccc"),
    )


# ---------------------------------------------------------------------------
# Grouped bar primitive
# ---------------------------------------------------------------------------

def grouped_bar(
    ax: "plt.Axes",
    group_labels: Sequence[str],
    series: Dict[str, Sequence[Optional[float]]],
    *,
    colors: Optional[Dict[str, str]] = None,
    bar_width: float = 0.8,
    value_fmt: str = "{:.2f}",
    annotate: bool = True,
    na_height: float = 0.0,
) -> None:
    """Draw a grouped bar chart.

    Args:
        ax:           Target axis.
        group_labels: X-axis group labels (one tick per group).
        series:       Mapping ``series_name -> [value per group]``.  ``None``
                      entries are rendered as hatched "n/a" bars.
        colors:       Optional series_name -> colour.  Falls back to palette.
        bar_width:    Total width occupied by each group's bars.
        value_fmt:    Format string for value annotations.
        annotate:     Annotate non-null bars with their value.
        na_height:    Bar height used to visualise missing (``None``) values.
    """
    n_series = max(1, len(series))
    n_groups = len(group_labels)
    x = list(range(n_groups))
    each = bar_width / n_series

    for s_idx, (s_name, values) in enumerate(series.items()):
        offset = (s_idx - (n_series - 1) / 2.0) * each
        positions = [xi + offset for xi in x]
        heights = [na_height if (v is None) else float(v) for v in values]
        col = (colors or {}).get(s_name) or color_for(s_name, METRIC_COLORS)
        bars = ax.bar(
            positions, heights, width=each * 0.92,
            label=s_name, color=col, edgecolor="white", linewidth=0.6,
        )
        for bar, raw in zip(bars, values):
            if raw is None:
                bar.set_hatch("///")
                bar.set_alpha(0.35)
                ax.text(
                    bar.get_x() + bar.get_width() / 2, na_height + 0.01,
                    "n/a", ha="center", va="bottom", fontsize=8, color="#999999",
                )
            elif annotate:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, float(raw),
                    value_fmt.format(float(raw)),
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
