from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOP_COLORS = ["#f2b3b3", "#f6d2a8", "#f8efb0"]  # red, orange, yellow


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _format_float(v: float) -> str:
    if np.isnan(v):
        return "-"
    if abs(v) >= 100:
        return f"{v:.2f}"
    if abs(v) >= 1:
        return f"{v:.4f}"
    if abs(v) >= 1e-3:
        return f"{v:.6f}"
    return f"{v:.2e}"


def _find_topk_indices(values: list[float], k: int, larger_better: bool) -> list[int]:
    pairs = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
    if larger_better:
        pairs.sort(key=lambda x: x[1], reverse=True)
    else:
        pairs.sort(key=lambda x: x[1])
    return [i for i, _ in pairs[:k]]


def build_display_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        method = r.get("method", "-")
        dataset = r.get("dataset", "-")
        bs = r.get("batch_size", "-")
        init = r.get("init", "-")

        if dataset != "-":
            method_cell = f"{dataset}-{method}"
        else:
            method_cell = f"{method}-bs{bs}-{init}"

        out.append(
            {
                "Method": method_cell,
                "MSE": _format_float(_to_float(r.get("final_mse", "nan"))),
                "SSIM": _format_float(_to_float(r.get("final_ssim", "nan"))),
                "PSNR": _format_float(_to_float(r.get("final_psnr", "nan"))),
                "Loss": _format_float(_to_float(r.get("final_loss", "nan"))),
            }
        )
    return out


def render_table(
    rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> None:
    columns = ["Method", "MSE", "SSIM", "PSNR", "Loss"]
    cell_text = [[r[c] for c in columns] for r in rows]

    n_rows = len(rows)
    fig_h = max(3.2, 0.42 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#efefef")
            cell.set_text_props(weight="bold")

    raw_mse = [_to_float(r.get("final_mse", "nan")) for r in raw_rows]
    raw_ssim = [_to_float(r.get("final_ssim", "nan")) for r in raw_rows]
    raw_psnr = [_to_float(r.get("final_psnr", "nan")) for r in raw_rows]
    raw_loss = [_to_float(r.get("final_loss", "nan")) for r in raw_rows]

    metric_map = {
        1: (raw_mse, False),
        2: (raw_ssim, True),
        3: (raw_psnr, True),
        4: (raw_loss, False),
    }

    for col_idx, (vals, larger_better) in metric_map.items():
        best_rows = _find_topk_indices(vals, k=3, larger_better=larger_better)
        for rank, ridx in enumerate(best_rows):
            table[(ridx + 1, col_idx)].set_facecolor(TOP_COLORS[rank])

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)

    fig.tight_layout()
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render publication-style metrics table with top-3 highlighting.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/idlg_mnist_full/all_results.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/idlg_tables"),
    )
    p.add_argument(
        "--title",
        type=str,
        default="Quantitative Comparison (Top-3 highlighted)",
    )
    p.add_argument(
        "--name",
        type=str,
        default="metrics_table",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_csv, "r", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    display_rows = build_display_rows(raw_rows)

    out_png = args.output_dir / f"{args.name}.png"
    out_pdf = args.output_dir / f"{args.name}.pdf"
    render_table(
        rows=display_rows,
        raw_rows=raw_rows,
        out_png=out_png,
        out_pdf=out_pdf,
        title=args.title,
    )

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()