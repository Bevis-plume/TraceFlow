from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.idlg.idlg_attack import AttackConfig, run_idlg
from baselines.idlg.model import LeNet, weights_init


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _show_gray(ax: plt.Axes, tensor_chw: torch.Tensor, title: str = "") -> None:
    img = tensor_chw.detach().cpu().numpy()[0]
    ax.imshow(np.clip(img, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")


def _draw_montage(fig: plt.Figure, grid_spec: Any, images: list[torch.Tensor], titles: list[str], block_title: str) -> None:
    rows = 3
    cols = 10
    sub = grid_spec.subgridspec(rows, cols, wspace=0.03, hspace=0.25)
    for i in range(rows * cols):
        ax = fig.add_subplot(sub[i // cols, i % cols])
        if i < len(images):
            _show_gray(ax, images[i], titles[i])
        else:
            ax.axis("off")
    x0, y0, x1, y1 = grid_spec.get_position(fig).extents
    fig.text((x0 + x1) / 2.0, y1 + 0.01, block_title, ha="center", va="bottom", fontsize=14, fontweight="bold")


def save_paper_progression(
    gt: torch.Tensor,
    dlg_snapshots: dict[int, torch.Tensor],
    idlg_snapshots: dict[int, torch.Tensor],
    iter_list: list[int],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(24, 8), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, wspace=0.06)

    dlg_imgs = [gt[0]] + [dlg_snapshots[it][0] for it in iter_list]
    dlg_titles = ["original"] + [f"iter={it}" for it in iter_list]

    idlg_imgs = [gt[0]] + [idlg_snapshots[it][0] for it in iter_list]
    idlg_titles = ["original"] + [f"iter={it}" for it in iter_list]

    _draw_montage(fig, outer[0, 0], dlg_imgs, dlg_titles, "DLG")
    _draw_montage(fig, outer[0, 1], idlg_imgs, idlg_titles, "iDLG")

    fig.suptitle(
        "Training Process Comparison on MNIST (paper-style): DLG vs iDLG",
        fontsize=16,
        y=0.98,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_convergence_curves(dlg_metrics: list[dict[str, float]], idlg_metrics: list[dict[str, float]], out_path: Path) -> None:
    xs = [m["iter"] for m in dlg_metrics]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, key, title in [
        (axes[0, 0], "loss", "Gradient Matching Loss"),
        (axes[0, 1], "mse", "MSE"),
        (axes[1, 0], "ssim", "SSIM"),
        (axes[1, 1], "psnr", "PSNR"),
    ]:
        ax.plot(xs, [m[key] for m in dlg_metrics], color="#1f77b4", label="DLG", linewidth=1.6)
        ax.plot(xs, [m[key] for m in idlg_metrics], color="#d62728", label="iDLG", linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.35)
        ax.legend()

    fig.suptitle("Convergence Curves: DLG vs iDLG", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_ablation_heatmaps(full_csv: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(open(full_csv, "r", encoding="utf-8")))
    for r in rows:
        r["batch_size"] = int(r["batch_size"])
        r["final_mse"] = float(r["final_mse"])
        r["final_ssim"] = float(r["final_ssim"])
        r["final_psnr"] = float(r["final_psnr"])

    batch_sizes = sorted({r["batch_size"] for r in rows})
    inits = ["normal", "uniform", "zeros"]

    def table(metric: str) -> np.ndarray:
        arr = np.zeros((len(batch_sizes), len(inits)), dtype=float)
        for i, bs in enumerate(batch_sizes):
            for j, init in enumerate(inits):
                val = [r[metric] for r in rows if r["batch_size"] == bs and r["init"] == init][0]
                arr[i, j] = val
        return arr

    mse_tbl = table("final_mse")
    ssim_tbl = table("final_ssim")
    psnr_tbl = table("final_psnr")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    for ax, data, title, cmap in [
        (axes[0], mse_tbl, "MSE (lower better)", "YlOrRd"),
        (axes[1], ssim_tbl, "SSIM (higher better)", "YlGn"),
        (axes[2], psnr_tbl, "PSNR (higher better)", "YlGnBu"),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(inits)), inits)
        ax.set_yticks(range(len(batch_sizes)), [str(b) for b in batch_sizes])
        ax.set_xlabel("init")
        ax.set_ylabel("batch size")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3g}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Ablation Heatmaps (MNIST, iDLG)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-style visualization pages for iDLG experiments.")
    p.add_argument("--iters", type=int, default=281)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sample-index", type=int, default=7)
    p.add_argument("--snapshot-step", type=int, default=10)
    p.add_argument("--init", type=str, default="normal", choices=["normal", "uniform", "zeros"])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--full-results-csv", type=Path, default=Path("results/idlg_mnist_full/all_results.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/idlg_paper_style"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = datasets.MNIST(root=str(args.data_root), train=False, download=True, transform=transforms.ToTensor())
    gt_img, gt_label = ds[int(args.sample_index)]
    gt_data = gt_img.unsqueeze(0)
    gt_label_t = torch.tensor([int(gt_label)], dtype=torch.long)

    iter_list = list(range(0, args.iters, args.snapshot_step))
    if iter_list[-1] != args.iters - 1:
        iter_list.append(args.iters - 1)

    results: dict[str, dict[str, object]] = {}
    for method in ["DLG", "iDLG"]:
        seed_everything(args.seed)
        net = LeNet(channel=1, hidden=588, num_classes=10).to(device)
        net.apply(weights_init)
        cfg = AttackConfig(iters=args.iters, lr=args.lr, init=args.init, method=method)
        results[method] = run_idlg(
            net=net,
            gt_data=gt_data.to(device),
            gt_label=gt_label_t.to(device),
            cfg=cfg,
            device=device,
            snapshot_iters=iter_list,
        )

    save_paper_progression(
        gt=gt_data,
        dlg_snapshots=results["DLG"]["snapshots"],
        idlg_snapshots=results["iDLG"]["snapshots"],
        iter_list=iter_list,
        out_path=out_dir / "paper_fig_training_process_dlg_vs_idlg.png",
    )

    save_convergence_curves(
        dlg_metrics=results["DLG"]["metrics"],
        idlg_metrics=results["iDLG"]["metrics"],
        out_path=out_dir / "paper_fig_convergence_curves_dlg_vs_idlg.png",
    )

    if args.full_results_csv.exists():
        save_ablation_heatmaps(
            full_csv=args.full_results_csv,
            out_path=out_dir / "paper_fig_ablation_heatmaps.png",
        )

    summary = {
        "sample_index": args.sample_index,
        "gt_label": int(gt_label),
        "iters": args.iters,
        "snapshot_step": args.snapshot_step,
        "init": args.init,
        "dlg_final": results["DLG"]["metrics"][-1],
        "idlg_final": results["iDLG"]["metrics"][-1],
        "iter_list": iter_list,
    }
    with open(out_dir / "paper_viz_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved paper-style figures to: {out_dir}")


if __name__ == "__main__":
    main()