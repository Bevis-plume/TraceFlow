from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

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


def get_sample(dataset_name: str, data_root: Path, sample_index: int) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    dataset_name = dataset_name.lower()
    tt = transforms.ToTensor()

    if dataset_name == "mnist":
        ds = datasets.MNIST(root=str(data_root), train=False, download=True, transform=tt)
        hidden = 588
        channel = 1
        num_classes = 10
    elif dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=tt)
        hidden = 768
        channel = 3
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    x, y = ds[int(sample_index)]
    gt_data = x.unsqueeze(0)
    gt_label = torch.tensor([int(y)], dtype=torch.long)
    return gt_data, gt_label, channel, hidden, num_classes


def _show_img(ax: Any, tensor_chw: torch.Tensor, title: str = "") -> None:
    img = tensor_chw.detach().cpu().permute(1, 2, 0).numpy()
    if img.shape[-1] == 1:
        ax.imshow(np.clip(img[..., 0], 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(np.clip(img, 0.0, 1.0))
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_cross_dataset_progression(
    rows_payload: list[dict[str, Any]],
    iter_list: list[int],
    out_path: Path,
) -> None:
    # Layout is compact: 4 rows x (1 original + N snapshots)
    cols = 1 + len(iter_list)
    rows = len(rows_payload)
    fig = plt.figure(figsize=(2.1 * cols, 2.1 * rows))

    for r, payload in enumerate(rows_payload):
        gt = payload["gt_data"]
        snapshots = payload["result"]["snapshots"]
        left_title = f"{payload['dataset'].upper()}-{payload['method']}"

        ax = plt.subplot(rows, cols, r * cols + 1)
        _show_img(ax, gt[0], f"{left_title}\noriginal")

        for c, it in enumerate(iter_list, start=2):
            ax = plt.subplot(rows, cols, r * cols + c)
            _show_img(ax, snapshots[it][0], f"iter={it}")

    fig.suptitle("Cross-Dataset Reconstruction Trajectory (MNIST vs CIFAR10)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_cross_dataset_curves(rows_payload: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    metric_keys = ["mse", "ssim", "psnr"]
    titles = ["MSE (lower better)", "SSIM (higher better)", "PSNR (higher better)"]

    for payload in rows_payload:
        metrics = payload["result"]["metrics"]
        xs = [m["iter"] for m in metrics]
        tag = f"{payload['dataset'].upper()}-{payload['method']}"
        for ax, key, title in zip(axes, metric_keys, titles):
            ax.plot(xs, [m[key] for m in metrics], linewidth=1.5, label=tag)
            ax.set_title(title)
            ax.set_xlabel("iteration")
            ax.grid(alpha=0.35)

    axes[0].legend(fontsize=8)
    fig.suptitle("Cross-Dataset Convergence Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate compact cross-dataset iDLG/DLG comparison pages.")
    p.add_argument("--iters", type=int, default=201)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--init", type=str, default="normal", choices=["normal", "uniform", "zeros"])
    p.add_argument("--snapshot-iters", type=int, nargs="+", default=[0, 40, 80, 120, 160, 200])
    p.add_argument("--mnist-index", type=int, default=7)
    p.add_argument("--cifar10-index", type=int, default=7)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results/idlg_cross_dataset"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_plan = [
        ("mnist", args.mnist_index),
        ("cifar10", args.cifar10_index),
    ]

    snapshot_iters = [i for i in args.snapshot_iters if 0 <= i < args.iters]
    if snapshot_iters[-1] != args.iters - 1:
        snapshot_iters.append(args.iters - 1)

    rows_payload: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    for dataset_name, sample_idx in datasets_plan:
        gt_data, gt_label, channel, hidden, num_classes = get_sample(dataset_name, args.data_root, sample_idx)

        for method in ["DLG", "iDLG"]:
            seed_everything(args.seed)
            net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes).to(device)
            net.apply(weights_init)
            cfg = AttackConfig(iters=args.iters, lr=args.lr, init=args.init, method=method)

            result = run_idlg(
                net=net,
                gt_data=gt_data.to(device),
                gt_label=gt_label.to(device),
                cfg=cfg,
                device=device,
                snapshot_iters=snapshot_iters,
            )

            payload = {
                "dataset": dataset_name,
                "method": method,
                "gt_data": gt_data,
                "gt_label": int(gt_label.item()),
                "result": result,
            }
            rows_payload.append(payload)

            final = result["metrics"][-1]
            row = {
                "dataset": dataset_name,
                "method": method,
                "init": args.init,
                "sample_index": sample_idx,
                "gt_label": int(gt_label.item()),
                "pred_label": int(result["pred_label"][0].item()),
                "final_loss": final["loss"],
                "final_mse": final["mse"],
                "final_ssim": final["ssim"],
                "final_psnr": final["psnr"],
            }
            final_rows.append(row)

            with open(raw_dir / f"{dataset_name}_{method.lower()}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(result["metrics"], f, indent=2, ensure_ascii=False)

    save_cross_dataset_progression(
        rows_payload=rows_payload,
        iter_list=snapshot_iters,
        out_path=output_dir / "cross_dataset_progression.png",
    )
    save_cross_dataset_curves(
        rows_payload=rows_payload,
        out_path=output_dir / "cross_dataset_curves.png",
    )

    with open(output_dir / "cross_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "settings": {
                    "iters": args.iters,
                    "lr": args.lr,
                    "init": args.init,
                    "snapshot_iters": snapshot_iters,
                },
                "runs": final_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(output_dir / "cross_dataset_final_metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "init",
                "sample_index",
                "gt_label",
                "pred_label",
                "final_loss",
                "final_mse",
                "final_ssim",
                "final_psnr",
            ],
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Saved compact cross-dataset outputs to: {output_dir}")


if __name__ == "__main__":
    main()