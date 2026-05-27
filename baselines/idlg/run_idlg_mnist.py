from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.idlg.idlg_attack import (
    AttackConfig,
    run_idlg,
    save_metric_curve,
    save_recon_vs_original,
    save_three_phase_comparison,
)
from baselines.idlg.model import LeNet, weights_init


@dataclass
class Experiment:
    batch_size: int
    init: str
    method: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_batch(data_root: Path, batch_size: int, sample_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    tt = transforms.ToTensor()
    dataset = datasets.MNIST(root=str(data_root), train=False, download=True, transform=tt)
    rng = np.random.default_rng(sample_seed)
    idxs = rng.choice(len(dataset), size=batch_size, replace=False)
    xs = []
    ys = []
    for idx in idxs:
        x, y = dataset[int(idx)]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def build_summary(metrics: list[dict[str, float]]) -> dict[str, float]:
    final = metrics[-1]
    best_ssim = max(m["ssim"] for m in metrics)
    best_psnr = max(m["psnr"] for m in metrics)
    best_mse = min(m["mse"] for m in metrics)
    return {
        "final_loss": final["loss"],
        "final_mse": final["mse"],
        "final_ssim": final["ssim"],
        "final_psnr": final["psnr"],
        "best_mse": best_mse,
        "best_ssim": best_ssim,
        "best_psnr": best_psnr,
    }


def run_experiment(
    exp: Experiment,
    gt_data: torch.Tensor,
    gt_label: torch.Tensor,
    iters: int,
    lr: float,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    net = LeNet(channel=1, hidden=588, num_classes=10).to(device)
    net.apply(weights_init)

    cfg = AttackConfig(iters=iters, lr=lr, init=exp.init, method=exp.method)
    result = run_idlg(
        net=net,
        gt_data=gt_data.to(device),
        gt_label=gt_label.to(device),
        cfg=cfg,
        device=device,
    )

    run_name = f"{exp.method}_bs{exp.batch_size}_{exp.init}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_three_phase_comparison(
        gt=gt_data,
        snapshots=result["snapshots"],
        phase_iters=result["phase_iters"],
        out_path=run_dir / "three_phase.png",
        suptitle=f"{run_name}: Three-Phase Reconstruction",
    )
    save_recon_vs_original(
        gt=gt_data,
        rec=result["reconstructed"],
        out_path=run_dir / "recon_vs_original.png",
        suptitle=f"{run_name}: Reconstructed vs Original",
    )
    save_metric_curve(
        metrics=result["metrics"],
        out_path=run_dir / "metrics_curve.png",
        suptitle=f"{run_name}: Metric Trends",
    )

    with open(run_dir / "iter_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2, ensure_ascii=False)

    summary = build_summary(result["metrics"])
    payload = {
        "run_name": run_name,
        "experiment": asdict(exp),
        "summary": summary,
        "gt_label": gt_label.tolist(),
        "pred_label": result["pred_label"].tolist(),
        "phase_iters": result["phase_iters"],
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce iDLG on MNIST with staged visualizations.")
    parser.add_argument("--iters", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--method", type=str, default="iDLG", choices=["iDLG", "DLG"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--inits", type=str, nargs="+", default=["normal", "uniform", "zeros"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/idlg_mnist"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_records: list[dict[str, Any]] = []

    for bs in args.batch_sizes:
        gt_data, gt_label = get_batch(args.data_root, bs, args.sample_seed)
        for init in args.inits:
            exp = Experiment(batch_size=bs, init=init, method=args.method)
            record = run_experiment(
                exp=exp,
                gt_data=gt_data,
                gt_label=gt_label,
                iters=args.iters,
                lr=args.lr,
                device=device,
                out_dir=out_dir,
            )
            all_records.append(record)
            summary = record["summary"]
            print(
                f"[Done] {record['run_name']} | "
                f"MSE={summary['final_mse']:.6f}, SSIM={summary['final_ssim']:.4f}, PSNR={summary['final_psnr']:.2f}"
            )

    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    with open(out_dir / "all_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_name",
                "method",
                "batch_size",
                "init",
                "final_loss",
                "final_mse",
                "final_ssim",
                "final_psnr",
                "best_mse",
                "best_ssim",
                "best_psnr",
            ]
        )
        for rec in all_records:
            s = rec["summary"]
            e = rec["experiment"]
            writer.writerow(
                [
                    rec["run_name"],
                    e["method"],
                    e["batch_size"],
                    e["init"],
                    s["final_loss"],
                    s["final_mse"],
                    s["final_ssim"],
                    s["final_psnr"],
                    s["best_mse"],
                    s["best_ssim"],
                    s["best_psnr"],
                ]
            )

    print(f"Saved reports to: {out_dir}")


if __name__ == "__main__":
    main()