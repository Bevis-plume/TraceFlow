from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AttackConfig:
    iters: int = 300
    lr: float = 1.0
    init: str = "normal"
    method: str = "iDLG"


def infer_labels_idlg(original_grads: list[torch.Tensor], batch_size: int) -> torch.Tensor:
    """Infer labels from the FC-layer gradient.

    Exact for CE + one-hot when batch_size == 1; heuristic top-k otherwise.
    """
    grad_fc_weight = original_grads[-2]
    score = torch.sum(grad_fc_weight, dim=-1)
    if batch_size == 1:
        return torch.argmin(score).view(1)
    return torch.topk(-score, k=batch_size).indices.view(batch_size)


def build_dummy(shape: torch.Size, init: str, device: torch.device) -> torch.Tensor:
    if init == "zeros":
        dummy = torch.zeros(shape, device=device)
    elif init == "uniform":
        dummy = torch.rand(shape, device=device)
    else:
        dummy = torch.randn(shape, device=device)
    return dummy.requires_grad_(True)


def mse(gt: torch.Tensor, rec: torch.Tensor) -> float:
    return torch.mean((gt - rec) ** 2).item()


def psnr(gt: torch.Tensor, rec: torch.Tensor, max_val: float = 1.0) -> float:
    err = torch.mean((gt - rec) ** 2)
    if err.item() == 0:
        return float("inf")
    return (10.0 * torch.log10(torch.tensor(max_val * max_val, device=gt.device) / err)).item()


def ssim_torch(gt: torch.Tensor, rec: torch.Tensor) -> float:
    """Small SSIM implementation for grayscale/RGB batched tensors in [0, 1]."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = gt.mean(dim=(2, 3), keepdim=True)
    mu_y = rec.mean(dim=(2, 3), keepdim=True)
    sig_x = ((gt - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_y = ((rec - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_xy = ((gt - mu_x) * (rec - mu_y)).mean(dim=(2, 3), keepdim=True)
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sig_x + sig_y + c2)
    return (num / den).mean().item()


def run_idlg(
    net: nn.Module,
    gt_data: torch.Tensor,
    gt_label: torch.Tensor,
    cfg: AttackConfig,
    device: torch.device,
    snapshot_iters: list[int] | None = None,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss().to(device)

    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = [g.detach().clone() for g in dy_dx]

    dummy_data = build_dummy(gt_data.size(), cfg.init, device)
    if cfg.method == "DLG":
        dummy_label = torch.randn((gt_data.shape[0], out.shape[-1]), device=device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=cfg.lr)
        inferred_label = None
    else:
        inferred_label = infer_labels_idlg(original_dy_dx, gt_data.shape[0]).to(device)
        optimizer = torch.optim.LBFGS([dummy_data], lr=cfg.lr)
        dummy_label = None

    # Three phases: early / middle / late.
    phase_iters = sorted({max(1, cfg.iters // 6), max(1, cfg.iters // 2), cfg.iters - 1})
    capture_iters = set(phase_iters)
    if snapshot_iters is not None:
        capture_iters.update(int(it) for it in snapshot_iters if 0 <= int(it) < cfg.iters)
    snapshots: dict[int, torch.Tensor] = {}
    metrics: list[dict[str, float]] = []

    for it in range(cfg.iters):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            pred = net(dummy_data)
            if cfg.method == "DLG":
                dummy_loss = -torch.mean(
                    torch.sum(
                        torch.softmax(dummy_label, dim=-1)
                        * torch.log(torch.softmax(pred, dim=-1) + 1e-12),
                        dim=-1,
                    )
                )
            else:
                dummy_loss = criterion(pred, inferred_label)

            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            grad_diff = torch.tensor(0.0, device=device)
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff = grad_diff + ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        loss = optimizer.step(closure)
        with torch.no_grad():
            rec = dummy_data.detach().clamp(0.0, 1.0)
            rec_mse = mse(gt_data, rec)
            rec_ssim = ssim_torch(gt_data, rec)
            rec_psnr = psnr(gt_data, rec)
            metrics.append(
                {
                    "iter": float(it),
                    "loss": float(loss.item()) if hasattr(loss, "item") else float(loss),
                    "mse": rec_mse,
                    "ssim": rec_ssim,
                    "psnr": rec_psnr,
                }
            )
            if it in capture_iters:
                snapshots[it] = rec.detach().cpu().clone()

    if cfg.method == "DLG":
        pred_label = torch.argmax(dummy_label.detach(), dim=-1).cpu()
    else:
        pred_label = inferred_label.detach().cpu()

    return {
        "reconstructed": dummy_data.detach().clamp(0.0, 1.0).cpu(),
        "phase_iters": phase_iters,
        "snapshots": snapshots,
        "metrics": metrics,
        "pred_label": pred_label,
    }


def _show_img(ax: Any, tensor_chw: torch.Tensor, title: str) -> None:
    img = tensor_chw.detach().cpu().permute(1, 2, 0).numpy()
    if img.shape[-1] == 1:
        ax.imshow(np.clip(img[..., 0], 0.0, 1.0), cmap="gray")
    else:
        ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title(title)
    ax.axis("off")


def save_three_phase_comparison(
    gt: torch.Tensor,
    snapshots: dict[int, torch.Tensor],
    phase_iters: list[int],
    out_path: Path,
    suptitle: str,
) -> None:
    rows = gt.shape[0]
    cols = 4
    fig = plt.figure(figsize=(3.0 * cols, 2.6 * rows))
    for r in range(rows):
        _show_img(plt.subplot(rows, cols, r * cols + 1), gt[r], "Original")
        _show_img(plt.subplot(rows, cols, r * cols + 2), snapshots[phase_iters[0]][r], f"Phase-1 ({phase_iters[0]})")
        _show_img(plt.subplot(rows, cols, r * cols + 3), snapshots[phase_iters[1]][r], f"Phase-2 ({phase_iters[1]})")
        _show_img(plt.subplot(rows, cols, r * cols + 4), snapshots[phase_iters[2]][r], f"Phase-3 ({phase_iters[2]})")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_recon_vs_original(gt: torch.Tensor, rec: torch.Tensor, out_path: Path, suptitle: str) -> None:
    n = gt.shape[0]
    fig = plt.figure(figsize=(2.8 * n, 5.2))
    for i in range(n):
        _show_img(plt.subplot(2, n, i + 1), gt[i], f"Original #{i}")
        _show_img(plt.subplot(2, n, n + i + 1), rec[i], f"Reconstructed #{i}")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_metric_curve(metrics: list[dict[str, float]], out_path: Path, suptitle: str) -> None:
    xs = [m["iter"] for m in metrics]
    mse_vals = [m["mse"] for m in metrics]
    ssim_vals = [m["ssim"] for m in metrics]
    psnr_vals = [m["psnr"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].plot(xs, mse_vals, color="#1f77b4")
    axes[0].set_title("MSE")
    axes[0].set_xlabel("Iter")
    axes[0].grid(alpha=0.35)

    axes[1].plot(xs, ssim_vals, color="#d62728")
    axes[1].set_title("SSIM")
    axes[1].set_xlabel("Iter")
    axes[1].grid(alpha=0.35)

    axes[2].plot(xs, psnr_vals, color="#2ca02c")
    axes[2].set_title("PSNR")
    axes[2].set_xlabel("Iter")
    axes[2].grid(alpha=0.35)

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)