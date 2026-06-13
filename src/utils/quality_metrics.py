"""
src/utils/quality_metrics.py
============================
Paper-level image quality helpers for TraceFlow.

All public functions accept image tensors in [-1, 1] unless documented
otherwise. Optional dependencies (lpips / torchmetrics) are used when available
and reported as warnings when missing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.utils.metrics import psnr as _psnr_01, ssim as _ssim_01


def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def mse_01(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(to_01(a), to_01(b)).item()


def linf_01(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max(torch.abs(to_01(a) - to_01(b))).item()


def psnr_01(a: torch.Tensor, b: torch.Tensor) -> float:
    return _psnr_01(to_01(a), to_01(b), max_val=1.0).item()


def ssim_01(a: torch.Tensor, b: torch.Tensor) -> float:
    return _ssim_01(to_01(a), to_01(b)).item()


def ms_ssim_01(a: torch.Tensor, b: torch.Tensor, levels: int = 4) -> float:
    vals = []
    x = to_01(a)
    y = to_01(b)
    for _ in range(max(1, levels)):
        if min(x.shape[-2:]) < 16:
            break
        vals.append(_ssim_01(x, y).clamp(min=0.0, max=1.0))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        y = F.avg_pool2d(y, kernel_size=2, stride=2)
    if not vals:
        return float("nan")
    return torch.stack(vals).mean().item()


@lru_cache(maxsize=2)
def _lpips_model(net: str = "alex", device: str = "cpu"):
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    model = lpips.LPIPS(net=net)
    model.eval()
    return model.to(torch.device(device))


def lpips_distance(a: torch.Tensor, b: torch.Tensor, device: Optional[torch.device] = None) -> Optional[float]:
    dev = device or a.device
    model = _lpips_model("alex", str(dev))
    if model is None:
        return None
    with torch.no_grad():
        aa = a.to(dev).clamp(-1.0, 1.0)
        bb = b.to(dev).clamp(-1.0, 1.0)
        return float(model(aa, bb).mean().item())


def pair_quality(a: torch.Tensor, b: torch.Tensor, prefix: str = "") -> Dict[str, Any]:
    """MSE/PSNR/SSIM/MS-SSIM/Linf/LPIPS for image tensors in [-1, 1]."""
    key = (lambda name: f"{prefix}_{name}" if prefix else name)
    out: Dict[str, Any] = {
        key("mse"): mse_01(a, b),
        key("psnr"): psnr_01(a, b),
        key("ssim"): ssim_01(a, b),
        key("ms_ssim"): ms_ssim_01(a, b),
        key("linf"): linf_01(a, b),
    }
    lp = lpips_distance(a, b, device=a.device)
    if lp is not None:
        out[key("lpips")] = lp
    else:
        out[key("lpips")] = None
        out.setdefault("warnings", []).append("lpips package/weights unavailable; LPIPS skipped")
    return out


def _iter_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
    blocked = ("grid", "trajectory", "watermarked")
    return sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in exts and not any(tok in p.stem.lower() for tok in blocked)
    ])


def load_image_batch(paths: Iterable[Path], image_size: int = 256, max_images: int = 256) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    imgs = []
    for path in list(paths)[:max_images]:
        with Image.open(path) as im:
            imgs.append(tfm(im.convert("RGB")))
    if not imgs:
        raise ValueError("No images found for quality metric batch.")
    return torch.stack(imgs, dim=0)


def distribution_metrics(
    real_dir: Path,
    fake_dir: Path,
    *,
    image_size: int = 256,
    max_images: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Compute FID/KID/Inception Score using torchmetrics when available."""
    real_paths = _iter_images(real_dir)
    fake_paths = _iter_images(fake_dir)
    out: Dict[str, Any] = {
        "real_images": len(real_paths),
        "fake_images": len(fake_paths),
        "max_images": max_images,
    }
    if not real_paths or not fake_paths:
        out["warning"] = "missing real or fake images; distribution metrics skipped"
        return out
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore
        from torchmetrics.image.kid import KernelInceptionDistance  # type: ignore
        from torchmetrics.image.inception import InceptionScore  # type: ignore
    except Exception as exc:
        out["warning"] = f"torchmetrics image dependencies unavailable: {exc}"
        return out

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = (load_image_batch(real_paths, image_size=image_size, max_images=max_images) * 255).to(torch.uint8).to(dev)
    fake = (load_image_batch(fake_paths, image_size=image_size, max_images=max_images) * 255).to(torch.uint8).to(dev)
    with torch.no_grad():
        fid = FrechetInceptionDistance(feature=2048).to(dev)
        fid.update(real, real=True)
        fid.update(fake, real=False)
        out["fid"] = float(fid.compute().item())

        subset = max(10, min(100, real.shape[0], fake.shape[0]))
        kid = KernelInceptionDistance(subset_size=subset).to(dev)
        kid.update(real, real=True)
        kid.update(fake, real=False)
        kid_mean, kid_std = kid.compute()
        out["kid_mean"] = float(kid_mean.item())
        out["kid_std"] = float(kid_std.item())

        inc = InceptionScore().to(dev)
        inc.update(fake)
        is_mean, is_std = inc.compute()
        out["inception_score_mean"] = float(is_mean.item())
        out["inception_score_std"] = float(is_std.item())
    return out
