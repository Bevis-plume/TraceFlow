"""
src/utils/metrics.py
====================
Image quality and watermark evaluation metrics.

Metrics provided
----------------
PSNR (Peak Signal-to-Noise Ratio):
    PSNR = 10 · log10( MAX² / MSE )                                  (1)
    Higher is better; ∞ for identical images.
    Typical threshold for "good" reconstruction: ≥ 30 dB.

SSIM (Structural Similarity Index Measure, Wang et al., 2004):
    Combines luminance (l), contrast (c), and structure (s) comparisons:
        SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ          (2)
    Range [−1, 1]; 1 = identical; < 0.3 → severe distortion.

Bit Accuracy:
    acc = (1/M) Σ_j 𝟙[round(ŵ_j) == w*_j]                           (3)
    Random baseline = 0.5 for a uniform 50/50 watermark.

Bit Error Rate (BER):
    BER = 1 − acc = (1/M) Σ_j 𝟙[round(ŵ_j) ≠ w*_j]                 (4)
    Lower is better; 0.0 = perfect recovery; 0.5 = random (no signal).
    Traceability claim: acc > 0.9 on attacker-recovered images.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Compute mean Peak Signal-to-Noise Ratio for a batch.

    Args:
        img1:    Reference image tensor (B, C, H, W) in [0, max_val].
        img2:    Distorted image tensor (B, C, H, W) in [0, max_val].
        max_val: Maximum possible pixel value (1.0 for normalised images).

    Returns:
        Scalar mean PSNR in dB over the batch.
        Returns +∞ if MSE == 0.
    """
    mse = F.mse_loss(img1, img2, reduction="none").mean(dim=[1, 2, 3])   # (B,)
    # Guard against log(0): identical images → +inf PSNR
    psnr_vals = torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(max_val ** 2 / mse),
    )
    return psnr_vals.mean()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Build a normalised 2-D Gaussian convolution kernel.

    Args:
        window_size: Kernel spatial size (e.g. 11).
        sigma:       Gaussian standard deviation.
        channels:    Number of image channels (kernel is replicated per-channel).

    Returns:
        Kernel tensor of shape (channels, 1, window_size, window_size).
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d.outer(g1d)                                 # (W, W)
    kernel = g2d.unsqueeze(0).unsqueeze(0)               # (1, 1, W, W)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute mean SSIM for a batch using sliding Gaussian windows.

    Implements the full SSIM formula from Wang et al. (2004):
        SSIM(x,y) = (2μ_xμ_y + C1)(2σ_xy + C2)
                    ─────────────────────────────
                    (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

    Args:
        img1:        Reference images (B, C, H, W) in [0, 1].
        img2:        Distorted images (B, C, H, W) in [0, 1].
        window_size: Gaussian kernel spatial size (default 11).
        sigma:       Gaussian standard deviation (default 1.5).
        C1, C2:      Stability constants (default (0.01)², (0.03)²).

    Returns:
        Scalar mean SSIM in [−1, 1] averaged over batch and channels.
    """
    B, C, H, W = img1.shape
    kernel = _gaussian_kernel(window_size, sigma, C).to(img1.device)
    pad = window_size // 2

    def _mu(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1 = _mu(img1)
    mu2 = _mu(img2)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _mu(img1 * img1) - mu1_sq
    sigma2_sq = _mu(img2 * img2) - mu2_sq
    sigma12   = _mu(img1 * img2) - mu1_mu2

    numerator   = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator                   # (B, C, H, W)
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Watermark bit accuracy and BER
# ---------------------------------------------------------------------------

def bit_accuracy(
    w_hat: torch.Tensor,
    w_target: torch.Tensor,
) -> torch.Tensor:
    """Fraction of correctly decoded bits (threshold at 0.5).

    acc = (1/M) Σ_j 𝟙[round(ŵ_j) == w*_j]                           (3)

    Args:
        w_hat:    Predicted bit probabilities (B, M) or (M,).
        w_target: Ground-truth float bits {0., 1.} — shape (M,) or (B, M).

    Returns:
        Scalar accuracy in [0, 1].  Random baseline = 0.5.
    """
    predicted = (w_hat >= 0.5).float()
    target    = w_target.expand_as(predicted)
    return (predicted == target).float().mean()


def bit_error_rate(
    w_hat: torch.Tensor,
    w_target: torch.Tensor,
) -> torch.Tensor:
    """Bit Error Rate: fraction of incorrectly decoded bits.

    BER = 1 − acc = (1/M) Σ_j 𝟙[round(ŵ_j) ≠ w*_j]                 (4)

    A BER near 0.5 indicates the watermark signal is absent or destroyed
    (random chance).  A BER < 0.1 confirms successful extraction.

    Args:
        w_hat:    Predicted bit probabilities (B, M) or (M,).
        w_target: Ground-truth float bits {0., 1.} — shape (M,) or (B, M).

    Returns:
        Scalar BER in [0, 1].  Lower is better.
    """
    return 1.0 - bit_accuracy(w_hat, w_target)


def ssim_skimage(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> float:
    """SSIM via scikit-image (reference implementation, CPU-only).

    Useful as a cross-check against the native PyTorch `ssim()` above.
    Falls back to the pure-PyTorch implementation if skimage is unavailable.

    Args:
        img1: Reference image (B, C, H, W) or (C, H, W) in [0, 1].
        img2: Distorted image — same shape.

    Returns:
        Mean SSIM as a Python float.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        import numpy as np

        # Take first image in batch if batched
        a = img1[0].cpu().numpy() if img1.dim() == 4 else img1.cpu().numpy()
        b = img2[0].cpu().numpy() if img2.dim() == 4 else img2.cpu().numpy()
        # skimage expects (H, W, C) for multichannel
        a = np.transpose(a, (1, 2, 0))
        b = np.transpose(b, (1, 2, 0))
        return float(sk_ssim(a, b, data_range=1.0, channel_axis=-1))
    except ImportError:
        # Graceful fallback to native PyTorch SSIM
        return ssim(
            img1.unsqueeze(0) if img1.dim() == 3 else img1,
            img2.unsqueeze(0) if img2.dim() == 3 else img2,
        ).item()


# ---------------------------------------------------------------------------
# Convenience: evaluate all metrics at once
# ---------------------------------------------------------------------------

def evaluate_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    w_hat: torch.Tensor,
    w_target: torch.Tensor,
) -> dict[str, float]:
    """Compute PSNR, SSIM, bit-accuracy, and BER in one call.

    Args:
        original:      Ground-truth images (B, C, H, W).
        reconstructed: Reconstructed / attacked images (B, C, H, W).
        w_hat:         Predicted watermark bits (B, M).
        w_target:      Target watermark bits (M,).

    Returns:
        Dict with keys "psnr_db", "ssim", "bit_acc", "ber".
    """
    with torch.no_grad():
        psnr_val = psnr(original, reconstructed).item()
        ssim_val = ssim(original, reconstructed).item()
        acc      = bit_accuracy(w_hat, w_target).item()
        ber      = 1.0 - acc
    return {
        "psnr_db": psnr_val,
        "ssim":    ssim_val,
        "bit_acc": acc,
        "ber":     ber,
    }
