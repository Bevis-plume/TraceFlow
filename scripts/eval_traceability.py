"""
scripts/eval_traceability.py
============================
Evaluate the defence's traceability claim on attacker-recovered images.

Core forensic logic
-------------------
The attacker produced:
    x_attack  ← VAE.decode( z'_dummy )          (visually incoherent)
    z'_dummy  ≈ z'  (permuted latent, key K unknown to attacker)

The DEFENDER now takes x_attack and applies the full defence pipeline
with the correct secret key K to prove copyright ownership:

    Step 1.  Re-encode:   z̃ = VAE.encode(x_attack).sample
    Step 2.  Re-permute:  z̃' = π_K(z̃) + β_K
             [Because x_attack was decoded from z'_dummy ≈ z', re-encoding
              gives z̃ ≈ z_dummy (unpermuted view), and re-permuting yields
              z̃' ≈ z'_dummy — the space where the Watermarker was trained.]
    Step 3.  Detect:      ŵ = Watermarker(z̃')
    Step 4.  Measure:     bit_accuracy(ŵ, w*)

Alternatively, the defender can use the saved z'_dummy directly
(if obtained through forensic analysis) as input to the Watermarker.

Outputs
-------
Console + results/eval_report.json:
    • SSIM between original and attacker image  (should be very low)
    • PSNR between original and attacker image  (should be very low)
    • Watermark bit accuracy on attacked image  (should be > 0.9)
    • Watermark bit accuracy on original image  (sanity check, ~1.0)

Usage
-----
    python -m scripts.eval_traceability \\
        --config configs/default.yml \\
        --checkpoint ./checkpoints/ckpt_best.pt \\
        --attack-dir ./attack_outputs \\
        --results-dir ./results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crypto.latent_permute import LatentPermuter
from src.models.unet import UNet
from src.models.vae import VAE
from src.models.watermarker import Watermarker, generate_random_watermark
from src.utils.image import to_vae_input
from src.utils.metrics import evaluate_reconstruction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Fix all random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Model restore (same helper as run_attack.py)
# ---------------------------------------------------------------------------

def restore_models(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[VAE, LatentPermuter, Watermarker, torch.Tensor]:
    """Load VAE, LatentPermuter, and Watermarker from checkpoint.

    Also restores the saved target watermark w* from the checkpoint so that
    the evaluation uses exactly the same bit-string as training.

    Args:
        cfg:             Full config dict.
        checkpoint_path: Path to .pt checkpoint.
        device:          Target device.

    Returns:
        Tuple (vae, permuter, watermarker, w_star) all on device.
    """
    vc = cfg["vae"]
    pc = cfg["permuter"]
    wc = cfg["watermarker"]
    latent_dim = vc["latent_channels"] * vc["latent_spatial"] ** 2

    vae = VAE(vc["in_channels"], vc["latent_channels"], vc["base_channels"], vc["kl_weight"])
    permuter = LatentPermuter(
        secret_key=pc["secret_key"],
        latent_dim=latent_dim,
        block_size=pc.get("block_size", 16),
        bias_scale=pc["bias_scale"],
    )
    watermarker = Watermarker(
        input_dim=wc["input_dim"],
        hidden_dims=wc["hidden_dims"],
        output_dim=wc["output_dim"],
        block_size=wc.get("block_size", pc.get("block_size", 16)),
        dropout=wc["dropout"],
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(ckpt["vae_state"])
    permuter.load_state_dict(ckpt.get("permuter_state", {}), strict=False)
    watermarker.load_state_dict(ckpt["watermarker_state"])

    # Restore the exact w* used during training
    if "w_star" in ckpt:
        w_star = ckpt["w_star"].to(device)
    else:
        # Fall back to regenerating it from seed (deterministic)
        w_star = generate_random_watermark(
            wc["output_dim"], seed=cfg["project"]["seed"]
        ).to(device)

    for m in [vae, permuter, watermarker]:
        m.to(device).eval()

    logger.info("Defence models loaded from %s", checkpoint_path)
    return vae, permuter, watermarker, w_star


# ---------------------------------------------------------------------------
# Forensic watermark extraction pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_watermark_from_image(
    image: torch.Tensor,        # (1, C, H, W) in [0,1]
    vae: VAE,
    permuter: LatentPermuter,
    watermarker: Watermarker,
    device: torch.device,
) -> torch.Tensor:
    """Defender's forensic pipeline: image → z → z' → ŵ.

    This is the core traceability logic.  Even if `image` is attacker-
    produced visual noise, as long as it was decoded from a permuted latent,
    re-encoding and re-permuting with the correct key K recovers z' ≈ z'_dummy
    which the Watermarker maps to the embedded bit-stream w*.

    Pipeline:
        image  →  VAE.encode  →  z  →  LatentPermuter.forward  →  z'
               →  Watermarker  →  ŵ ∈ (0,1)^M

    Args:
        image:      Input image tensor (1, C_x, H, W) in [0, 1].
        vae:        Trained VAE (defence side, with correct weights).
        permuter:   LatentPermuter with correct secret key K.
        watermarker: Trained Watermarker.
        device:     Compute device.

    Returns:
        w_hat: Predicted bit probabilities (1, M).
    """
    image = to_vae_input(image).to(device)

    # Step 1: encode
    mu, logvar = vae.encode(image)
    z = vae.reparameterise(mu, logvar)           # (1, C_z, H_z, W_z)

    # Step 2: re-permute with correct key K
    # This maps attacker's z ≈ π_K^{-1}(z'_dummy) back to z' ≈ z'_dummy
    z_prime = permuter(z)                         # (1, C_z, H_z, W_z)

    # Step 3: detect
    w_hat = watermarker(z_prime)                  # (1, M)
    return w_hat


@torch.no_grad()
def extract_watermark_from_latent(
    z_prime: torch.Tensor,      # (1, C_z, H_z, W_z)
    watermarker: Watermarker,
    device: torch.device,
) -> torch.Tensor:
    """Shortcut: directly feed the saved z'_dummy to Watermarker.

    This path is used when the defender has forensic access to the recovered
    permuted latent (saved by run_attack.py).

    Args:
        z_prime:     Permuted latent (1, C_z, H_z, W_z).
        watermarker: Trained Watermarker.
        device:      Compute device.

    Returns:
        w_hat: Predicted bit probabilities (1, M).
    """
    return watermarker(z_prime.to(device))


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_png_as_tensor(path: str) -> torch.Tensor:
    """Load a PNG/JPEG image file as a float tensor (1, C, H, W) in [0, 1].

    Args:
        path: Filesystem path to the image file.

    Returns:
        Float tensor (1, C, H, W) in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    t = T.ToTensor()(img)       # (C, H, W) in [0,1]
    return t.unsqueeze(0)       # (1, C, H, W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate defence traceability")
    parser.add_argument("--config",      type=str, default="configs/default.yml")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--attack-dir",  type=str, default="./attack_outputs",
                        help="Directory containing attack_image.png, "
                             "target_image.png, z_prime_dummy.pt")
    parser.add_argument("--results-dir", type=str, default="./results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])

    device = torch.device(cfg["project"]["device"])
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Restore defence models ─────────────────────────────────────────
    vae, permuter, watermarker, w_star = restore_models(
        cfg, args.checkpoint, device
    )

    # ── Load artefacts produced by run_attack.py ───────────────────────
    attack_path  = os.path.join(args.attack_dir, "attack_image.png")
    target_path  = os.path.join(args.attack_dir, "target_image.png")
    latent_path  = os.path.join(args.attack_dir, "z_prime_dummy.pt")

    for p in [attack_path, target_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")

    x_attack  = load_png_as_tensor(attack_path)    # attacker's noise image
    x_target  = load_png_as_tensor(target_path)    # original ground-truth

    # ── Metric 1: image quality (should be very low / noise-like) ─────
    logger.info("Computing image quality metrics (SSIM / PSNR)…")
    # Both tensors are in [0,1]; evaluate on CPU to avoid device mismatches
    psnr_val = torch.tensor(0.0)
    ssim_val = torch.tensor(0.0)
    from src.utils.metrics import psnr as psnr_fn, ssim as ssim_fn
    psnr_val = psnr_fn(x_target, x_attack).item()
    ssim_val = ssim_fn(x_target, x_attack).item()
    logger.info("  PSNR  = %.2f dB  (lower confirms visual incoherence)", psnr_val)
    logger.info("  SSIM  = %.4f     (lower confirms visual incoherence)", ssim_val)

    # ── Metric 2: watermark extraction via image re-encoding ──────────
    logger.info("Extracting watermark from attack image via re-encode pipeline…")
    w_hat_image = extract_watermark_from_image(
        x_attack, vae, permuter, watermarker, device
    )                                                            # (1, M)
    bit_acc_image = (
        ((w_hat_image >= 0.5).float() == w_star.unsqueeze(0)).float().mean().item()
    )
    logger.info(
        "  Bit accuracy (from image re-encode) = %.4f  "
        "(random baseline = 0.50, defence claim ≥ 0.90)",
        bit_acc_image,
    )

    # ── Metric 3: watermark extraction via saved latent (direct path) ─
    bit_acc_latent: float | None = None
    if os.path.exists(latent_path):
        logger.info("Extracting watermark from saved permuted latent (direct)…")
        z_prime_dummy = torch.load(latent_path, map_location=device)
        w_hat_latent = extract_watermark_from_latent(
            z_prime_dummy, watermarker, device
        )
        bit_acc_latent = (
            ((w_hat_latent >= 0.5).float() == w_star.unsqueeze(0)).float().mean().item()
        )
        logger.info(
            "  Bit accuracy (from direct latent)   = %.4f  "
            "(should equal or exceed image-path accuracy)",
            bit_acc_latent,
        )
    else:
        logger.warning("z_prime_dummy.pt not found; skipping direct-latent eval.")

    # ── Metric 4: sanity check on original target image ───────────────
    logger.info("Sanity check: watermark extraction on the original target image…")
    w_hat_orig = extract_watermark_from_image(
        x_target,
        vae,
        permuter,
        watermarker,
        device,
    )
    bit_acc_orig = (
        ((w_hat_orig >= 0.5).float() == w_star.unsqueeze(0)).float().mean().item()
    )
    logger.info("  Bit accuracy (original image)        = %.4f", bit_acc_orig)

    # ── Compile report ─────────────────────────────────────────────────
    report = {
        "image_quality": {
            "psnr_db": round(psnr_val, 4),
            "ssim":    round(ssim_val, 6),
            "interpretation": (
                "Attack image should have low PSNR/SSIM vs original, "
                "confirming semantically incoherent reconstruction."
            ),
        },
        "watermark_traceability": {
            "bit_acc_from_image_reencode": round(bit_acc_image, 6),
            "bit_acc_from_direct_latent":  (
                round(bit_acc_latent, 6) if bit_acc_latent is not None else None
            ),
            "bit_acc_original_image":      round(bit_acc_orig, 6),
            "random_baseline":             0.5,
            "defence_claim_threshold":     0.9,
            "traceability_passed": bit_acc_image >= 0.9,
        },
        "watermark_bits": int(w_star.numel()),
        "w_hat_sample_image": w_hat_image[0].tolist(),
        "w_star_sample":      w_star.tolist(),
    }

    report_path = os.path.join(args.results_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Full evaluation report saved → %s", report_path)

    # ── Final verdict ──────────────────────────────────────────────────
    passed = report["watermark_traceability"]["traceability_passed"]
    verdict = "PASSED ✓" if passed else "FAILED ✗"
    logger.info(
        "\n%s\n"
        "  PSNR=%.2f dB  SSIM=%.4f\n"
        "  Watermark bit-acc (re-encode)=%.4f  (direct latent)=%s\n"
        "  Traceability: %s",
        "=" * 60,
        psnr_val, ssim_val,
        bit_acc_image,
        f"{bit_acc_latent:.4f}" if bit_acc_latent is not None else "N/A",
        verdict,
    )


if __name__ == "__main__":
    main()
