"""
scripts/run_attack.py
=====================
Simulate a white-box gradient-inversion attack on the trained defence.

Pipeline
--------
1. Load a trained checkpoint (VAE + LatentPermuter + UNet + Watermarker).
2. Sample one image x from CIFAR-10 test set.
3. Run Trainer.get_target_gradients(x) to obtain "leaked" UNet gradients.
4. Run GradientInversionAttack.run() to reconstruct x from the gradients.
5. Save:
   • attack_image.png — the attacker's reconstructed (expected: visual noise)
   • target_image.png — the original reference image
   • z_prime_dummy.pt  — raw permuted latent (used by eval_traceability.py)

Usage
-----
    python -m scripts.run_attack --config configs/default.yml \\
        --checkpoint ./checkpoints/ckpt_best.pt \\
        --sample-idx 0 --output-dir ./attack_outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.inversion import AttackConfig, GradientInversionAttack
from src.crypto.latent_permute import LatentPermuter
from src.models.unet import UNet
from src.models.vae import VAE
from src.models.watermarker import Watermarker
from src.pipeline.trainer import Trainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility (identical to train_defense.py)
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
# Restore models from checkpoint
# ---------------------------------------------------------------------------

def restore_from_checkpoint(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[VAE, LatentPermuter, UNet, Watermarker]:
    """Load model weights from a saved checkpoint.

    Args:
        cfg:             Full config dict.
        checkpoint_path: Path to .pt checkpoint file.
        device:          Target device.

    Returns:
        Tuple (vae, permuter, unet, watermarker), all on device, in eval mode.
    """
    vc = cfg["vae"]
    pc = cfg["permuter"]
    uc = cfg["unet"]
    wc = cfg["watermarker"]

    latent_dim = vc["latent_channels"] * vc["latent_spatial"] ** 2

    vae = VAE(vc["in_channels"], vc["latent_channels"], vc["base_channels"], vc["kl_weight"])
    permuter = LatentPermuter(pc["secret_key"], latent_dim, pc["bias_scale"])
    unet = UNet(
        vc["latent_channels"], uc["model_channels"], uc["channel_mult"],
        uc["num_res_blocks"], uc["time_embed_dim"], uc["dropout"],
    )
    watermarker = Watermarker(wc["input_dim"], wc["hidden_dims"], wc["output_dim"], wc["dropout"])

    ckpt = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(ckpt["vae_state"])
    unet.load_state_dict(ckpt["unet_state"])
    watermarker.load_state_dict(ckpt["watermarker_state"])
    permuter.load_state_dict(ckpt["permuter_state"])

    for m in [vae, permuter, unet, watermarker]:
        m.to(device).eval()

    logger.info("Models loaded from %s", checkpoint_path)
    return vae, permuter, unet, watermarker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run gradient-inversion attack")
    parser.add_argument("--config",     type=str, default="configs/default.yml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Index of the CIFAR-10 test image to attack")
    parser.add_argument("--output-dir", type=str, default="./attack_outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])

    device = torch.device(cfg["project"]["device"])
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────
    vae, permuter, unet, watermarker = restore_from_checkpoint(
        cfg, args.checkpoint, device
    )

    # ── Get one test sample ────────────────────────────────────────────
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        root=cfg["data"]["root"], train=False, download=True, transform=transform
    )
    x, label = test_set[args.sample_idx]
    x = x.unsqueeze(0).to(device)   # (1, 3, 32, 32)
    logger.info("Attacking sample index=%d  label=%d", args.sample_idx, label)

    # ── Build a minimal Trainer just to call get_target_gradients ──────
    t_cfg = cfg["training"]
    d_cfg = cfg["diffusion"]
    trainer_config = TrainerConfig(
        device=str(device),
        diffusion_timesteps=d_cfg["timesteps"],
        beta_start=d_cfg["beta_start"],
        beta_end=d_cfg["beta_end"],
        lambda_wm=t_cfg["lambda_wm"],
        wm_bits=cfg["watermarker"]["output_dim"],
        wm_seed=cfg["project"]["seed"],
    )
    trainer = Trainer(vae, permuter, unet, watermarker, trainer_config)

    logger.info("Computing target gradients (simulating gradient leakage)…")
    target_grads = trainer.get_target_gradients(x)   # list of CPU tensors

    # ── Run attack ─────────────────────────────────────────────────────
    a_cfg = cfg["attack"]
    attack_config = AttackConfig(
        optimizer=a_cfg["optimizer"],
        max_iter=a_cfg["max_iter"],
        lr=a_cfg["lr"],
        dummy_init=a_cfg["dummy_init"],
        diffusion_timesteps=d_cfg["timesteps"],
        beta_start=d_cfg["beta_start"],
        beta_end=d_cfg["beta_end"],
        lambda_wm=t_cfg["lambda_wm"],
        wm_bits=cfg["watermarker"]["output_dim"],
        verbose=50,
    )

    attacker = GradientInversionAttack(
        vae, permuter, unet, watermarker, attack_config, device=str(device)
    )

    logger.info("Starting gradient inversion (optimizer=%s, max_iter=%d)…",
                a_cfg["optimizer"], a_cfg["max_iter"])
    results = attacker.run(target_grads, image_shape=x.shape)

    # ── Save outputs ───────────────────────────────────────────────────
    # Attacker's reconstructed image (expected: semantically incoherent)
    attack_img_path = os.path.join(args.output_dir, "attack_image.png")
    vutils.save_image(results["x_reconstructed"], attack_img_path)
    logger.info("Attack image saved → %s", attack_img_path)

    # Original target image (for reference / SSIM comparison)
    # Denormalise from [−1,1] → [0,1]
    x_display = (x.cpu() * 0.5 + 0.5).clamp(0, 1)
    target_img_path = os.path.join(args.output_dir, "target_image.png")
    vutils.save_image(x_display, target_img_path)
    logger.info("Target image saved  → %s", target_img_path)

    # Permuted latent (for traceability evaluation)
    latent_path = os.path.join(args.output_dir, "z_prime_dummy.pt")
    torch.save(results["z_prime_dummy"], latent_path)
    logger.info("Permuted latent saved → %s", latent_path)

    logger.info("Attack complete.  Final optimisation loss = %.4f",
                results["final_loss"].item())


if __name__ == "__main__":
    main()
