"""
scripts/train_defense.py
========================
Entry-point: train the joint defence (VAE + UNet + Watermarker).

Usage
-----
    python -m scripts.train_defense --config configs/default.yml [options]

    # Resume from checkpoint:
    python -m scripts.train_defense --config configs/default.yml \\
        --resume ./checkpoints/ckpt_best.pt
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
import yaml
from torch.utils.data import DataLoader

# Ensure project root is importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Fix all random seeds for full reproducibility.

    Fixes: Python random, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and return a YAML config file as a nested dict.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataloader(cfg: dict) -> DataLoader:
    """Build a CIFAR-10 training DataLoader from config.

    Args:
        cfg: Full config dict (uses cfg["data"] and cfg["training"] keys).

    Returns:
        DataLoader yielding (image_tensor, label) pairs; images in [0, 1].
    """
    transform = T.Compose([
        T.ToTensor(),                        # [0,1], (C,H,W)
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # → [−1,1]
    ])
    # Note: VAE uses Sigmoid output → we normalise here for training stability
    # but keep images semantically in a known range.

    dataset = torchvision.datasets.CIFAR10(
        root=cfg["data"]["root"],
        train=True,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
    )
    logger.info("CIFAR-10 train set: %d samples, batch_size=%d",
                len(dataset), cfg["training"]["batch_size"])
    return loader


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(cfg: dict) -> tuple[VAE, LatentPermuter, UNet, Watermarker]:
    """Instantiate all four model components from config.

    Args:
        cfg: Full config dict.

    Returns:
        Tuple (vae, permuter, unet, watermarker).
    """
    vc = cfg["vae"]
    vae = VAE(
        in_channels=vc["in_channels"],
        latent_channels=vc["latent_channels"],
        base_channels=vc["base_channels"],
        kl_weight=vc["kl_weight"],
    )

    pc = cfg["permuter"]
    latent_dim = (
        vc["latent_channels"]
        * vc["latent_spatial"]
        * vc["latent_spatial"]
    )
    permuter = LatentPermuter(
        secret_key=pc["secret_key"],
        latent_dim=latent_dim,
        bias_scale=pc["bias_scale"],
    )

    uc = cfg["unet"]
    unet = UNet(
        in_channels=vc["latent_channels"],
        model_channels=uc["model_channels"],
        channel_mult=uc["channel_mult"],
        num_res_blocks=uc["num_res_blocks"],
        time_embed_dim=uc["time_embed_dim"],
        dropout=uc["dropout"],
    )

    wc = cfg["watermarker"]
    watermarker = Watermarker(
        input_dim=wc["input_dim"],
        hidden_dims=wc["hidden_dims"],
        output_dim=wc["output_dim"],
        dropout=wc["dropout"],
    )

    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [vae, unet, watermarker]
    )
    logger.info("Trainable parameters (VAE+UNet+WM): %d", total_params)
    return vae, permuter, unet, watermarker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Trackable Latent Diffusion defence")
    parser.add_argument("--config", type=str, default="configs/default.yml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed_everything(cfg["project"]["seed"])

    train_loader = build_dataloader(cfg)
    vae, permuter, unet, watermarker = build_models(cfg)

    t_cfg_raw = cfg["training"]
    d_cfg_raw = cfg["diffusion"]
    trainer_config = TrainerConfig(
        device=cfg["project"]["device"],
        num_epochs=t_cfg_raw["num_epochs"],
        batch_size=t_cfg_raw["batch_size"],
        learning_rate=t_cfg_raw["learning_rate"],
        weight_decay=t_cfg_raw["weight_decay"],
        grad_clip_norm=t_cfg_raw["grad_clip_norm"],
        lambda_wm=t_cfg_raw["lambda_wm"],
        log_interval=t_cfg_raw["log_interval"],
        save_interval=t_cfg_raw["save_interval"],
        checkpoint_dir=t_cfg_raw["checkpoint_dir"],
        diffusion_timesteps=d_cfg_raw["timesteps"],
        beta_start=d_cfg_raw["beta_start"],
        beta_end=d_cfg_raw["beta_end"],
        wm_bits=cfg["watermarker"]["output_dim"],
        wm_seed=cfg["project"]["seed"],
    )

    trainer = Trainer(vae, permuter, unet, watermarker, trainer_config)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    trainer.fit(train_loader, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
