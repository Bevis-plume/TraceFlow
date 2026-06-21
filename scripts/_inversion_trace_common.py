
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.models.autoencoder_backend import AutoencoderBackend
from src.security.factory import build_latent_transform
from src.watermarking.factory import build_watermark_modules
from src.watermarking.forensic_trace import (
    binary_auroc,
    bit_accuracy_from_logits,
    clean_negative_loss,
    owner_match_scores,
    positive_owner_loss,
)

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}


class FlatImageDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int) -> None:
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise FileNotFoundError(f'No images found under {self.root}')
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.paths[index]).convert('RGB')
        return self.transform(img)


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            yield batch


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_checkpoint(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    dev = torch.device(device_str)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return dev


def build_ae(cfg: dict[str, Any], state: dict[str, Any], device: torch.device) -> AutoencoderBackend:
    data_cfg = cfg.get('data', {})
    ae_cfg = dict(state.get('ae_cfg') or cfg.get('autoencoder', {}))
    ae_cfg.setdefault('image_size', int(data_cfg.get('image_size', 32)))
    ae_cfg.setdefault('freeze', True)
    ae = AutoencoderBackend(**ae_cfg).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def build_transform(cfg: dict[str, Any], state: dict[str, Any], ae: AutoencoderBackend, image_size: int, device: torch.device) -> torch.nn.Module:
    c, h, _ = ae.latent_shape(image_size)
    security_cfg = dict(cfg.get('security', {}))
    meta = dict(state.get('transform_meta') or {})
    if meta.get('type') == 'keyed':
        lt = dict(security_cfg.get('latent_transform', {}))
        lt['type'] = 'keyed'
        for key in ('block_size', 'block_layout', 'bias_scale'):
            if key in meta:
                lt[key] = meta[key]
        security_cfg['latent_transform'] = lt
    transform = build_latent_transform(security_cfg, latent_channels=c, latent_size=h).to(device)
    transform.eval()
    for p in transform.parameters():
        p.requires_grad_(False)
    return transform


def build_watermark(cfg: dict[str, Any], state: dict[str, Any], image_size: int, device: torch.device):
    wm_cfg = dict(cfg.get('watermark', {}))
    wm_state = state.get('watermark') if isinstance(state, dict) else None
    arch_cfg = dict(wm_state.get('config') or wm_cfg) if isinstance(wm_state, dict) else wm_cfg
    # Keep the detector architecture from the checkpoint, but preserve new forensic objective knobs.
    for key, value in wm_cfg.items():
        if key.startswith('trace_') or key.startswith('lambda_trace') or key == 'objective':
            arch_cfg[key] = value
    modules = build_watermark_modules({'watermark': arch_cfg}, image_size=image_size, device=device)
    if isinstance(wm_state, dict):
        if wm_state.get('extractor') is not None:
            modules['extractor'].load_state_dict(wm_state['extractor'], strict=False)
        if wm_state.get('latent_detector') is not None:
            modules['latent_detector'].load_state_dict(wm_state['latent_detector'], strict=False)
        if wm_state.get('decoder_adapter') is not None:
            modules['decoder_adapter'].load_state_dict(wm_state['decoder_adapter'], strict=False)
    modules['decoder_adapter'].eval()
    for p in modules['decoder_adapter'].parameters():
        p.requires_grad_(False)
    return modules


def cifar_loader(cfg: dict[str, Any], image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    data_cfg = cfg.get('data', {})
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.CIFAR10(root=str(data_cfg.get('root', 'data')), train=True, download=bool(data_cfg.get('download', True)), transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=torch.cuda.is_available())


def image_loader(path: str | Path, image_size: int, batch_size: int, num_workers: int, shuffle: bool = True) -> DataLoader:
    ds = FlatImageDataset(path, image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle, pin_memory=torch.cuda.is_available())


def encode_trace_latent(ae: AutoencoderBackend, transform: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    z = ae.encode(images)
    return transform(z)


def save_detector_checkpoint(path: Path, cfg: dict[str, Any], state: dict[str, Any], modules: dict[str, Any], step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'step': step,
        'config': cfg,
        'ae_cfg': state.get('ae_cfg') or cfg.get('autoencoder', {}),
        'transform_meta': state.get('transform_meta') or cfg.get('security', {}).get('latent_transform', {}),
        'watermark': {
            'extractor': modules['extractor'].state_dict(),
            'latent_detector': modules['latent_detector'].state_dict(),
            'decoder_adapter': modules['decoder_adapter'].state_dict(),
            'bits': modules['bits'].detach().cpu(),
            'config': modules['config'],
            'objective': 'inversion_trace',
        },
    }
    torch.save(payload, path)
