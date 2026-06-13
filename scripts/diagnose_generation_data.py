"""
scripts.diagnose_generation_data
================================
Diagnose whether generation quality is limited by data, VAE reconstruction, or
sampling/training.  Intended for server preflight before expensive ImageNet-256
TraceFlow runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchvision.utils as vutils
import yaml
from torchvision.transforms.functional import to_pil_image

from src.utils.config_composer import apply_overrides


def _save_grid(images: torch.Tensor, path: Path, *, nrow: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(images.clamp(-1, 1), nrow=nrow, normalize=True, value_range=(-1, 1))
    to_pil_image(grid).save(str(path))


def _count_images(root: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _class_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def diagnose(args: argparse.Namespace) -> None:
    with open(args.config, encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.set_overrides)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved_config = out / "diagnosis_resolved_config.yml"
    with open(resolved_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device_str = cfg.get("project", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    from src.data.image_datasets import build_dataset
    data_cfg = cfg["data"]
    dataset = build_dataset(
        name=data_cfg["name"],
        root=data_cfg.get("root", "./data"),
        image_size=data_cfg["image_size"],
        download=data_cfg.get("download", False),
        smoke=False,
    )

    n = min(args.num_samples, len(dataset))
    images, labels, sample_paths = [], [], []
    for idx in range(n):
        sample = dataset[idx]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        label = sample[1] if isinstance(sample, (tuple, list)) and len(sample) > 1 else 0
        images.append(image)
        labels.append(int(label))
        paths = getattr(dataset, "samples", None)
        if isinstance(paths, list) and idx < len(paths):
            sample_paths.append(str(paths[idx][0]))
        else:
            sample_paths.append(None)

    x = torch.stack(images, dim=0).to(device)
    _save_grid(x.cpu(), out / "real_grid.png", nrow=max(1, int(n ** 0.5)))

    from src.models.autoencoder_backend import AutoencoderBackend
    ae_cfg = cfg["autoencoder"]
    ae = AutoencoderBackend(
        backend=ae_cfg.get("backend", "local"),
        pretrained_model_name_or_path=ae_cfg.get("pretrained_model_name_or_path"),
        latent_channels=ae_cfg["latent_channels"],
        image_size=cfg["data"]["image_size"],
        latent_size=ae_cfg["latent_size"],
        scaling_factor=ae_cfg.get("scaling_factor", 1.0),
        freeze=True,
        base_channels=ae_cfg.get("base_channels", 64),
    ).to(device)
    ae.eval()

    with torch.no_grad():
        z = ae.encode(x)
        recon = ae.decode(z)
    _save_grid(recon.cpu(), out / "vae_recon_grid.png", nrow=max(1, int(n ** 0.5)))
    comparison = torch.cat([x.cpu(), recon.cpu()], dim=0)
    _save_grid(comparison, out / "real_vs_vae_recon_grid.png", nrow=n)

    root = Path(str(data_cfg.get("root", "./data")))
    class_dirs = _class_dirs(root) if data_cfg.get("name") == "imagefolder" else []
    report: Dict[str, Any] = {
        "data": {
            "name": data_cfg.get("name"),
            "root": str(root),
            "image_size": data_cfg.get("image_size"),
            "dataset_len": len(dataset),
            "image_count_on_disk": _count_images(root) if root.exists() else 0,
            "class_count": len(class_dirs),
            "class_examples": [p.name for p in class_dirs[:10]],
            "labels": labels,
            "sample_paths": sample_paths,
        },
        "vae": {
            "backend": ae_cfg.get("backend"),
            "path": ae_cfg.get("pretrained_model_name_or_path"),
            "latent_shape": list(z.shape),
            "latent_mean": float(z.float().mean().item()),
            "latent_std": float(z.float().std().item()),
            "latent_min": float(z.float().min().item()),
            "latent_max": float(z.float().max().item()),
        },
        "outputs": {
            "real_grid": str(out / "real_grid.png"),
            "vae_recon_grid": str(out / "vae_recon_grid.png"),
            "real_vs_vae_recon_grid": str(out / "real_vs_vae_recon_grid.png"),
        },
    }

    if args.checkpoint:
        sample_root = out / "checkpoint_samples"
        sample_root.mkdir(parents=True, exist_ok=True)
        sample_runs = [
            ("euler50", "euler", 50),
            ("heun100", "heun", 100),
            ("heun250", "heun", 250),
        ]
        report["sampling"] = {}
        for name, sampler, steps in sample_runs:
            dest = sample_root / name
            cmd = [
                sys.executable, "-u", "-B", "-m", "scripts.sample_flow_transformer",
                "--config", str(resolved_config),
                "--checkpoint", args.checkpoint,
                "--sampler", sampler,
                "--steps", str(steps),
                "--num-samples", str(args.num_samples),
                "--output-dir", str(dest),
                "--seed", str(args.seed),
            ]
            print("[diagnose] $ " + " ".join(cmd), flush=True)
            rc = subprocess.run(cmd).returncode
            report["sampling"][name] = {
                "sampler": sampler,
                "steps": steps,
                "returncode": rc,
                "grid": str(dest / "sample_grid.png"),
            }

    with open(out / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# TraceFlow Data And Generation Diagnosis",
        "",
        f"- Dataset: `{report['data']['name']}`",
        f"- Root: `{report['data']['root']}`",
        f"- Dataset length: `{report['data']['dataset_len']}`",
        f"- Class count: `{report['data']['class_count']}`",
        f"- VAE latent std: `{report['vae']['latent_std']:.4f}`",
        "",
        "Inspect `real_grid.png` and `vae_recon_grid.png` first. If VAE reconstruction is already poor, generation quality is data/VAE-limited.",
    ]
    (out / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[diagnose] wrote {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose TraceFlow data, VAE reconstruction, and optional generator samples.")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--set", dest="set_overrides", action="append", default=[])
    return p.parse_args()


if __name__ == "__main__":
    diagnose(_parse_args())
