"""Evaluate forensic TraceFlow detectors.

Headline metrics are raw inversion traceability and clean false positives.
Post-watermark/re-stamp metrics are intentionally not used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts._inversion_trace_common import (
    binary_auroc,
    bit_accuracy_from_logits,
    build_ae,
    build_transform,
    build_watermark,
    cifar_loader,
    cycle,
    encode_trace_latent,
    image_loader,
    load_checkpoint,
    load_yaml,
    owner_match_scores,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Evaluate forensic inversion-trace detectors.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True, help='Detector checkpoint from train_inversion_trace_detector.')
    ap.add_argument('--positive-dir', required=True)
    ap.add_argument('--negative-image-dir', action='append', default=[])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-batches', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=2)
    return ap.parse_args()


def _eval_loader(name, loader, modules, ae, transform, owner_bits, device, num_batches, *, ae_recon=False):
    image_detector = modules['extractor'].eval()
    latent_detector = modules['latent_detector'].eval()
    img_accs, lat_accs = [], []
    img_scores, lat_scores = [], []
    it = cycle(loader)
    with torch.no_grad():
        for _ in range(num_batches):
            x = next(it).to(device)
            if ae_recon:
                x = ae.decode(ae.encode(x)).clamp(-1, 1)
            img_logits = image_detector(x)
            z = encode_trace_latent(ae, transform, x)
            lat_logits = latent_detector(z)
            img_accs.append(bit_accuracy_from_logits(img_logits, owner_bits))
            lat_accs.append(bit_accuracy_from_logits(lat_logits, owner_bits))
            img_scores.extend(owner_match_scores(img_logits, owner_bits).detach().cpu().tolist())
            lat_scores.extend(owner_match_scores(lat_logits, owner_bits).detach().cpu().tolist())
    return {
        'name': name,
        'image_owner_bit_acc': float(sum(img_accs) / len(img_accs)),
        'latent_owner_bit_acc': float(sum(lat_accs) / len(lat_accs)),
        'image_scores': img_scores,
        'latent_scores': lat_scores,
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    cfg = load_yaml(args.config)
    state = load_checkpoint(args.checkpoint)
    image_size = int(cfg.get('data', {}).get('image_size', 32))

    ae = build_ae(cfg, state, device)
    transform = build_transform(cfg, state, ae, image_size, device)
    modules = build_watermark(cfg, state, image_size, device)
    owner_bits = modules['bits'].to(device).float()

    positives = image_loader(args.positive_dir, image_size, args.batch_size, args.num_workers, shuffle=True)
    cifar = cifar_loader(cfg, image_size, args.batch_size, args.num_workers)

    pos = _eval_loader('raw_no_key_inversion', positives, modules, ae, transform, owner_bits, device, args.num_batches)
    clean_cifar = _eval_loader('clean_cifar_input', cifar, modules, ae, transform, owner_bits, device, args.num_batches)
    clean_ae = _eval_loader('clean_ae_recon', cifar, modules, ae, transform, owner_bits, device, args.num_batches, ae_recon=True)

    negatives = [clean_cifar, clean_ae]
    for idx, path in enumerate(args.negative_image_dir):
        loader = image_loader(path, image_size, args.batch_size, args.num_workers, shuffle=True)
        negatives.append(_eval_loader(f'clean_image_dir_{idx}', loader, modules, ae, transform, owner_bits, device, args.num_batches))

    neg_img_scores = []
    neg_lat_scores = []
    for item in negatives:
        neg_img_scores.extend(item['image_scores'])
        neg_lat_scores.extend(item['latent_scores'])

    metrics = {
        'raw_inversion_image_bit_acc': pos['image_owner_bit_acc'],
        'raw_inversion_latent_bit_acc': pos['latent_owner_bit_acc'],
        'clean_cifar_image_fp': clean_cifar['image_owner_bit_acc'],
        'clean_cifar_latent_fp': clean_cifar['latent_owner_bit_acc'],
        'clean_ae_recon_image_fp': clean_ae['image_owner_bit_acc'],
        'clean_ae_recon_latent_fp': clean_ae['latent_owner_bit_acc'],
        'trace_auroc_image': binary_auroc(pos['image_scores'], neg_img_scores),
        'trace_auroc_latent': binary_auroc(pos['latent_scores'], neg_lat_scores),
        'positive': {k: v for k, v in pos.items() if not k.endswith('_scores')},
        'negatives': [{k: v for k, v in item.items() if not k.endswith('_scores')} for item in negatives],
        'num_positive_scores': len(pos['image_scores']),
        'num_negative_scores': len(neg_img_scores),
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'inversion_trace_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print(f'[trace-eval] wrote {out / "inversion_trace_metrics.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
