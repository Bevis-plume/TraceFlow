"""Train forensic TraceFlow detectors on raw no-key inversion artifacts.

This is detector-only fine-tuning: generator, autoencoder, key transform, and
decoder adapter are frozen. Clean images are trained toward chance for the owner
code; raw inversion artifacts are trained toward the fixed owner code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts._inversion_trace_common import (
    build_ae,
    build_transform,
    build_watermark,
    cifar_loader,
    clean_negative_loss,
    cycle,
    encode_trace_latent,
    image_loader,
    load_checkpoint,
    load_yaml,
    positive_owner_loss,
    bit_accuracy_from_logits,
    resolve_device,
    save_detector_checkpoint,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Train forensic inversion-trace detectors.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default=None, help='Optional full/keyed checkpoint to initialize detectors/AE metadata.')
    ap.add_argument('--positive-dir', required=True, help='Directory containing raw no-key inversion positive images.')
    ap.add_argument('--negative-image-dir', action='append', default=[], help='Extra clean negative image folders.')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--log-interval', type=int, default=50)
    ap.add_argument('--save-interval', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    cfg = load_yaml(args.config)
    state = load_checkpoint(args.checkpoint)
    image_size = int(cfg.get('data', {}).get('image_size', 32))
    wm_cfg = cfg.get('watermark', {})
    lambda_img = float(wm_cfg.get('lambda_trace_img', 1.0))
    lambda_lat = float(wm_cfg.get('lambda_trace_latent', 1.0))
    lambda_clean = float(wm_cfg.get('lambda_trace_clean_negative', 2.0))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ae = build_ae(cfg, state, device)
    transform = build_transform(cfg, state, ae, image_size, device)
    modules = build_watermark(cfg, state, image_size, device)
    image_detector = modules['extractor'].train()
    latent_detector = modules['latent_detector'].train()
    owner_bits = modules['bits'].to(device).float()

    opt = torch.optim.AdamW(list(image_detector.parameters()) + list(latent_detector.parameters()), lr=args.learning_rate)

    pos_iter = cycle(image_loader(args.positive_dir, image_size, args.batch_size, args.num_workers))
    neg_iter = cycle(cifar_loader(cfg, image_size, args.batch_size, args.num_workers))
    extra_neg_iters = [cycle(image_loader(p, image_size, args.batch_size, args.num_workers)) for p in args.negative_image_dir]

    log_path = output / 'train_log.jsonl'
    with log_path.open('w', encoding='utf-8') as log_f:
        for step in range(1, args.steps + 1):
            pos = next(pos_iter).to(device)
            clean = next(neg_iter).to(device)
            neg_batches = [('cifar_input', clean)]
            with torch.no_grad():
                neg_batches.append(('ae_recon', ae.decode(ae.encode(clean)).clamp(-1, 1)))
            for idx, it in enumerate(extra_neg_iters):
                neg_batches.append((f'negative_image_dir_{idx}', next(it).to(device)))

            opt.zero_grad(set_to_none=True)
            pos_img_logits = image_detector(pos)
            with torch.no_grad():
                pos_latent = encode_trace_latent(ae, transform, pos)
            pos_lat_logits = latent_detector(pos_latent)
            loss_pos_img = positive_owner_loss(pos_img_logits, owner_bits)
            loss_pos_lat = positive_owner_loss(pos_lat_logits, owner_bits)

            loss_neg_img = torch.zeros((), device=device)
            loss_neg_lat = torch.zeros((), device=device)
            neg_img_accs = []
            neg_lat_accs = []
            for _, neg in neg_batches:
                neg_img_logits = image_detector(neg)
                with torch.no_grad():
                    neg_latent = encode_trace_latent(ae, transform, neg)
                neg_lat_logits = latent_detector(neg_latent)
                loss_neg_img = loss_neg_img + clean_negative_loss(neg_img_logits)
                loss_neg_lat = loss_neg_lat + clean_negative_loss(neg_lat_logits)
                neg_img_accs.append(bit_accuracy_from_logits(neg_img_logits.detach(), owner_bits))
                neg_lat_accs.append(bit_accuracy_from_logits(neg_lat_logits.detach(), owner_bits))
            loss_neg_img = loss_neg_img / max(1, len(neg_batches))
            loss_neg_lat = loss_neg_lat / max(1, len(neg_batches))

            loss = lambda_img * loss_pos_img + lambda_lat * loss_pos_lat + lambda_clean * (loss_neg_img + loss_neg_lat)
            loss.backward()
            opt.step()

            if step % args.log_interval == 0 or step == 1:
                rec = {
                    'step': step,
                    'loss': float(loss.detach().cpu()),
                    'pos_img_loss': float(loss_pos_img.detach().cpu()),
                    'pos_lat_loss': float(loss_pos_lat.detach().cpu()),
                    'neg_img_loss': float(loss_neg_img.detach().cpu()),
                    'neg_lat_loss': float(loss_neg_lat.detach().cpu()),
                    'pos_img_bit_acc': bit_accuracy_from_logits(pos_img_logits.detach(), owner_bits),
                    'pos_lat_bit_acc': bit_accuracy_from_logits(pos_lat_logits.detach(), owner_bits),
                    'neg_img_owner_acc': sum(neg_img_accs) / len(neg_img_accs),
                    'neg_lat_owner_acc': sum(neg_lat_accs) / len(neg_lat_accs),
                }
                print('[trace-train] ' + ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in rec.items()))
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()

            if step % args.save_interval == 0 or step == args.steps:
                save_detector_checkpoint(output / f'step_{step:06d}.pt', cfg, state, modules, step)
                save_detector_checkpoint(output / 'latest.pt', cfg, state, modules, step)

    print(f'[trace-train] latest checkpoint: {output / "latest.pt"}')
    print(f'[trace-train] log: {log_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
