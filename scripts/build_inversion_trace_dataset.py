"""Build a small forensic-trace dataset from no-key inversion outputs.

The output layout is:
    output_dir/positives/*.png
    output_dir/manifest.json

Positive images should be raw no-key inversion artifacts, not defender re-stamped
images. Grids from eval_traceflow_inversion can be split automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}


def _iter_images(paths: Iterable[Path], pattern: str) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            out.append(path)
        elif path.is_dir():
            out.extend([p for p in path.rglob(pattern) if p.suffix.lower() in IMAGE_EXTS])
    return sorted(set(out))


def _segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start = None
    for i, keep in enumerate(mask.tolist() + [False]):
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    return segs


def split_grid(img: Image.Image, tile_size: int) -> list[Image.Image]:
    """Split torchvision-style grids using near-black separator rows/columns."""
    rgb = img.convert('RGB')
    arr = np.asarray(rgb).astype(np.float32)
    row_black = (arr.mean(axis=(1, 2)) < 8.0) & (arr.std(axis=(1, 2)) < 8.0)
    col_black = (arr.mean(axis=(0, 2)) < 8.0) & (arr.std(axis=(0, 2)) < 8.0)
    row_segs = _segments(~row_black, max(8, tile_size // 2))
    col_segs = _segments(~col_black, max(8, tile_size // 2))

    crops: list[Image.Image] = []
    for y0, y1 in row_segs:
        for x0, x1 in col_segs:
            crop = rgb.crop((x0, y0, x1, y1))
            if crop.width < 8 or crop.height < 8:
                continue
            c_arr = np.asarray(crop).astype(np.float32)
            if c_arr.std() < 3.0:
                continue
            crops.append(crop.resize((tile_size, tile_size), Image.Resampling.BICUBIC))
    return crops


def main() -> int:
    ap = argparse.ArgumentParser(description='Build forensic trace positive dataset from inversion outputs.')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--inversion-run-dir', action='append', default=[])
    ap.add_argument('--positive-dir', action='append', default=[])
    ap.add_argument('--pattern', default='*raw_nokey*.png')
    ap.add_argument('--tile-size', type=int, default=32)
    ap.add_argument('--split-grids', action='store_true')
    ap.add_argument('--max-images', type=int, default=0)
    args = ap.parse_args()

    output = Path(args.output_dir)
    pos_out = output / 'positives'
    pos_out.mkdir(parents=True, exist_ok=True)

    sources = [Path(p) for p in args.positive_dir]
    for run in args.inversion_run_dir:
        sources.append(Path(run) / 'images')
    images = _iter_images(sources, args.pattern)
    if not images:
        raise SystemExit(f'No positive images found from {sources} with pattern {args.pattern!r}')

    written = 0
    manifest = {'sources': [str(p) for p in images], 'positives': []}
    for src in images:
        img = Image.open(src).convert('RGB')
        crops = split_grid(img, args.tile_size) if args.split_grids else [img.resize((args.tile_size, args.tile_size), Image.Resampling.BICUBIC)]
        if not crops:
            crops = [img.resize((args.tile_size, args.tile_size), Image.Resampling.BICUBIC)]
        for idx, crop in enumerate(crops):
            if args.max_images and written >= args.max_images:
                break
            name = f'pos_{written:06d}_{src.stem}_{idx:03d}.png'
            rel = Path('positives') / name
            crop.save(output / rel)
            manifest['positives'].append({'file': str(rel), 'source': str(src), 'crop_index': idx})
            written += 1
        if args.max_images and written >= args.max_images:
            break

    manifest['num_positive_images'] = written
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'[trace-dataset] wrote {written} positive images to {pos_out}')
    print(f'[trace-dataset] manifest: {output / "manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
