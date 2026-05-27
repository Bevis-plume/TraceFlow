# iDLG Baseline (Modular Reproduction)

This baseline follows the requested architecture:

```text
baselines/idlg/
├── model.py
├── idlg_attack.py
├── run_idlg_mnist.py
└── README.md
```

Official reference implementation:
- `Improved-Deep-Leakage-from-Gradients/iDLG.py`

## File Roles

- `model.py`
   - LeNet model used in iDLG paper/release.
   - Weight initialization utility (`weights_init`).

- `idlg_attack.py`
   - iDLG / DLG core optimization loop with LBFGS.
   - Label inference from final FC gradient for iDLG.
   - Three key visualizations:
      - Three-stage reconstruction (`three_phase.png`)
      - Reconstructed vs original (`recon_vs_original.png`)
      - Metric curves (`metrics_curve.png`)
   - Metrics: MSE, SSIM, PSNR.

- `run_idlg_mnist.py`
   - MNIST experiment entry.
   - Ablation across `batch size` and `initialization`.
   - Writes per-run JSON + global CSV/JSON summary.

## Run In Your Conda Env

From repo root:

```bash
conda run -n ch3-3 python baselines/idlg/run_idlg_mnist.py \
   --iters 240 \
   --batch-sizes 1 2 4 \
   --inits normal uniform zeros \
   --output-dir results/idlg_mnist
```

Quick smoke test:

```bash
conda run -n ch3-3 python baselines/idlg/run_idlg_mnist.py \
   --iters 40 \
   --batch-sizes 1 \
   --inits normal \
   --output-dir results/idlg_mnist_smoke
```

## Outputs

For each run (example `iDLG_bs1_normal`):

- `results/idlg_mnist/iDLG_bs1_normal/three_phase.png`
- `results/idlg_mnist/iDLG_bs1_normal/recon_vs_original.png`
- `results/idlg_mnist/iDLG_bs1_normal/metrics_curve.png`
- `results/idlg_mnist/iDLG_bs1_normal/iter_metrics.json`
- `results/idlg_mnist/iDLG_bs1_normal/summary.json`

Global summary:

- `results/idlg_mnist/all_results.json`
- `results/idlg_mnist/all_results.csv`

## Notes

- iDLG label inference is exact when `batch_size=1` under CE + one-hot setup.
- For `batch_size>1`, this baseline uses top-k heuristic label inference, so reconstruction quality usually drops.
