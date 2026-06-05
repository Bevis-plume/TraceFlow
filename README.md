# TraceFlow

TraceFlow is a research codebase for traceable latent rectified-flow image generation under gradient/model inversion attacks.

## One Config File

The active configuration is intentionally simple:

```text
configs/traceflow.yml
```

Edit that one file for normal work. Change fields such as:

```yaml
training:
  num_steps: 50000
  batch_size: 24
  grad_accum_steps: 3

entries:
  train_final:
    overrides:
      training:
        batch_size: 16
        grad_accum_steps: 4
        mixed_precision: bf16

data:
  name: cifar10
  root: ./data

experiments:
  exp04:
    attack_steps: 100
```

Older split configs are archived under `docs/archive/` only for reference.

## Final Server Commands

TraceFlow exposes one preflight entry plus three recommended workflow entries. All of them read the
single config file and write a complete downloadable artifact bundle containing
resolved configs, logs, checkpoints, samples, figures, reports, and metrics.

Check everything before a formal server run:

```bash
python -B -m scripts.traceflow check-ready \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/preflight
```

Train only the generative model:

```bash
python -B -m scripts.traceflow train-generator \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/generator_50k \
  --detach
```

Train the final TraceFlow model:

```bash
python -B -m scripts.traceflow train-final \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/final_50k \
  --detach
```

Run exp01-exp05 and generate paper artifacts:

```bash
python -B -m scripts.traceflow run-all \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/paper_all_50k \
  --detach
```

Watch a detached run:

```bash
tail -f /root/autodl-tmp/traceflow_runs/final_50k/logs/main.log
```

For training progress, prefer the structured JSONL log:

```bash
watch -n 10 'tail -n 3 /root/autodl-tmp/traceflow_runs/final_50k/outputs/traceflow-cifar50k-final/train_log.jsonl'
```

Temporary server overrides are still possible without editing YAML:

```bash
python -B -m scripts.traceflow train-final \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/final_20k \
  --set data.root=/root/autodl-tmp/traceflow_data/images_256 \
  --set training.num_steps=20000 \
  --detach
```

The top-level `batch_size=24, grad_accum_steps=3` is used by generator-only
training. Full TraceFlow uses the safer entry override
`batch_size=16, grad_accum_steps=4, mixed_precision=bf16` because the watermark
path adds a decode/re-encode cycle. If you want to push memory harder after a
successful preflight, create a separate bundle and try:

```bash
python -B -m scripts.traceflow train-final \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/final_50k_bs16 \
  --set training.batch_size=16 \
  --set training.grad_accum_steps=4 \
  --detach
```

If that run hits CUDA OOM, fall back to `batch_size=12, grad_accum_steps=6`.

Advanced/debug commands (`train`, `experiment`, `figures`, `readiness`) remain
available, but normal server runs should use the three entries above.

## Active Experiments

| ID | Purpose |
|---|---|
| exp01 | generation baseline |
| exp02 | keyed latent semantic failure |
| exp03 | TraceFlow identity ablation |
| exp04 | full TraceFlow inversion |
| exp05 | robustness under transforms |

## Visualisation

Training dashboards are generated automatically into each bundle:

```text
<bundle>/figures/training/
<bundle>/reports/training_summary.md
<bundle>/reports/loss_diagnosis.md
```

When sample checkpoints are produced, training figures also include:

```text
<bundle>/figures/training/latent_trajectory_3d.png
<bundle>/figures/training/latent_trajectory_3d.pdf
```

This is a PCA 3D projection of the reverse-flow latent trajectory from the
initial noise point toward the generated sample.

Paper-level figures include `fig1_pipeline` through `fig6_training_dashboard`, plus `summary.csv` and `summary.md`.

## Quick Checks

```bash
python -B -m scripts.test_keyed_bottleneck
python -B -m scripts.test_traceflow_grad_paths
python -B -m scripts.test_traceflow_watermarking
python -B -m scripts.test_data_loading
python -B -m scripts.traceflow check-ready --bundle-dir /tmp/traceflow_preflight --set project.device=auto
python -B -m scripts.traceflow train-generator --dry-run --smoke --bundle-dir /tmp/traceflow_train_generator_check --set project.device=auto
python -B -m scripts.traceflow train-final --dry-run --smoke --bundle-dir /tmp/traceflow_train_final_check --set project.device=auto
python -B -m scripts.traceflow run-all --dry-run --smoke --bundle-dir /tmp/traceflow_run_all_check --set project.device=auto
```
