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
  num_steps: 100000
  batch_size: 96
  grad_accum_steps: 1

entries:
  train_final:
    overrides:
      training:
        batch_size: 48
        grad_accum_steps: 2
        mixed_precision: bf16

watermark:
  alpha: 0.08
  lambda_wm_img: 2.0
  lambda_wm_latent: 1.0
  lambda_residual: 0.001

data:
  name: imagefolder
  root: data/imagenette_woof_320/train
model:
  num_classes: 20

assets:
  data_dir: data
  weights_dir: weights

experiments:
  exp04:
    attack_steps: 100
```

Paper/default runs use a merged Imagenette-320 + Imagewoof-320 ImageFolder
dataset cropped to 256x256 for fast paper pilots. CIFAR-10 is still supported
only for smoke/debug runs, because upsampling CIFAR to 256 produces very blurry
samples and is not a fair final showcase for DiT-style generation.

Expected ImageFolder layout:

```text
data/imagenette_woof_320/train/
  n01440764/*.JPEG
  n01443537/*.JPEG
  ...
```

Older split configs are archived under `docs/archive/` only for reference.

## Prepare Local Assets

To make the server run boring in the good way, pre-download the small paper
datasets and metric weights into project-local folders before packaging:

```bash
python -B -m scripts.traceflow prepare-assets --config configs/traceflow.yml
```

This creates/checks:

```text
data/imagenette2-320/train/
data/imagewoof2-320/train/
data/imagenette_woof_320/train/  # merged ImageFolder used by training
weights/torch/          # Inception/FID/KID cache used by torchmetrics
weights/huggingface/    # optional model/cache home
pretrained/sd-vae-ft-mse/
```

The Imagenette and Imagewoof archives are each only a few hundred MB. The merged
folder is built with hardlinks by default, so it should not duplicate image data
on normal filesystems. LPIPS/Inception/FID weights are also well below 1 GB in
normal setups, and keeping them in `weights/` avoids surprise downloads on the
server. If optional metric packages are missing locally, `prepare-assets`
records a warning; the same command can be rerun after `pip install -r
requirements.txt`.

## Final Server Commands

TraceFlow's recommended server path is one command: `run-all`. It reads
`configs/traceflow.yml`, writes one downloadable artifact bundle, trains only the
necessary checkpoints, reuses those checkpoints for exp01-exp05, then generates
training dashboards, paper figures, metrics, and readiness reports.

Check everything before a formal server run:

```bash
python -B -m scripts.traceflow check-ready \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/preflight_pro6000
```

Run the complete RTX PRO 6000 paper pipeline:

```bash
python -B -m scripts.traceflow run-all \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/paper_pro6000_imagenettewoof256 \
  --detach
```

`run-all` performs:

```text
Stage 0: dataset/VAE diagnosis -> reports/data_diagnosis/
Stage 1: train/reuse generator, keyed, identity, and full TraceFlow checkpoints
Stage 2: run exp01-exp05 with train_policy=never, reusing those checkpoints
Stage 3: generate training figures, paper figures, readiness, and manifest files
```

The final bundle contains the files you download for the paper:

```text
/root/autodl-tmp/traceflow_runs/paper_pro6000_imagenettewoof256/
  configs/
  logs/
  outputs/
  checkpoints/
    generator/latest.pt
    keyed/latest.pt
    identity/latest.pt
    traceflow/latest.pt
  results/exp01 ... results/exp05
  figures/training/
  figures/paper/
  reports/checkpoint_manifest.json
  reports/readiness_report.md
  reports/manifest.json
```

Watch a detached run:

```bash
tail -f /root/autodl-tmp/traceflow_runs/paper_pro6000_imagenettewoof256/logs/main.log
```

Force all four model stages to retrain instead of reusing existing checkpoints:

```bash
python -B -m scripts.traceflow run-all \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/paper_pro6000_imagenettewoof256 \
  --force-train \
  --detach
```

For a faster probe, keep the same one-command pipeline and lower the step
budget:

```bash
python -B -m scripts.traceflow run-all \
  --config configs/traceflow.yml \
  --bundle-dir /root/autodl-tmp/traceflow_runs/paper_pro6000_imagenettewoof256_probe \
  --set training.num_steps=20000 \
  --detach
```

Advanced/debug entries (`train-generator`, `train-keyed`, `train-identity`,
`train-final`, and `eval-all`) remain available for staged debugging, but the
formal paper workflow should use `run-all`.

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
python -B -m scripts.traceflow train-keyed --dry-run --smoke --bundle-dir /tmp/traceflow_train_keyed_check --set project.device=auto
python -B -m scripts.traceflow train-identity --dry-run --smoke --bundle-dir /tmp/traceflow_train_identity_check --set project.device=auto
python -B -m scripts.traceflow train-final --dry-run --smoke --bundle-dir /tmp/traceflow_train_final_check --set project.device=auto
python -B -m scripts.traceflow run-all --dry-run --smoke --bundle-dir /tmp/traceflow_run_all_check --set project.device=auto
python -B -m scripts.traceflow eval-all --dry-run --smoke --bundle-dir /tmp/traceflow_eval_all_check --generator-checkpoint /tmp/fake_generator.pt --keyed-checkpoint /tmp/fake_keyed.pt --identity-checkpoint /tmp/fake_identity.pt --traceflow-checkpoint /tmp/fake_traceflow.pt --set project.device=auto
```
