# CUDA Experiment Runbook

TraceFlow uses one active config file:

```text
configs/traceflow.yml
```

The formal paper path is Imagenette-320/ImageFolder data cropped to 256x256 for fast paper pilots on the server data disk. CIFAR is only a legacy/debug option and is not part of the final CUDA runbook.

## Environment

```bash
cd /root/autodl-tmp/TraceFlow
conda activate traceflow-cuda

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## VAE

The upload bundle should include the local VAE:

```text
pretrained/sd-vae-ft-mse/
```

Check it on the server:

```bash
python - <<'PY'
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("pretrained/sd-vae-ft-mse")
print("local VAE loaded")
PY
```

## Assets

Before packaging locally, run:

```bash
python -B -m scripts.traceflow prepare-assets --config configs/traceflow.yml
```

The upload bundle should include:

```text
data/imagenette2-320/train/
data/imagewoof2-320/train/
data/imagenette_woof_320/train/
weights/
pretrained/sd-vae-ft-mse/
```

On the server, check class and image counts from the project-local data folder:

```bash
find data/imagenette_woof_320/train -mindepth 1 -maxdepth 1 -type d | wc -l

find data/imagenette_woof_320/train -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.JPEG" -o -iname "*.png" -o -iname "*.webp" \) | wc -l
```

The merged Imagenette+Imagewoof train folder should have about 20 class folders.
For ImageNet-1K follow the same ImageFolder layout and only change `data.root`,
`model.num_classes`, and `naming.dataset_tag` in `configs/traceflow.yml`.

## Preflight

```bash
python -B -m scripts.traceflow check-ready   --config configs/traceflow.yml   --bundle-dir /root/autodl-tmp/traceflow_runs/preflight_pro6000
```

## Data/VAE Diagnosis

```bash
python -B -m scripts.traceflow diagnose-data   --config configs/traceflow.yml   --bundle-dir /root/autodl-tmp/traceflow_runs/data_diagnosis_pro6000
```

Inspect:

```text
reports/data_diagnosis/real_grid.png
reports/data_diagnosis/vae_recon_grid.png
reports/data_diagnosis/dataset_report.md
```

## RTX PRO 6000 Benchmark

```bash
python -B -m scripts.traceflow benchmark-pro6000   --config configs/traceflow.yml   --bundle-dir /root/autodl-tmp/traceflow_runs/pro6000_benchmark   --steps 300   --detach
```

Watch:

```bash
tail -f /root/autodl-tmp/traceflow_runs/pro6000_benchmark/logs/main.log
```

Read report:

```bash
cat /root/autodl-tmp/traceflow_runs/pro6000_benchmark/reports/pro6000_benchmark.md
```

## Full Paper Run

```bash
python -B -m scripts.traceflow run-all   --config configs/traceflow.yml   --bundle-dir /root/autodl-tmp/traceflow_runs/paper_pro6000_imagenettewoof256   --detach
```

Watch:

```bash
tail -f /root/autodl-tmp/traceflow_runs/paper_pro6000_imagenettewoof256/logs/main.log
```

Download the completed bundle:

```text
/root/autodl-tmp/traceflow_runs/paper_pro6000_imagenettewoof256/
```

Resolved private configs are written under each bundle's `configs/` directory. Public config copies redact `secret_key`.
