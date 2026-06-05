# CUDA Experiment Runbook

Use one config file:

```text
configs/traceflow.yml
```

## Environment

```bash
conda create -n traceflow-cuda python=3.11 -y
conda activate traceflow-cuda
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from diffusers import AutoencoderKL; print('diffusers ok')"
```

## Data

For final ImageFolder runs, edit `configs/traceflow.yml`:

```yaml
data:
  name: imagefolder
  root: /root/autodl-tmp/traceflow_data/images_256
  image_size: 256
```

For CIFAR pipeline validation, keep:

```yaml
data:
  name: cifar10
  root: ./data
```

## Run

```bash
python -B -m scripts.traceflow experiment
```

Or override without editing:

```bash
python -B -m scripts.traceflow experiment   --set data.root=/root/autodl-tmp/traceflow_data/images_256   --set training.num_steps=50000
```

## Figures And Readiness

```bash
python -B -m scripts.traceflow figures --results-dir results/traceflow_cifar_50k
python -B -m scripts.traceflow readiness --results-dir results/traceflow_cifar_50k --strict
```

Resolved private configs are written to `local_configs/resolved/`. Public result config copies redact `secret_key`.
