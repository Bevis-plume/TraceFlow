# TraceFlow CIFAR-10 32x32 Latent-16 VAE Paper Runbook

This runbook drives the serious, paper-quality CIFAR-10 rerun. Priorities, in
order: **image quality first, watermark traceability second, no-key inversion
resistance third, robustness supportive.**

It is split into two clearly separated halves:

- **[A. Local Mac development](#a-local-mac-development)** — static checks, smoke
  tests, and figure generation from existing metrics. **No CUDA, no real
  training.**
- **[B. Server RTX 5090 paper run](#b-server-rtx-5090-paper-run)** — the full
  paper sequence on a single **NVIDIA RTX 5090 (32 GB)**.

Config: [`configs/traceflow_cifar32.yml`](../configs/traceflow_cifar32.yml)
Bundle root: `runs/traceflow-cifar32_lat16_vae/` by default, and the recommended
server bundle is `/root/autodl-fs/traceflow_runs/traceflow_cifar32_lat16_vae_paper_30k`.
This is separated from both the old 256 run and the failed CIFAR latent-32 pilot;
do not reuse `cifar32`/latent-32 checkpoints with this config.

---

## Why this path exists

The previous 256x256 Imagenette/Imagewoof run produced poor generation quality,
and the watermark/keyed path degraded samples too much. Two root fixes:

1. **The local autoencoder must be a sampleable VAE prior.** Reconstruction
   alone is not enough: a deterministic AE can reconstruct real latents while
   decoding random/flow-generated latents as noise. We now train the local AE
   with posterior sampling + KL/free-bits, save recon/prior/posterior
   diagnostics, and load it frozen into every downstream TraceFlow run. Directly
   decoding `N(0,I)` latents (`ae_prior_flow_grid.png`) is expected to look like
   noise for a flow/diffusion model; it is only a sanity diagnostic. The actual
   quality gate is the generator sample plus `denoise_probe_step*.png`. **Do not
   use the SD/diffusers VAE for CIFAR.**
2. **The watermark objective is rebalanced** so image quality is primary: lower
   `alpha`, lower watermark loss weights, higher preservation/invisibility
   weights, and robustness is delayed (disabled) for the first paper run. Target
   generated watermark accuracy is ~0.80–0.90; anything clearly above 0.50 is
   meaningful. Clean false positive should stay near 0.50.
3. **The paper default is latent-16 VAE, not latent-32.** CIFAR-10 remains native
   32x32, but the local VAE compresses to `4x16x16`. With `patch_size=2`, the DiT
   sees `8x8=64` tokens instead of the latent-32 pilot's `16x16=256` tokens.
   This makes training much cheaper and avoids treating CIFAR almost like a
   pixel-space diffusion problem.

---

## A. Local Mac development

The Mac is **only** for development, static checks, smoke tests, and figure
generation from existing metrics. **Never run real training here and never assume
CUDA exists.** Smoke mode uses random data, a tiny model, a few steps, and falls
back to MPS/CPU automatically.

### A1. Install / check dependencies

```bash
pip install -r requirements.txt
python -m scripts.traceflow check-ready --config configs/traceflow_cifar32.yml
```

### A2. Static 32 GB batch-size preflight (no CUDA needed)

```bash
python -m scripts.traceflow estimate-5090 --config configs/traceflow_cifar32.yml
```

Prints, per training stage, the configured micro-batch, the conservative 32 GB
ceiling, and the OOM-retry ladder `run-all` would fall back to. Run this on the
Mac before launching anything on the server.

### A3. Smoke test the autoencoder pretraining

```bash
python -m scripts.traceflow train-autoencoder --config configs/traceflow_cifar32.yml --smoke
```

### A4. Smoke test the generator trainer

```bash
python -m scripts.traceflow train-generator --config configs/traceflow_cifar32.yml --smoke
```

### A5. Smoke test the full run-all pipeline

```bash
python -m scripts.traceflow run-all --config configs/traceflow_cifar32.yml --smoke
```

Smoke `run-all` skips the real dataset diagnosis stage and uses random data.
LPIPS is used only when the AlexNet weights are already cached, so smoke tests do
not unexpectedly download large metric weights.

### A6. Generate the curated paper figures from existing metrics

```bash
python -m scripts.traceflow paper-figures --config configs/traceflow_cifar32.yml
```

This needs no GPU — it reads whatever metrics already exist in the bundle (e.g.
copied back from the server) and writes `PAPER_CIFAR32_RESULTS/`. Missing methods
are reported as `missing`; metrics that do not apply (watermark accuracy for a
no-watermark baseline) are `not_applicable`.

---

## B. Server RTX 5090 paper run

All defaults stay within 32 GB and avoid OOM. The intended server path is:
**download CIFAR on the server → preflight → run-all → curated paper figures**.

### B1. Download CIFAR-10 on the server

```bash
python -m scripts.traceflow prepare-data --config configs/traceflow_cifar32.yml
```

This downloads/extracts CIFAR-10 into `data/cifar-10-batches-py/`. The config has
`data.download: true` for server runs, so training can also download it lazily,
but running `prepare-data` first makes failures obvious before expensive jobs.

### B2. Full preflight

```bash
BUNDLE=/root/autodl-fs/traceflow_runs/traceflow_cifar32_lat16_vae_paper_30k
python -m scripts.traceflow check-ready --config configs/traceflow_cifar32.yml --bundle-dir "$BUNDLE"
python -m scripts.traceflow estimate-5090 --config configs/traceflow_cifar32.yml
```

`check-ready` validates Python modules, CUDA availability, CIFAR files, local-AE
configuration, writable bundle paths, and resolved experiment configs.
`estimate-5090` prints the configured micro-batches against the conservative
32 GB ceilings.

### B3. One-shot run-all

```bash
BUNDLE=/root/autodl-fs/traceflow_runs/traceflow_cifar32_lat16_vae_paper_30k
python -m scripts.traceflow run-all \
  --config configs/traceflow_cifar32.yml \
  --bundle-dir "$BUNDLE" \
  --set training.num_steps=30000 \
  --set sampling.steps=50 \
  --attack geiping_pixel \
  --attack-steps 300 \
  --attacker no_key \
  --foreground
```

`run-all` now automatically trains or reuses the shared local autoencoder first
inside the active bundle (`checkpoints/autoencoder/latest.pt` under `--bundle-dir`),
then trains/reuses the baseline generator,
keyed latent baseline, TraceFlow identity model, and full TraceFlow model. The
full TraceFlow model is warm-started from the baseline generator to protect image
quality. Evaluation then reuses those checkpoints and never trains inside eval.

### B4. Curated paper figures

`run-all` automatically writes curated CIFAR paper figures into the bundle at
`runs/traceflow-cifar32_lat16_vae/<run>/PAPER_CIFAR32_RESULTS/`. You can regenerate or copy
them to a top-level folder with:

```bash
python -m scripts.traceflow paper-figures --config configs/traceflow_cifar32.yml \
  --output-dir PAPER_CIFAR32_RESULTS
```

Figures use method names (Baseline Generator / Keyed Latent / TraceFlow Identity
/ Full TraceFlow / No-Key Inversion / Defender Decode / Clean Images /
Watermarked Forensic) — never `exp01`–`exp05`.

### Optional manual stage commands

The following remain useful for debugging, but are not required for the normal
server flow because `run-all` performs the required training automatically.

#### Train the baseline generator

```bash
python -m scripts.traceflow train-generator --config configs/traceflow_cifar32.yml
```

#### Train the keyed latent baseline

```bash
python -m scripts.traceflow train-keyed --config configs/traceflow_cifar32.yml
```

#### Train the TraceFlow identity watermark model

```bash
python -m scripts.traceflow train-identity --config configs/traceflow_cifar32.yml
```

#### Train the full TraceFlow keyed watermark model (warm-started)

To protect image quality, warm-start from the baseline generator rather than
from scratch:

```bash
python -m scripts.traceflow train-final \
  --config configs/traceflow_cifar32.yml \
  --init-from runs/traceflow-cifar32_lat16_vae/<generator-run>/checkpoints/<generator-run>/latest.pt
```

`run-all` does this warm-start automatically (see below). `--init-from` loads the
flow model + EMA only; the optimizer/step are fresh and the watermark modules
start fresh.

#### Run evaluation using existing checkpoints

```bash
python -m scripts.traceflow eval-all --config configs/traceflow_cifar32.yml
```

Inversion evaluation runs at **batch size 1**, with latent and pixel attacks as
**separate processes**, clearing the CUDA cache between stages — safe for 32 GB.

#### Generate the curated paper figures (method names, one folder)

```bash
python -m scripts.traceflow paper-figures --config configs/traceflow_cifar32.yml
```

All final assets land in **`PAPER_CIFAR32_RESULTS/`**. Figures use method names
(Baseline Generator / Keyed Latent / TraceFlow Identity / Full TraceFlow /
No-Key Inversion / Defender Decode / Clean Images / Watermarked Forensic) — never
`exp01`–`exp05`.

### B5. Compact server command sequence

Recommended server sequence:

```bash
BUNDLE=/root/autodl-fs/traceflow_runs/traceflow_cifar32_lat16_vae_paper_30k
python -m scripts.traceflow prepare-data --config configs/traceflow_cifar32.yml
python -m scripts.traceflow check-ready --config configs/traceflow_cifar32.yml --bundle-dir "$BUNDLE"
python -m scripts.traceflow estimate-5090 --config configs/traceflow_cifar32.yml
python -m scripts.traceflow run-all \
  --config configs/traceflow_cifar32.yml \
  --bundle-dir "$BUNDLE" \
  --set training.num_steps=30000 \
  --set sampling.steps=50 \
  --attack geiping_pixel \
  --attack-steps 300 \
  --attacker no_key \
  --foreground
# Optional stronger final inversion once 30k samples look good:
python -m scripts.traceflow eval-all \
  --config configs/traceflow_cifar32.yml \
  --bundle-dir "$BUNDLE" \
  --attack geiping_pixel \
  --attack-steps 1000 \
  --attacker no_key \
  --foreground
```

### Pilot vs paper run sizes

- **Pilot** (mechanism validation, ~5k–10k steps):

  ```bash
  python -m scripts.traceflow run-all --config configs/traceflow_cifar32.yml \
    --set training.num_steps=8000
  ```

- **Paper** (default 30k steps): no override needed.

- **Extend to 50k** (resume the paper run):

  ```bash
  python -m scripts.traceflow train-final --config configs/traceflow_cifar32.yml \
    --set training.num_steps=50000 \
    --resume runs/traceflow-cifar32_lat16_vae/<final-run>/checkpoints/<final-run>/latest.pt
  ```

### Export / backup the curated results

```bash
tar czf paper_cifar32_results.tgz PAPER_CIFAR32_RESULTS/
# copy the bundle's metrics back to the Mac to regenerate figures offline:
rsync -av runs/traceflow-cifar32_lat16_vae/<run>/results/ <mac>:/path/to/runs/.../results/
```

---

## OOM safety summary (RTX 5090 32 GB)

- Conservative micro-batches: generator/keyed `128`, identity `64`, full
  TraceFlow `48` (all overridable in `entries`); validate with `estimate-5090`.
- `mixed_precision: bf16`, `torch_compile: false`.
- **Inversion/eval batch size is fixed at `1`**; latent and pixel attacks run as
  **separate processes**; CUDA cache is cleared between major eval stages;
  checkpoints load on CPU first, then move to device.
- `run_all.oom_retry: true` (default) retries the watermarked stages with a
  **descending micro-batch ladder derived from the configured batch size** (e.g.
  full TraceFlow `48 → 24 → 12 → 6 → 4`) on CUDA OOM, and records every attempt
  (batch size, grad-accum, return code) in the checkpoint manifest. No stage
  silently continues after OOM with partial metrics.
- Training segments are merged and deduplicated by step for figures, so resumed
  runs do not produce misleading charts.

---

## Latent-32 pilot status

The old `traceflow_cifar32_paper_10k` / latent-32 outputs and the deterministic
latent-16 pilot outputs are diagnostic only. They either used `4x32x32` latents
or a non-sampleable AE prior, and should not be mixed with this latent-16 VAE
paper line. Start a fresh bundle and fresh AE checkpoint for every latent-16 VAE
run.

## Output map

```
runs/traceflow-cifar32_lat16_vae/<run>/
  checkpoints/        # per-stage checkpoints (+ aliases for run-all)
  outputs/            # train_log*.jsonl, sample grids, resolved configs
  results/exp01..05/  # per-method eval metrics.json (+ inversion/) — internal IDs
  reports/            # AE diagnostics, manifests, readiness
runs/traceflow-cifar32_lat16_vae/<run>/checkpoints/autoencoder/latest.pt  # run-all shared AE
checkpoints/cifar32_lat16_vae_ae/latest.pt                                  # standalone train-autoencoder default

PAPER_CIFAR32_RESULTS/             # curated, method-named, no exp IDs anywhere
  figures/      # fig1..fig8 (PNG 300 dpi + vector PDF)
  tables/       # summary.csv, summary.md, method_metrics.csv
  diagnostics/  # readiness.md + copied AE reconstruction grid/metrics
  samples/      # latest generated sample grid per method
  README.md     # overview + figure index + provenance
```

`runs/` and `PAPER_CIFAR32_RESULTS/` are git-ignored; only source, configs, and
docs are tracked.
