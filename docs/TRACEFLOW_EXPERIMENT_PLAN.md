# TraceFlow Experiment Plan

## Goal

Evaluate whether TraceFlow can generate usable images while making gradient/model inversion outputs traceable and semantically degraded without the key.

## Experiments

| ID | Question | Key metrics |
|---|---|---|
| exp01 | Does the base generator train/sample normally? | final loss, generated samples |
| exp02 | Does keyed latent training break no-key decoding? | with-key samples vs no-key samples |
| exp03 | Does the watermark mechanism work without the key transform? | generated image/latent bit accuracy, false positive |
| exp04 | Does full TraceFlow survive inversion evaluation? | gradient matching loss, no-key PSNR/SSIM, raw image/latent bit accuracy |
| exp05 | Are raw inversion outputs still detectable after transforms? | JPEG/resize/blur/noise/crop bit accuracy |

## Success Criteria

- Full TraceFlow generated images retain acceptable quality relative to baseline.
- No-key latent inversion outputs have low semantic similarity to originals.
- Raw inversion outputs recover the embedded message above random chance in image and/or latent detectors.
- Clean false-positive detection remains near random chance.
- Robustness transforms do not collapse detection below the selected threshold.

## Commands

```bash
python -B -m scripts.run_experiments --all --smoke --dry-run
python -B -m scripts.run_experiments --all --output-dir results/traceflow_cuda
python -B -m scripts.make_traceflow_figures --results-dir results/traceflow_cuda --output-dir results/traceflow_cuda/figures
python -B -m scripts.check_experiment_readiness --results-dir results/traceflow_cuda
```
