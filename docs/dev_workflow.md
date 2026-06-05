# TraceFlow Development Workflow

Use macOS for code edits, unit tests, smoke dry-runs, and figure-script validation. Use CUDA for real training with the diffusers AutoencoderKL backend.

## Mac Smoke Checks

```bash
python -B -m scripts.test_keyed_bottleneck
python -B -m scripts.test_traceflow_grad_paths
python -B -m scripts.test_traceflow_watermarking
python -B -m scripts.test_data_loading
python -B -m scripts.run_experiments --all --smoke --dry-run --output-dir results/traceflow_cleanup_dry
```

## Active Model Path

Only the final `watermark.type: traceflow` path is active. Earlier prototype watermark variants have been removed from active code. Keep ablations focused on baseline, keyed-only, TraceFlow identity, full TraceFlow, and robustness.

## CUDA Development

Before long runs, use suite/runtime YAMLs such as `configs/suites/paper_imagefolder_50k.yml` and `configs/runtimes/pilot_50k.yml`. Replace placeholder keys only in local resolved configs under `local_configs/`, which is ignored.
