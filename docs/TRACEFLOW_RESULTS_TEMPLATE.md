# TraceFlow Results Template

Paste generated tables and figures from `scripts.make_traceflow_figures` here after full CUDA experiments finish.

## Table 1: Ablation Metrics

Use `results/traceflow_cuda/figures/summary.md`.

## Figure Checklist

- `fig1_pipeline`: TraceFlow method pipeline.
- `fig2_ablation`: baseline/keyed/identity/full comparison.
- `fig3_attack_grid`: original, no-key reconstruction, defender reconstruction, pixel attack, post-watermark sanity.
- `fig4_curves`: training and attack curves.
- `fig5_robustness`: detector bit accuracy under transformations.

## Claims to Verify

- Generation quality is not destroyed by TraceFlow losses.
- Keyed latent space makes no-key decoding semantically poor.
- Raw inversion outputs carry detectable TraceFlow signal.
- Latent detector adds value beyond image detector.
- Robustness remains above the selected traceability threshold.
