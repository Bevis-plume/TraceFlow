# TraceFlow CIFAR-32 Paper — Metrics Summary

All values from `results/`. `not applicable` = method has no watermark/latent-transform. `—` = data not collected.

| Category | Metric | Exp01 Baseline | Exp02 Keyed | Exp03 Full TraceFlow |
|---|---|---|---|---|
| Generation | Training steps | 50000 | 50000 | 30000 |
| Generation | Flow loss (total) | 0.9406 | 1.0233 | 1.1402 |
| Generation | Warm-start | No (scratch) | No (scratch) | Yes (from Exp01) |
| Autoencoder | PSNR (dB) | 39.5 | (shared) | (shared) |
| Autoencoder | SSIM | 0.993 | (shared) | (shared) |
| Inversion | No-key GML | — | 12.6 | 46.1 |
| Inversion | No-key PSNR (dB) | — | 9.6 | 9.4 |
| Inversion | No-key SSIM | — | 0.127 | 0.126 |
| Inversion | Defender PSNR (dB) | — | 18.8 | 12.5 |
| Inversion | Defender SSIM | — | 0.731 | 0.407 |
| Watermark | Raw no-key image bit acc | not applicable | not applicable | 0.5078 |
| Watermark | Post-WM defender image bit acc | not applicable | not applicable | 0.6875 |
| Watermark | Post-WM defender latent bit acc | not applicable | not applicable | 0.7031 |
