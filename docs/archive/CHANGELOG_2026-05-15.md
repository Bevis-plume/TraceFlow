> **ARCHIVED** — This document describes the earlier Trackable_Inversion prototype,
> which has been removed from active code. It is preserved here for historical reference only.
> Active TraceFlow documentation is in `docs/dev_workflow.md` and `README.md`.

# Change Log - 2026-05-15

## Summary

This update turns the project from a simple latent permutation watermarking
prototype into a more algorithmic traceable inversion-defense system.

The original design used:

- key-derived latent permutation plus bias
- one global MLP watermark detector
- inconsistent image normalisation across training, attack, and evaluation
- random diffusion timestep/noise during target-gradient extraction

The updated design uses:

- keyed block-wise orthogonal latent mixing
- block-coded watermark extraction
- explicit `[0, 1]` VAE image range helpers
- fixed timestep/noise support for reproducible gradient-inversion testing
- safer checkpoint handling for key-derived buffers

## Algorithm Changes

### Keyed Block-Wise Orthogonal Mixing

`src/crypto/latent_permute.py` now implements a keyed invertible mixing layer:

```text
z' = M_K(z) + beta_K
z  = M_K^T(z' - beta_K)
```

Each block transform is an orthogonal matrix derived from the secret key.  This
keeps the latent transform exactly invertible for the defender while making the
attacker-facing latent representation harder to interpret than a plain index
permutation.

The key-derived buffers are registered with `persistent=False`, so new
checkpoints do not store the mixing matrices or bias directly.  They are
rebuilt from the secret key when the model is instantiated.

### Block-Coded Watermark

`src/models/watermarker.py` was redesigned from a global MLP into a shared
block-wise detector.  The mixed latent is split into blocks, each block predicts
a small group of watermark bits, and the bit groups are concatenated into the
final 64-bit trace signal.

This aligns the watermark extraction mechanism with the block-wise latent
mixing mechanism and gives the method a clearer algorithmic structure.

### Image Range Contract

`src/utils/image.py` was added to make the VAE image convention explicit:

- dataset images enter the VAE in `[0, 1]`
- decoded images are clamped to `[0, 1]`
- evaluation reuses the same convention

`scripts/train_defense.py`, `scripts/run_attack.py`, and
`scripts/eval_traceability.py` now share this contract.

### Reproducible Gradient Inversion

`Trainer.get_target_gradients()` now accepts optional fixed diffusion timestep
and noise tensors.  `scripts/run_attack.py` uses the same timestep/noise for
target-gradient extraction and attack optimisation, so the gradient matching
objective is no longer comparing against a randomly different diffusion draw.

`src/attacks/inversion.py` also now uses stable `[0, 1]` initialisation and
fills missing gradients with zeros in the matching loss.

## Engineering Fixes

- `src/models/unet.py` now chooses a valid GroupNorm group count automatically,
  preventing crashes when channel counts are not divisible by 32.
- `.gitignore` now ignores Python bytecode caches.
- `configs/default.yml` adds `block_size` settings for the permuter and
  watermarker.

## Validation Run

Dependencies were installed in the Codex Python 3.12 runtime:

```text
torch 2.12.0
torchvision 0.27.0
pyyaml 6.0.3
scikit-image 0.26.0
matplotlib 3.10.9
```

Smoke tests performed:

1. Random tensor VAE/mixing/watermark/UNet forward pass
2. Real CIFAR-10 image forward pass
3. Tiny two-step gradient-inversion attack smoke test

Observed real-image smoke-test values:

```text
label = 3
image = (1, 3, 32, 32)
latent = (1, 4, 8, 8)
mixed latent = (1, 4, 8, 8)
invert_max_abs_error = 4.77e-07
watermark = (1, 64)
attack_decode PSNR = 13.63
attack_decode SSIM = 0.3186
```

Observed tiny attack smoke-test values:

```text
num_target_grads = 80
final_loss = 0.9258
x_reconstructed = (1, 3, 32, 32)
z_prime_dummy = (1, 4, 8, 8)
psnr_vs_target = 13.89
ssim_vs_target = 0.3192
```

Generated visual check:

```text
results/smoke_cifar_target_vs_mixed_decode.png
```

## Important Caveat

These tests verify that the upgraded code path runs correctly.  They are not
final experimental results because the model has not been retrained after the
algorithm change.  A formal report still needs a full run:

```bash
python -m scripts.train_defense --config configs/default.yml
python -m scripts.run_attack --config configs/default.yml --checkpoint ./checkpoints/ckpt_best.pt
python -m scripts.eval_traceability --config configs/default.yml --checkpoint ./checkpoints/ckpt_best.pt
```
