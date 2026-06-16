"""
scripts/debug_flow_overfit.py
===============================
Decisive, self-contained diagnostic for the rectified-flow generator.

It isolates the FlowTransformer + sampler from the autoencoder and the dataset:
a tiny fixed set of K distinct, structured latent "modes" (one per class) is the
entire data distribution. We overfit the flow on it, then sample.

Interpretation
--------------
- If sampling reproduces the memorised modes (low error)  -> the flow + sampler
  code is CORRECT; the colourful-noise failure is undertraining / latent-prior /
  EMA / schedule, not a model bug.
- If sampling fails to reproduce even a handful of memorised modes -> there is a
  genuine code/architecture bug in the flow forward, the time conditioning, or
  the sampler.

This script does not touch CUDA-only paths and runs on CPU/MPS in well under a
minute. It is intentionally dependency-light (only torch + the project model).
"""

from __future__ import annotations

import argparse

import torch

from src.generation.rectified_flow import sample_euler, flow_loss
from src.models.flow_transformer import build_flow_transformer
from src.utils.checkpoint import EMAModel


def make_modes(num_modes: int, c: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Build K distinct, smooth, unit-scaled latent targets (one per class)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    modes = []
    for k in range(num_modes):
        chans = []
        for ch in range(c):
            fx = float(torch.randint(1, 4, (1,), generator=g).item())
            fy = float(torch.randint(1, 4, (1,), generator=g).item())
            phase = float(torch.rand(1, generator=g).item()) * 3.14159
            amp = 0.5 + float(torch.rand(1, generator=g).item())
            pattern = amp * torch.sin(fx * 3.14159 * xs + fy * 3.14159 * ys + phase + k)
            chans.append(pattern)
        modes.append(torch.stack(chans, dim=0))
    z = torch.stack(modes, dim=0).to(device)
    # Standardise to ~unit per-channel std, matching normalised AE latents.
    z = (z - z.mean()) / (z.std() + 1e-6)
    return z


@torch.no_grad()
def time_sensitivity(model: torch.nn.Module, z0: torch.Tensor, y: torch.Tensor) -> float:
    """Mean output change as t sweeps 0->1 on a fixed input (probes conditioning)."""
    outs = []
    for t_val in torch.linspace(0.0, 1.0, 11):
        t = torch.full((z0.size(0),), float(t_val), device=z0.device)
        outs.append(model(z0, t, y))
    outs = torch.stack(outs, dim=0)
    return float(outs.std(dim=0).mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--modes", type=int, default=4)
    ap.add_argument("--time-scale", type=float, default=1000.0)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema-decay", type=float, default=0.9999,
                    help="EMA decay used for the EMA sampling path comparison.")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)
    print(f"[overfit] device={device} time_scale={args.time_scale} modes={args.modes}")

    C, H, W = 4, 16, 16
    K = args.modes
    z_modes = make_modes(K, C, H, W, device)        # (K, C, H, W)
    y_modes = torch.arange(K, device=device)        # one class per mode

    model = build_flow_transformer(
        preset=None,
        latent_channels=C,
        latent_size=H,
        patch_size=2,
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        class_conditional=True,
        num_classes=K,
        time_scale=args.time_scale,
    ).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMAModel(model, decay=args.ema_decay)

    for step in range(1, args.steps + 1):
        opt.zero_grad()
        loss = flow_loss(model, z_modes, y=y_modes)
        loss.backward()
        opt.step()
        ema.update(model)
        if step % 500 == 0 or step == 1:
            print(f"[overfit] step={step:5d} flow_loss={loss.item():.4f}")

    model.eval()
    sens = time_sensitivity(model, z_modes, y_modes)
    print(f"[overfit] time-sensitivity (output std across t) = {sens:.4f}")
    data_std = float(z_modes.std().item())

    def _sample_and_score(tag: str) -> float:
        with torch.no_grad():
            z_gen = sample_euler(
                model, (K, C, H, W), args.sample_steps, device, y=y_modes
            )
        err = (z_gen - z_modes).pow(2).mean(dim=(1, 2, 3)).sqrt()
        gen_std = float(z_gen.std().item())
        print(f"[overfit] [{tag}] per-mode sample RMSE: "
              + ", ".join(f"{e:.3f}" for e in err.tolist()))
        print(f"[overfit] [{tag}] mean sample RMSE = {err.mean().item():.4f}  "
              f"(gen_std={gen_std:.3f} vs data_std={data_std:.3f})")
        return float(err.mean().item())

    # Sample from the RAW (current) weights.
    raw_rmse = _sample_and_score("raw")
    # Sample from the EMA weights, exactly like the real training loop does.
    with ema.average_parameters(model):
        ema_rmse = _sample_and_score(f"ema d={args.ema_decay}")
    err = torch.tensor([raw_rmse])

    # A correct flow that memorised K modes should reproduce them with small
    # RMSE relative to the inter-mode distance.
    inter = torch.cdist(z_modes.flatten(1), z_modes.flatten(1))
    inter = inter[inter > 0]
    typical_gap = float(inter.mean().item()) / (C * H * W) ** 0.5
    thresh = 0.25 * (typical_gap + 1e-6) + 0.2
    raw_ok = raw_rmse < thresh
    ema_ok = ema_rmse < thresh
    print(f"[overfit] typical inter-mode RMSE gap = {typical_gap:.3f} (pass threshold {thresh:.3f})")
    print(f"[overfit] RAW model: {'PASS' if raw_ok else 'FAIL'} (RMSE {raw_rmse:.3f})")
    print(f"[overfit] EMA model (d={args.ema_decay}): "
          f"{'PASS' if ema_ok else 'FAIL'} (RMSE {ema_rmse:.3f})")
    if raw_ok and not ema_ok:
        print("[overfit] VERDICT: flow+sampler CODE is correct; EMA decay is too "
              "high for this run length and CRIPPLES sampling (damped velocity).")
    elif raw_ok and ema_ok:
        print("[overfit] VERDICT: flow+sampler and EMA both OK at this length.")
    else:
        print("[overfit] VERDICT: RAW sampling fails -> genuine flow/sampler code bug.")


if __name__ == "__main__":
    main()
