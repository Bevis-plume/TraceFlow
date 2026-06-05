"""scripts/test_traceflow_watermarking.py — TraceFlow dual-head watermark smoke test.

Validates:
  1. TraceLatentDetector forward shape.
  2. build_watermark_modules with type=traceflow returns the correct dict.
  3. Full TraceFlow pipeline: invert key → decode_with_grad → adapter → image detector
     → encode_with_grad → key transform → latent detector.
  4. Gradients flow into decoder adapter, image detector, latent detector, and
     back to z_hat_k (the input protected latent) through a combined TraceFlow loss.

No datasets, training runs, or downloads required.

Usage
-----
    python -m scripts.test_traceflow_watermarking
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.autoencoder_backend import AutoencoderBackend
from src.security.keyed_bottleneck import KeyedLatentBottleneck
from src.watermarking.latent_watermark import TraceLatentDetector
from src.watermarking.decoder_watermark import TraceDecoderAdapter
from src.watermarking.image_watermark import ImageWatermarkDetector
from src.watermarking.message import generate_watermark_bits, expand_bits, generate_random_batch_bits
from src.watermarking.factory import build_watermark_modules
from src.watermarking.metrics import bit_accuracy, bit_error_rate


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def main() -> None:
    print("[test] ── TraceFlow dual-head watermark smoke test ──")

    torch.manual_seed(7)

    B = 4
    bit_length = 32
    image_size = 64
    latent_channels = 4
    latent_size = 8
    channels = 3
    alpha = 0.02
    hidden_dim = 128
    latent_detector_hidden_dim = 64

    # ------------------------------------------------------------------
    # 1. TraceLatentDetector standalone shape test
    # ------------------------------------------------------------------
    latent_det_standalone = TraceLatentDetector(
        bit_length=bit_length,
        latent_channels=latent_channels,
        hidden_dim=latent_detector_hidden_dim,
    )
    z_dummy = torch.randn(B, latent_channels, latent_size, latent_size)
    probs_dummy = latent_det_standalone(z_dummy)
    assert probs_dummy.shape == (B, bit_length), (
        f"TraceLatentDetector shape: {tuple(probs_dummy.shape)} != ({B}, {bit_length})"
    )
    assert probs_dummy.min().item() >= 0.0 and probs_dummy.max().item() <= 1.0, (
        "TraceLatentDetector probs must be in [0, 1]"
    )
    print(f"[test] TraceLatentDetector standalone shape/range: PASS")

    # ------------------------------------------------------------------
    # 2. build_watermark_modules — traceflow type
    # ------------------------------------------------------------------
    wm_cfg = {
        "watermark": {
            "enabled": True,
            "type": "traceflow",
            "bit_length": bit_length,
            "seed": 42,
            "alpha": alpha,
            "extractor_hidden_dim": hidden_dim,
            "latent_channels": latent_channels,
            "latent_detector_hidden_dim": latent_detector_hidden_dim,
            "lambda_wm_img": 1.0,
            "lambda_wm_latent": 0.5,
            "lambda_img": 0.1,
            "lambda_cycle": 0.2,
            "lambda_residual": 0.01,
            "cycle_target": "protected_latent",
        }
    }
    wm = build_watermark_modules(wm_cfg, image_size=image_size, channels=channels)
    assert wm is not None and wm["enabled"], "build_watermark_modules returned None or disabled"
    assert wm["type"] == "traceflow", f"type mismatch: {wm['type']}"
    assert wm["extractor"] is not None, "extractor missing"
    assert wm["decoder_adapter"] is not None, "decoder_adapter missing"
    assert wm["latent_detector"] is not None, "latent_detector missing"
    
    cfg = wm["config"]
    for key in (
        "lambda_wm_img", "lambda_wm_latent", "lambda_img",
        "lambda_cycle", "lambda_residual", "cycle_target",
        "latent_detector_hidden_dim",
    ):
        assert key in cfg, f"config missing key: {key!r}"
    assert cfg["cycle_target"] == "protected_latent", (
        f"cycle_target: {cfg['cycle_target']!r}"
    )
    print("[test] build_watermark_modules traceflow: PASS")

    # ------------------------------------------------------------------
    # 3. Full TraceFlow pipeline — verify shapes end-to-end
    # ------------------------------------------------------------------
    # Components
    autoencoder = AutoencoderBackend(
        backend="local",
        latent_channels=latent_channels,
        image_size=image_size,
        latent_size=latent_size,
        freeze=True,
    )
    autoencoder.eval()

    klb = KeyedLatentBottleneck(
        secret_key="traceflow_test_key",
        latent_channels=latent_channels,
        latent_size=latent_size,
        block_size=16,
        bias_scale=0.1,
    )

    adapter: TraceDecoderAdapter = wm["decoder_adapter"]
    image_detector: ImageWatermarkDetector = wm["extractor"]
    latent_detector: TraceLatentDetector = wm["latent_detector"]
    bits = wm["bits"]
    batch_bits = expand_bits(bits, B)

    # Simulate: start from a protected latent z_hat_k (output of keyed transform on z_hat)
    z_hat = torch.randn(B, latent_channels, latent_size, latent_size)
    z_hat_k = klb(z_hat)  # protected

    # Step 1: invert key transform
    z_hat_inv = klb.invert(z_hat_k)  # should recover z_hat

    # Step 2: decode with gradient path
    x_dec = autoencoder.decode_with_grad(z_hat_inv)
    assert x_dec.shape == (B, channels, image_size, image_size), (
        f"x_dec shape: {tuple(x_dec.shape)}"
    )

    # Step 3: apply watermark adapter
    residual = adapter(x_dec, batch_bits)
    x_w = torch.clamp(x_dec + alpha * residual, -1.0, 1.0)
    assert x_w.shape == (B, channels, image_size, image_size), (
        f"x_w shape: {tuple(x_w.shape)}"
    )

    # Step 4: image detector
    logits_img = image_detector.logits(x_w)
    assert logits_img.shape == (B, bit_length), (
        f"image detector logits shape: {tuple(logits_img.shape)}"
    )
    probs_img = image_detector(x_w)
    assert probs_img.shape == (B, bit_length), (
        f"image detector probs shape: {tuple(probs_img.shape)}"
    )
    assert probs_img.min().item() >= 0.0 and probs_img.max().item() <= 1.0
    print(f"[test] image detector shape/range: PASS  bit_acc={bit_accuracy(probs_img, batch_bits):.4f}")

    # Step 5: re-encode watermarked image with gradient path
    z_re = autoencoder.encode_with_grad(x_w)
    assert z_re.shape == (B, latent_channels, latent_size, latent_size), (
        f"z_re shape: {tuple(z_re.shape)}"
    )

    # Step 6: apply key transform to re-encoded latent
    z_re_k = klb(z_re)
    assert z_re_k.shape == (B, latent_channels, latent_size, latent_size), (
        f"z_re_k shape: {tuple(z_re_k.shape)}"
    )

    # Step 7: latent detector
    logits_lat = latent_detector.logits(z_re_k)
    assert logits_lat.shape == (B, bit_length), (
        f"latent detector logits shape: {tuple(logits_lat.shape)}"
    )
    probs_lat = latent_detector(z_re_k)
    assert probs_lat.shape == (B, bit_length), (
        f"latent detector probs shape: {tuple(probs_lat.shape)}"
    )
    assert probs_lat.min().item() >= 0.0 and probs_lat.max().item() <= 1.0
    print(f"[test] latent detector shape/range: PASS  bit_acc={bit_accuracy(probs_lat, batch_bits):.4f}")

    print("[test] Full TraceFlow pipeline shapes: PASS")

    # ------------------------------------------------------------------
    # 4. Gradient flow through the combined TraceFlow loss
    # ------------------------------------------------------------------
    # All trainable modules
    adapter.train()
    image_detector.train()
    latent_detector.train()
    autoencoder.eval()  # frozen AE

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(adapter.parameters())
        + list(image_detector.parameters())
        + list(latent_detector.parameters()),
        lr=1e-3,
    )
    optimizer.zero_grad()

    # z_hat_k is the "input protected latent" that we also want gradients w.r.t.
    z_hat_k_grad = torch.randn(
        B, latent_channels, latent_size, latent_size, requires_grad=True
    )

    # Forward pass (same pipeline)
    z_inv = klb.invert(z_hat_k_grad)
    x_d = autoencoder.decode_with_grad(z_inv)
    res = adapter(x_d, batch_bits)
    x_wm = torch.clamp(x_d + alpha * res, -1.0, 1.0)
    logits_img = image_detector.logits(x_wm)
    z_r = autoencoder.encode_with_grad(x_wm)
    z_r_k = klb(z_r)
    logits_lat = latent_detector.logits(z_r_k)

    # TraceFlow combined loss
    lambda_wm_img = cfg["lambda_wm_img"]
    lambda_wm_latent = cfg["lambda_wm_latent"]
    lambda_img = cfg["lambda_img"]
    lambda_cycle = cfg["lambda_cycle"]
    lambda_residual = cfg["lambda_residual"]

    loss_img_wm = bce(logits_img, batch_bits)
    loss_lat_wm = bce(logits_lat, batch_bits)
    loss_img = (x_wm - x_d.detach()).pow(2).mean()
    loss_cycle = (z_r_k - z_hat_k_grad.detach()).pow(2).mean()
    loss_res = res.pow(2).mean()

    loss = (
        lambda_wm_img * loss_img_wm
        + lambda_wm_latent * loss_lat_wm
        + lambda_img * loss_img
        + lambda_cycle * loss_cycle
        + lambda_residual * loss_res
    )

    assert math.isfinite(loss.item()), f"TraceFlow combined loss not finite: {loss.item()}"
    loss.backward()

    # Check parameter gradients
    adapter_gn = _grad_norm(adapter.parameters())
    img_det_gn = _grad_norm(image_detector.parameters())
    lat_det_gn = _grad_norm(latent_detector.parameters())

    assert adapter_gn > 0.0, f"no gradients into adapter (norm={adapter_gn})"
    assert img_det_gn > 0.0, f"no gradients into image_detector (norm={img_det_gn})"
    assert lat_det_gn > 0.0, f"no gradients into latent_detector (norm={lat_det_gn})"

    # Check gradient w.r.t. input protected latent
    assert z_hat_k_grad.grad is not None, "z_hat_k_grad.grad is None"
    z_grad_norm = z_hat_k_grad.grad.detach().norm().item()
    assert z_grad_norm > 0.0, f"no gradient flows to z_hat_k (norm={z_grad_norm})"
    assert math.isfinite(z_grad_norm), f"z_hat_k gradient not finite: {z_grad_norm}"

    optimizer.step()

    print(
        f"[test] TraceFlow gradient flow: loss={loss.item():.4f} "
        f"adapter={adapter_gn:.3e} img_det={img_det_gn:.3e} "
        f"lat_det={lat_det_gn:.3e} z_hat_k={z_grad_norm:.3e}: PASS"
    )

    # ------------------------------------------------------------------
    # 4b. Raw detection does NOT call decoder_adapter; post-watermark does.
    # ------------------------------------------------------------------
    from scripts.eval_traceflow_inversion import _detect_raw, _detect_post_watermark

    adapter.eval()
    image_detector.eval()
    latent_detector.eval()

    eval_wm = {
        "enabled": True,
        "type": "traceflow",
        "bits": bits,
        "extractor": image_detector,
        "decoder_adapter": adapter,
        "latent_detector": latent_detector,
        "config": cfg,
    }

    # Spy on the adapter: count forward calls.
    call_counter = {"n": 0}
    orig_adapter_forward = adapter.forward

    def _counting_forward(*a, **kw):
        call_counter["n"] += 1
        return orig_adapter_forward(*a, **kw)

    adapter.forward = _counting_forward  # type: ignore[method-assign]

    x_attack = torch.randn(B, channels, image_size, image_size).tanh()
    device = torch.device("cpu")

    try:
        # RAW detection must NOT touch the decoder_adapter.
        before = call_counter["n"]
        raw_metrics = _detect_raw(
            eval_wm, klb, autoencoder, x_attack, batch_bits, device
        )
        assert call_counter["n"] == before, (
            f"RAW detection called decoder_adapter {call_counter['n'] - before} time(s) — "
            "it must not re-stamp the watermark."
        )
        for k in ("image_bit_acc", "image_ber", "latent_bit_acc", "latent_ber"):
            assert k in raw_metrics, f"raw metrics missing {k!r}"

        # POST-WATERMARK detection MUST call the decoder_adapter exactly once.
        before = call_counter["n"]
        post_metrics, x_wm = _detect_post_watermark(
            eval_wm, klb, autoencoder, x_attack, batch_bits, device
        )
        assert call_counter["n"] == before + 1, (
            f"post_watermark detection called decoder_adapter "
            f"{call_counter['n'] - before} time(s); expected exactly 1."
        )
        assert x_wm.shape == (B, channels, image_size, image_size), (
            f"post_watermark image shape: {tuple(x_wm.shape)}"
        )
        for k in ("image_bit_acc", "image_ber", "latent_bit_acc", "latent_ber"):
            assert k in post_metrics, f"post metrics missing {k!r}"
    finally:
        adapter.forward = orig_adapter_forward  # type: ignore[method-assign]

    print(
        "[test] raw detection skips decoder_adapter / post_watermark applies it once: PASS"
    )

    # ------------------------------------------------------------------
    # 4c. Field-name separation: raw_* and post_watermark_* keys are distinct.
    # ------------------------------------------------------------------
    raw_named = {f"raw_pixel_{k}": v for k, v in raw_metrics.items()}
    post_named = {f"post_watermark_pixel_{k}": v for k, v in post_metrics.items()}
    assert set(raw_named).isdisjoint(set(post_named)), (
        "raw_* and post_watermark_* metric field names must not collide"
    )
    assert all(k.startswith("raw_") for k in raw_named)
    assert all(k.startswith("post_watermark_") for k in post_named)
    print("[test] raw_* vs post_watermark_* metric naming separation: PASS")

    # ------------------------------------------------------------------
    # 5. generate_random_batch_bits and message_mode
    # ------------------------------------------------------------------
    from src.watermarking.message import generate_random_batch_bits

    # 5a. generate_random_batch_bits produces non-identical rows.
    rand_bits = generate_random_batch_bits(bit_length, B)
    assert rand_bits.shape == (B, bit_length), (
        f"generate_random_batch_bits shape: {tuple(rand_bits.shape)} != ({B}, {bit_length})"
    )
    assert rand_bits.dtype == torch.float32, "generate_random_batch_bits should return float32"
    assert rand_bits.min().item() >= 0.0 and rand_bits.max().item() <= 1.0, (
        "generate_random_batch_bits values must be in {0, 1}"
    )
    # With B=4 and bit_length=32, the probability all rows are identical is ~1/(2^32)^3
    all_same = all(torch.equal(rand_bits[i], rand_bits[0]) for i in range(1, B))
    assert not all_same, (
        "generate_random_batch_bits: all rows are identical — random_per_sample broken"
    )
    print(f"[test] generate_random_batch_bits non-identical rows: PASS")

    # 5b. factory resolved config includes message_mode='random_per_sample' by default.
    assert cfg.get("message_mode") == "random_per_sample", (
        f"traceflow default message_mode should be 'random_per_sample', "
        f"got {cfg.get('message_mode')!r}"
    )
    # Explicit override to 'fixed_owner_bits' is respected.
    _inner_override = dict(wm_cfg["watermark"])
    _inner_override["message_mode"] = "fixed_owner_bits"
    wm_fixed = build_watermark_modules(
        {"watermark": _inner_override}, image_size=image_size, channels=channels
    )
    assert wm_fixed["config"]["message_mode"] == "fixed_owner_bits", (
        f"message_mode override not respected: {wm_fixed['config']['message_mode']!r}"
    )
    print("[test] message_mode in factory resolved config: PASS")

    # 5c. Full traceflow pipeline works with random_per_sample (non-identical) bits.
    adapter.eval()
    image_detector.eval()
    latent_detector.eval()
    autoencoder.eval()

    rand_batch_bits_c = generate_random_batch_bits(bit_length, B)
    assert not all(torch.equal(rand_batch_bits_c[i], rand_batch_bits_c[0]) for i in range(1, B)), (
        "Test requires non-identical batch bits for random_per_sample coverage"
    )

    with torch.no_grad():
        z_test = torch.randn(B, latent_channels, latent_size, latent_size)
        z_test_k = klb(z_test)
        z_test_inv = klb.invert(z_test_k)
        x_test = autoencoder.decode_with_grad(z_test_inv)
        res_test = adapter(x_test, rand_batch_bits_c)
        x_wm_test = torch.clamp(x_test + alpha * res_test, -1.0, 1.0)
        logits_img_test = image_detector.logits(x_wm_test)
        p_img_test = torch.sigmoid(logits_img_test)
        z_re_test = autoencoder.encode_with_grad(x_wm_test)
        z_re_k_test = klb(z_re_test)
        logits_lat_test = latent_detector.logits(z_re_k_test)
        p_lat_test = torch.sigmoid(logits_lat_test)

    assert p_img_test.shape == (B, bit_length), (
        f"random_per_sample image detector shape {tuple(p_img_test.shape)} != ({B}, {bit_length})"
    )
    assert p_lat_test.shape == (B, bit_length), (
        f"random_per_sample latent detector shape {tuple(p_lat_test.shape)} != ({B}, {bit_length})"
    )
    # Loss must be computable with per-sample non-identical bits.
    bce_test = nn.BCEWithLogitsLoss()
    loss_test = bce_test(logits_img_test, rand_batch_bits_c) + bce_test(logits_lat_test, rand_batch_bits_c)
    assert math.isfinite(loss_test.item()), (
        f"loss not finite with random_per_sample bits: {loss_test.item()}"
    )
    print(
        f"[test] TraceFlow pipeline with random_per_sample bits: PASS  "
        f"loss={loss_test.item():.4f}"
    )

    # CUDA bf16 regression guard for the production training path. PyTorch
    # rejects sigmoid+BCELoss under autocast, so TraceFlow must keep training
    # losses on logits with BCEWithLogitsLoss.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        img_det_cuda = ImageWatermarkDetector(
            bit_length=bit_length,
            image_size=image_size,
            channels=channels,
            hidden_dim=hidden_dim,
        ).to(device)
        lat_det_cuda = TraceLatentDetector(
            bit_length=bit_length,
            latent_channels=latent_channels,
            hidden_dim=latent_detector_hidden_dim,
        ).to(device)
        x_cuda = torch.randn(B, channels, image_size, image_size, device=device)
        z_cuda = torch.randn(B, latent_channels, latent_size, latent_size, device=device)
        bits_cuda = generate_random_batch_bits(bit_length, B, device=device)
        bce_logits = nn.BCEWithLogitsLoss()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_amp = bce_logits(img_det_cuda.logits(x_cuda), bits_cuda)
            loss_amp = loss_amp + bce_logits(lat_det_cuda.logits(z_cuda), bits_cuda)
        assert math.isfinite(loss_amp.item()), f"bf16 logits BCE loss not finite: {loss_amp.item()}"
        print(f"[test] bf16 autocast logits BCE path: PASS  loss={loss_amp.item():.4f}")
    else:
        print("[test] bf16 autocast logits BCE path: SKIP (CUDA unavailable)")

    print("[test] ── ALL TRACEFLOW WATERMARK SMOKE CHECKS PASSED ──")


if __name__ == "__main__":
    main()
