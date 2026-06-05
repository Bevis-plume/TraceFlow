"""
src/watermarking/message.py
============================
Watermark message (bit-string) utilities.

The watermark message is a fixed bit vector of length ``bit_length``.  For
reproducibility the bits are derived deterministically from an integer
``seed`` so that the *same* message can be regenerated at sampling time
without storing the bits in the checkpoint.

Security note
-------------
The watermark seed/bits are NOT secret keys.  Do not store secret keys in
checkpoints.  For research-controlled smoke/dev runs it is acceptable to save
the fixed watermark bits in the checkpoint (``save_bits=true``); for "real"
runs the bits can instead be regenerated from the seed.
"""

from __future__ import annotations

from typing import Optional

import torch


def generate_watermark_bits(
    bit_length: int,
    seed: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Deterministically generate a fixed watermark bit vector.

    Args:
        bit_length:  Number of bits in the message.
        seed:        Integer seed; the same seed always yields the same bits.
        device:      Optional target device for the returned tensor.

    Returns:
        Float tensor of shape ``[bit_length]`` with 0/1 values.
    """
    if bit_length <= 0:
        raise ValueError(f"bit_length must be positive, got {bit_length}.")

    # Use a CPU generator for cross-device determinism, then move to device.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    bits = torch.randint(0, 2, (bit_length,), generator=gen, dtype=torch.float32)
    if device is not None:
        bits = bits.to(device)
    return bits


def generate_random_batch_bits(
    bit_length: int,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate independent random bit messages for each sample in a batch.

    Unlike :func:`generate_watermark_bits`, which produces a *single* fixed owner
    message, this function generates ``batch_size`` independent random messages.
    Use this for ``random_per_sample`` training so every sample is conditioned on
    a different watermark message, preventing the detector from short-circuiting
    to a constant output.

    Args:
        bit_length:  Number of bits per message.
        batch_size:  Number of independent messages to generate.
        generator:   Optional :class:`torch.Generator` for reproducibility.
                     Uses the global PyTorch RNG when ``None``.
        device:      Optional target device for the returned tensor.

    Returns:
        Float tensor of shape ``[batch_size, bit_length]`` with 0/1 values.
        Each row is drawn independently and uniformly at random.
    """
    if bit_length <= 0:
        raise ValueError(f"bit_length must be positive, got {bit_length}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    bits = torch.randint(
        0, 2, (batch_size, bit_length),
        generator=generator,
        dtype=torch.float32,
    )
    if device is not None:
        bits = bits.to(device)
    return bits


def expand_bits(bits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast a single ``[bit_length]`` message to ``[B, bit_length]``.

    Args:
        bits:        Float tensor of shape ``[bit_length]`` (or already ``[B, bit_length]``).
        batch_size:  Target batch size B.

    Returns:
        Float tensor of shape ``[batch_size, bit_length]``.
    """
    if bits.dim() == 2:
        if bits.size(0) == batch_size:
            return bits
        if bits.size(0) == 1:
            return bits.expand(batch_size, -1).contiguous()
        raise ValueError(
            f"Cannot expand bits of batch {bits.size(0)} to batch_size {batch_size}."
        )
    if bits.dim() != 1:
        raise ValueError(f"bits must be 1-D or 2-D, got shape {tuple(bits.shape)}.")
    return bits.unsqueeze(0).expand(batch_size, -1).contiguous()
