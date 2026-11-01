"""Triton AOT kernel for inference-mode 2-D batch normalization."""

import triton
import triton.language as tl


@triton.jit
def batch_norm2d_kernel(
    x,
    weight,
    bias,
    running_mean,
    running_var,
    out,
    total,
    channels,
    spatial,
    eps,
    BLOCK: tl.constexpr,
):
    """Normalizes an NCHW tensor with precomputed running statistics.

    Computes ``(x - running_mean) * rsqrt(running_var + eps) * weight + bias``
    per channel. Launch grid: ``(cdiv(total, BLOCK),)``.

    Args:
        x: Pointer to the contiguous ``[N, channels, H, W]`` input.
        weight: Pointer to the ``[channels]`` scale.
        bias: Pointer to the ``[channels]`` shift.
        running_mean: Pointer to the ``[channels]`` running mean.
        running_var: Pointer to the ``[channels]`` running variance.
        out: Pointer to the contiguous output, same shape as ``x``.
        total: Number of elements in ``x``.
        channels: Number of channels ``C``.
        spatial: Elements per channel plane (``H * W``).
        eps: Small constant added to the variance for stability.
        BLOCK: Number of elements per program instance.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Recover the channel index from the flat NCHW offset.
    channel = (offs // spatial) % channels
    values = tl.load(x + offs, mask=mask, other=0.0)
    scale = tl.load(weight + channel, mask=mask, other=0.0)
    offset = tl.load(bias + channel, mask=mask, other=0.0)
    mean = tl.load(running_mean + channel, mask=mask, other=0.0)
    variance = tl.load(running_var + channel, mask=mask, other=0.0)
    normalized = (values - mean) * tl.rsqrt(variance + eps)
    tl.store(out + offs, normalized * scale + offset, mask=mask)
