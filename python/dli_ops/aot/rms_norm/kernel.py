"""Triton AOT kernel for RMS normalization."""

import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x, weight, out, rows, eps, hidden: tl.constexpr, BLOCK_D: tl.constexpr):
    """Computes ``x / sqrt(mean(x^2) + eps) * weight`` for each row.

    Each program instance handles one whole row, so ``BLOCK_D`` must be at
    least ``hidden``. Launch grid: ``(rows,)``.

    Args:
        x: Pointer to the contiguous ``[rows, hidden]`` input.
        weight: Pointer to the ``[hidden]`` scale vector.
        out: Pointer to the contiguous ``[rows, hidden]`` output.
        rows: Number of rows (product of all leading dimensions).
        eps: Small constant added to the mean square for stability.
        hidden: Size of the normalized dimension.
        BLOCK_D: Power-of-two tile width covering one row (>= ``hidden``).
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < hidden
    values = tl.load(x + row * hidden + offs, mask=mask, other=0.0)
    w = tl.load(weight + offs, mask=mask, other=0.0)
    # Masked lanes load 0, so they do not perturb the sum; divide by true ``hidden``.
    mean_square = tl.sum(values * values, axis=0) / hidden
    normed = values * tl.rsqrt(mean_square + eps) * w
    tl.store(out + row * hidden + offs, normed, mask=mask)
