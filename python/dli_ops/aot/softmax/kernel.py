"""Triton AOT kernel for row-wise softmax."""

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x, out, rows, width, BLOCK: tl.constexpr):
    """Computes a numerically stable softmax over the last dimension.

    Each program instance handles one whole row, so ``BLOCK`` must be at least
    ``width``. Launch grid: ``(rows,)``.

    Args:
        x: Pointer to the contiguous ``[rows, width]`` input.
        out: Pointer to the contiguous ``[rows, width]`` output.
        rows: Number of rows (product of all leading dimensions).
        width: Size of the softmax dimension.
        BLOCK: Power-of-two tile width covering one row (>= ``width``).
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < width
    values = tl.load(x + row * width + offs, mask=mask, other=-float("inf"))
    # Subtract the row max so exp() cannot overflow; masked lanes are -inf -> 0.
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out + row * width + offs, numerator / denominator, mask=mask)
