"""Triton AOT kernel for element-wise ReLU."""

import triton
import triton.language as tl


@triton.jit
def relu_kernel(x, out, total, BLOCK: tl.constexpr):
    """Computes ``out[i] = max(x[i], 0)`` over a flattened tensor.

    Launch grid: ``(cdiv(total, BLOCK),)``.

    Args:
        x: Pointer to the contiguous input of ``total`` elements.
        out: Pointer to the contiguous output of ``total`` elements.
        total: Number of elements in ``x`` and ``out``.
        BLOCK: Number of elements handled by each program instance.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    values = tl.load(x + offs, mask=mask, other=0.0)
    tl.store(out + offs, tl.maximum(values, 0.0), mask=mask)
