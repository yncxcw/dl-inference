"""Triton AOT kernel for element-wise SiLU (swish)."""

import triton
import triton.language as tl


@triton.jit
def silu_kernel(x, out, total, BLOCK: tl.constexpr):
    """Computes ``out[i] = x[i] * sigmoid(x[i])`` over a flattened tensor.

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
    # x * sigmoid(x) written as x / (1 + exp(-x)).
    tl.store(out + offs, values / (1.0 + tl.exp(-values)), mask=mask)
