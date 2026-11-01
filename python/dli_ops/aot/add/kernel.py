"""Triton AOT kernel for broadcast element-wise addition."""

import triton
import triton.language as tl


@triton.jit
def add_kernel(a, b, out, total, width, BLOCK: tl.constexpr):
    """Computes ``out[i] = a[i] + b[i % width]`` over a flattened tensor.

    The second operand is broadcast along the trailing dimension, so ``b`` is a
    1-D tensor of ``width`` elements while ``a`` and ``out`` are contiguous and
    flattened to ``total`` elements. Launch grid: ``(cdiv(total, BLOCK),)``.

    Args:
        a: Pointer to the contiguous input of ``total`` elements.
        b: Pointer to the broadcast operand of ``width`` elements.
        out: Pointer to the contiguous output of ``total`` elements.
        total: Number of elements in ``a`` and ``out``.
        width: Number of elements in ``b`` (size of the broadcast dimension).
        BLOCK: Number of elements handled by each program instance.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    av = tl.load(a + offs, mask=mask, other=0.0)
    # ``b`` is broadcast over the trailing dimension, hence the modulo.
    bv = tl.load(b + (offs % width), mask=mask, other=0.0)
    tl.store(out + offs, av + bv, mask=mask)
