import triton
import triton.language as tl


@triton.jit
def relu_kernel(x, out, total, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    values = tl.load(x + offs, mask=mask, other=0.0)
    tl.store(out + offs, tl.maximum(values, 0.0), mask=mask)
