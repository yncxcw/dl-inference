import triton
import triton.language as tl


@triton.jit
def add_kernel(a, b, out, total, width, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    av = tl.load(a + offs, mask=mask, other=0.0)
    bv = tl.load(b + (offs % width), mask=mask, other=0.0)
    tl.store(out + offs, av + bv, mask=mask)
