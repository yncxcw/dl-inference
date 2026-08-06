import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x, weight, out, rows, eps, hidden: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < hidden
    values = tl.load(x + row * hidden + offs, mask=mask, other=0.0)
    w = tl.load(weight + offs, mask=mask, other=0.0)
    mean_square = tl.sum(values * values, axis=0) / hidden
    normed = values * tl.rsqrt(mean_square + eps) * w
    tl.store(out + row * hidden + offs, normed, mask=mask)
