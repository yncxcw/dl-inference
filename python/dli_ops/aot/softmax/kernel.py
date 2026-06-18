import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x, out, rows, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < width
    values = tl.load(x + row * width + offs, mask=mask, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out + row * width + offs, numerator / denominator, mask=mask)
