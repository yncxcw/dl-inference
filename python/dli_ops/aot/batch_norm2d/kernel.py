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
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    channel = (offs // spatial) % channels
    values = tl.load(x + offs, mask=mask, other=0.0)
    scale = tl.load(weight + channel, mask=mask, other=0.0)
    offset = tl.load(bias + channel, mask=mask, other=0.0)
    mean = tl.load(running_mean + channel, mask=mask, other=0.0)
    variance = tl.load(running_var + channel, mask=mask, other=0.0)
    normalized = (values - mean) * tl.rsqrt(variance + eps)
    tl.store(out + offs, normalized * scale + offset, mask=mask)
