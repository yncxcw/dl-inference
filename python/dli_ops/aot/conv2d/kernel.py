import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x,
    w,
    bias,
    out,
    total,
    n_dim,
    h_dim,
    w_dim,
    out_c,
    out_h,
    out_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    in_c: tl.constexpr,
    k_h: tl.constexpr,
    k_w: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % out_w
    oh = (offs // out_w) % out_h
    oc = (offs // (out_w * out_h)) % out_c
    batch = offs // (out_w * out_h * out_c)
    acc = tl.zeros((BLOCK,), tl.float32)
    if has_bias:
        acc += tl.load(bias + oc, mask=mask, other=0.0)
    for ic in range(0, in_c):
        for kh in range(0, k_h):
            for kw in range(0, k_w):
                ih = oh * stride_h + kh - pad_h
                iw = ow * stride_w + kw - pad_w
                valid = mask & (batch < n_dim) & (ih >= 0) & (ih < h_dim) & (iw >= 0) & (iw < w_dim)
                xv = tl.load(
                    x + ((batch * in_c + ic) * h_dim + ih) * w_dim + iw, mask=valid, other=0.0
                )
                wv = tl.load(w + ((oc * in_c + ic) * k_h + kh) * k_w + kw, mask=mask, other=0.0)
                acc += xv * wv
    tl.store(out + offs, acc, mask=mask)
