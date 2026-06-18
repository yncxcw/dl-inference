import triton
import triton.language as tl


@triton.jit
def max_pool2d_kernel(x, out, total, n_dim, c_dim, h_dim, w_dim, out_h, out_w,
                      stride_h, stride_w, pad_h, pad_w,
                      k_h: tl.constexpr, k_w: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % out_w
    oh = (offs // out_w) % out_h
    c = (offs // (out_w * out_h)) % c_dim
    batch = offs // (out_w * out_h * c_dim)
    acc = tl.full((BLOCK,), -float("inf"), tl.float32)
    for kh in range(0, k_h):
        for kw in range(0, k_w):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (batch < n_dim) & (ih >= 0) & (ih < h_dim) & (iw >= 0) & (iw < w_dim)
            values = tl.load(x + ((batch * c_dim + c) * h_dim + ih) * w_dim + iw,
                             mask=valid, other=-float("inf"))
            acc = tl.maximum(acc, values)
    tl.store(out + offs, acc, mask=mask)
