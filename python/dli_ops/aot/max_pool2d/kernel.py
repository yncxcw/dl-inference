"""Triton AOT kernel for 2-D max pooling."""

import triton
import triton.language as tl


@triton.jit
def max_pool2d_kernel(
    x,
    out,
    total,
    n_dim,
    c_dim,
    h_dim,
    w_dim,
    out_h,
    out_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    k_h: tl.constexpr,
    k_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Max-pools an NCHW tensor; one output element per lane.

    Launch grid: ``(cdiv(total, BLOCK),)`` where ``total = N * C * out_h * out_w``.

    Args:
        x: Pointer to the contiguous ``[n_dim, c_dim, h_dim, w_dim]`` input.
        out: Pointer to the contiguous ``[n_dim, c_dim, out_h, out_w]`` output.
        total: Number of output elements.
        n_dim: Batch size.
        c_dim: Number of channels.
        h_dim: Input height.
        w_dim: Input width.
        out_h: Output height.
        out_w: Output width.
        stride_h: Vertical stride.
        stride_w: Horizontal stride.
        pad_h: Vertical padding (implicit ``-inf`` padding).
        pad_w: Horizontal padding (implicit ``-inf`` padding).
        k_h: Pooling window height.
        k_w: Pooling window width.
        BLOCK: Number of output elements per program instance.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Decompose the flat output offset into NCHW coordinates.
    ow = offs % out_w
    oh = (offs // out_w) % out_h
    c = (offs // (out_w * out_h)) % c_dim
    batch = offs // (out_w * out_h * c_dim)
    acc = tl.full((BLOCK,), -float("inf"), tl.float32)
    for kh in range(0, k_h):
        for kw in range(0, k_w):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            # Padded taps load -inf so they never win the max.
            valid = mask & (batch < n_dim) & (ih >= 0) & (ih < h_dim) & (iw >= 0) & (iw < w_dim)
            values = tl.load(
                x + ((batch * c_dim + c) * h_dim + ih) * w_dim + iw, mask=valid, other=-float("inf")
            )
            acc = tl.maximum(acc, values)
    tl.store(out + offs, acc, mask=mask)
