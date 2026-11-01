"""Triton AOT kernel for direct 2-D convolution."""

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
    """Direct NCHW convolution; one output element per lane.

    Launch grid: ``(cdiv(total, BLOCK),)`` where ``total = N * out_c * out_h * out_w``.

    Args:
        x: Pointer to the contiguous ``[n_dim, in_c, h_dim, w_dim]`` input.
        w: Pointer to the contiguous ``[out_c, in_c, k_h, k_w]`` weight.
        bias: Pointer to the ``[out_c]`` bias; ignored when ``has_bias`` is false.
        out: Pointer to the contiguous ``[n_dim, out_c, out_h, out_w]`` output.
        total: Number of output elements.
        n_dim: Batch size.
        h_dim: Input height.
        w_dim: Input width.
        out_c: Number of output channels.
        out_h: Output height.
        out_w: Output width.
        stride_h: Vertical stride.
        stride_w: Horizontal stride.
        pad_h: Vertical zero padding.
        pad_w: Horizontal zero padding.
        in_c: Number of input channels (unrolled reduction bound).
        k_h: Kernel height.
        k_w: Kernel width.
        has_bias: Whether ``bias`` is added to the result.
        BLOCK: Number of output elements per program instance.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Decompose the flat output offset into NCHW coordinates.
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
                # Out-of-bounds taps contribute zero (implicit zero padding).
                valid = mask & (batch < n_dim) & (ih >= 0) & (ih < h_dim) & (iw >= 0) & (iw < w_dim)
                xv = tl.load(
                    x + ((batch * in_c + ic) * h_dim + ih) * w_dim + iw, mask=valid, other=0.0
                )
                wv = tl.load(w + ((oc * in_c + ic) * k_h + kh) * k_w + kw, mask=mask, other=0.0)
                acc += xv * wv
    tl.store(out + offs, acc, mask=mask)
