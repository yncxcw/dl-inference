"""Triton AOT kernel for a linear (fully connected) layer."""

import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x,
    weight,
    bias,
    out,
    m,
    n,
    k: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Computes ``out = x @ weight.T (+ bias)`` with fp32 accumulation.

    Launch grid: ``(cdiv(m, BLOCK_M), cdiv(n, BLOCK_N))``.

    Args:
        x: Pointer to the contiguous ``[m, k]`` input.
        weight: Pointer to the contiguous ``[n, k]`` weight (PyTorch layout).
        bias: Pointer to the ``[n]`` bias; ignored when ``has_bias`` is false.
        out: Pointer to the contiguous ``[m, n]`` output.
        m: Number of input rows.
        n: Number of output features.
        k: Number of input features (reduction length).
        has_bias: Whether ``bias`` is added to the result.
        BLOCK_M: Tile size along ``m``.
        BLOCK_N: Tile size along ``n``.
        BLOCK_K: Tile size along the reduction dimension ``k``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, k, BLOCK_K):
        kk = start + offs_k
        a = tl.load(
            x + offs_m[:, None] * k + kk[None, :],
            mask=(offs_m[:, None] < m) & (kk[None, :] < k),
            other=0.0,
        )
        # Weight is [n, k]; load it transposed as a [BLOCK_K, BLOCK_N] tile for tl.dot.
        b = tl.load(
            weight + offs_n[None, :] * k + kk[:, None],
            mask=(offs_n[None, :] < n) & (kk[:, None] < k),
            other=0.0,
        )
        acc += tl.dot(a, b)
    if has_bias:
        acc += tl.load(bias + offs_n, mask=offs_n < n, other=0.0)[None, :]
    tl.store(
        out + offs_m[:, None] * n + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )
