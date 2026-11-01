"""Triton AOT kernel for 2-D matrix transpose."""

import triton
import triton.language as tl


@triton.jit
def transpose2d_kernel(x, out, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Writes the transpose of a row-major ``[rows, cols]`` matrix.

    Launch grid: ``(cdiv(rows, BLOCK_M), cdiv(cols, BLOCK_N))``.

    Args:
        x: Pointer to the contiguous ``[rows, cols]`` input.
        out: Pointer to the contiguous ``[cols, rows]`` output.
        rows: Number of rows of ``x``.
        cols: Number of columns of ``x``.
        BLOCK_M: Tile height along ``rows``.
        BLOCK_N: Tile width along ``cols``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    values = tl.load(
        x + offs_m[:, None] * cols + offs_n[None, :],
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < cols),
        other=0.0,
    )
    tl.store(
        out + offs_n[None, :] * rows + offs_m[:, None],
        values,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < cols),
    )
