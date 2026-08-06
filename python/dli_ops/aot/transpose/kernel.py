import triton
import triton.language as tl


@triton.jit
def transpose2d_kernel(x, out, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
