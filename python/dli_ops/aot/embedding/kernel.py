import triton
import triton.language as tl


@triton.jit
def embedding_kernel(ids, table, out, total, hidden: tl.constexpr,
                     BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    token_ids = tl.load(ids + offs_t, mask=offs_t < total, other=0)
    values = tl.load(table + token_ids[:, None] * hidden + offs_d[None, :],
                     mask=(offs_t[:, None] < total) & (offs_d[None, :] < hidden),
                     other=0.0)
    tl.store(out + offs_t[:, None] * hidden + offs_d[None, :],
             values,
             mask=(offs_t[:, None] < total) & (offs_d[None, :] < hidden))
