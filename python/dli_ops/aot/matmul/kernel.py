import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a, b, out, m, n,
                  k: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, k, BLOCK_K):
        kk = start + offs_k
        av = tl.load(a + offs_m[:, None] * k + kk[None, :],
                     mask=(offs_m[:, None] < m) & (kk[None, :] < k),
                     other=0.0)
        bv = tl.load(b + kk[:, None] * n + offs_n[None, :],
                     mask=(kk[:, None] < k) & (offs_n[None, :] < n),
                     other=0.0)
        acc += tl.dot(av, bv)
    tl.store(out + offs_m[:, None] * n + offs_n[None, :],
             acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))
