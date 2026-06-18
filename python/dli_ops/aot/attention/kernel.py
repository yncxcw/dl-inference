import triton
import triton.language as tl


@triton.jit
def attention_kernel(q, k, v, out,
                     seq_q: tl.constexpr, seq_k: tl.constexpr,
                     head_dim: tl.constexpr, causal: tl.constexpr,
                     scale: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_D: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_base = pid_bh * seq_q * head_dim
    kv_base = pid_bh * seq_k * head_dim
    q_block = tl.load(q + q_base + offs_m[:, None] * head_dim + offs_d[None, :],
                      mask=(offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim),
                      other=0.0)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    past = seq_k - seq_q
    for start_n in range(0, seq_k, BLOCK_N):
        n = start_n + offs_n
        k_block = tl.load(k + kv_base + n[None, :] * head_dim + offs_d[:, None],
                          mask=(n[None, :] < seq_k) & (offs_d[:, None] < head_dim),
                          other=0.0)
        v_block = tl.load(v + kv_base + n[:, None] * head_dim + offs_d[None, :],
                          mask=(n[:, None] < seq_k) & (offs_d[None, :] < head_dim),
                          other=0.0)
        scores = tl.dot(q_block, k_block) * scale
        if causal:
            scores = tl.where(n[None, :] <= past + offs_m[:, None], scores, -float("inf"))
        m_next = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_next[:, None])
        alpha = tl.exp(m_i - m_next)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_next
    out_block = acc / l_i[:, None]
    tl.store(out + q_base + offs_m[:, None] * head_dim + offs_d[None, :],
             out_block,
             mask=(offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim))
