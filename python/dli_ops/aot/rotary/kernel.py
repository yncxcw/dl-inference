import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    q,
    k,
    cos,
    sin,
    out_q,
    out_k,
    total_pairs,
    start_pos,
    head_dim: tl.constexpr,
    seq: tl.constexpr,
    rotary_pairs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_pairs
    pair = offs % rotary_pairs
    token = (offs // rotary_pairs) % seq
    vector = offs // rotary_pairs
    base = vector * head_dim + pair * 2
    c = tl.load(cos + (start_pos + token) * rotary_pairs + pair, mask=mask, other=1.0)
    s = tl.load(sin + (start_pos + token) * rotary_pairs + pair, mask=mask, other=0.0)
    q0 = tl.load(q + base, mask=mask, other=0.0)
    q1 = tl.load(q + base + 1, mask=mask, other=0.0)
    k0 = tl.load(k + base, mask=mask, other=0.0)
    k1 = tl.load(k + base + 1, mask=mask, other=0.0)
    tl.store(out_q + base, q0 * c - q1 * s, mask=mask)
    tl.store(out_q + base + 1, q0 * s + q1 * c, mask=mask)
    tl.store(out_k + base, k0 * c - k1 * s, mask=mask)
    tl.store(out_k + base + 1, k0 * s + k1 * c, mask=mask)
