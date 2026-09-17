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
    q_total_pairs,
    k_total_pairs,
    start_pos,
    head_dim: tl.constexpr,
    seq: tl.constexpr,
    rotary_pairs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q_mask = offs < q_total_pairs
    k_mask = offs < k_total_pairs
    mask = q_mask | k_mask
    pair = offs % rotary_pairs
    token = (offs // rotary_pairs) % seq
    vector = offs // rotary_pairs
    base = vector * head_dim
    c = tl.load(cos + (start_pos + token) * rotary_pairs + pair, mask=mask, other=1.0)
    s = tl.load(sin + (start_pos + token) * rotary_pairs + pair, mask=mask, other=0.0)
    # Qwen uses rotate-half RoPE: dimension i in the rotary prefix is paired
    # with i + rotary_pairs, rather than pairing adjacent dimensions.
    q0 = tl.load(q + base + pair, mask=q_mask, other=0.0)
    q1 = tl.load(q + base + pair + rotary_pairs, mask=q_mask, other=0.0)
    k0 = tl.load(k + base + pair, mask=k_mask, other=0.0)
    k1 = tl.load(k + base + pair + rotary_pairs, mask=k_mask, other=0.0)
    tl.store(out_q + base + pair, q0 * c - q1 * s, mask=q_mask)
    tl.store(out_q + base + pair + rotary_pairs, q0 * s + q1 * c, mask=q_mask)
    tl.store(out_k + base + pair, k0 * c - k1 * s, mask=k_mask)
    tl.store(out_k + base + pair + rotary_pairs, k0 * s + k1 * c, mask=k_mask)
