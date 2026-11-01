"""Triton AOT kernel for rotary position embeddings (RoPE)."""

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
    """Rotates adjacent (even, odd) element pairs of ``q`` and ``k``.

    Each lane processes one pair in both tensors, applying
    ``(x0, x1) -> (x0 * cos - x1 * sin, x0 * sin + x1 * cos)``. Elements past
    ``2 * rotary_pairs`` in each head vector are not written. Launch grid:
    ``(cdiv(total_pairs, BLOCK),)``.

    Args:
        q: Pointer to the contiguous query tensor, flattened to
            ``[vectors, head_dim]`` with ``seq`` as the fastest vector index.
        k: Pointer to the key tensor, same layout as ``q``.
        cos: Pointer to the ``[max_positions, rotary_pairs]`` cosine table.
        sin: Pointer to the ``[max_positions, rotary_pairs]`` sine table.
        out_q: Pointer to the output query tensor, same layout as ``q``.
        out_k: Pointer to the output key tensor, same layout as ``k``.
        total_pairs: Total number of rotated pairs (``vectors * rotary_pairs``).
        start_pos: Position of the first token (offset into the tables, e.g.
            the KV-cache length during decoding).
        head_dim: Size of each head vector.
        seq: Number of tokens per sequence.
        rotary_pairs: Number of rotated pairs per head vector.
        BLOCK: Number of pairs per program instance.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_pairs
    # Decompose the flat pair index into (vector, token, pair-in-vector).
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
