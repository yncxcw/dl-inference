from __future__ import annotations

import math
import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class AttentionOperatorNumericsTest(unittest.TestCase):
    def test_attention(self) -> None:
        q = torch.tensor([[[[0.2, -0.4], [1.0, 0.5]], [[-0.3, 0.7], [0.8, -0.6]]]])
        k = torch.tensor([[[[0.1, 0.9], [-0.5, 0.2]], [[0.6, -0.1], [0.3, 0.4]]]])
        v = torch.tensor([[[[1.0, -1.0], [0.5, 0.25]], [[-0.2, 0.8], [1.2, -0.7]]]])
        actual = run_operator("attention", {"q": q, "k": k, "v": v}, attrs={"causal": True})[
            "output"
        ]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool))
        expected = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        assert_close(self, actual, expected)

    def test_grouped_query_attention_decode(self) -> None:
        generator = torch.Generator().manual_seed(17)
        q = torch.randn(2, 8, 1, 2, generator=generator)
        k = torch.randn(2, 2, 5, 2, generator=generator)
        v = torch.randn(2, 2, 5, 2, generator=generator)

        actual = run_operator("attention", {"q": q, "k": k, "v": v}, attrs={"causal": True})[
            "output"
        ]

        heads_per_kv_head = q.shape[1] // k.shape[1]
        expanded_k = k.repeat_interleave(heads_per_kv_head, dim=1)
        expanded_v = v.repeat_interleave(heads_per_kv_head, dim=1)
        scores = (q @ expanded_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        expected = torch.softmax(scores, dim=-1) @ expanded_v
        assert_close(self, actual, expected)

    def test_noncausal_decode_masks_padded_key_columns(self) -> None:
        generator = torch.Generator().manual_seed(19)
        q = torch.randn(1, 4, 1, 2, generator=generator)
        k = torch.randn(1, 2, 5, 2, generator=generator)
        v = torch.randn(1, 2, 5, 2, generator=generator)

        actual = run_operator("attention", {"q": q, "k": k, "v": v}, attrs={"causal": False})[
            "output"
        ]

        expanded_k = k.repeat_interleave(2, dim=1)
        expanded_v = v.repeat_interleave(2, dim=1)
        scores = (q @ expanded_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        expected = torch.softmax(scores, dim=-1) @ expanded_v
        assert_close(self, actual, expected)

    def test_qwen35_head_dim_decode_uses_opted_in_shared_memory(self) -> None:
        generator = torch.Generator().manual_seed(23)
        q = torch.randn(1, 8, 1, 256, generator=generator)
        k = torch.randn(1, 2, 3, 256, generator=generator)
        v = torch.randn(1, 2, 3, 256, generator=generator)

        actual = run_operator("attention", {"q": q, "k": k, "v": v}, attrs={"causal": True})[
            "output"
        ]

        expanded_k = k.repeat_interleave(4, dim=1)
        expanded_v = v.repeat_interleave(4, dim=1)
        scores = (q @ expanded_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        expected = torch.softmax(scores, dim=-1) @ expanded_v
        assert_close(self, actual, expected)


if __name__ == "__main__":
    unittest.main()
