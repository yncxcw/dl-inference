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


if __name__ == "__main__":
    unittest.main()
