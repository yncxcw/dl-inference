from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class RotaryEmbeddingOperatorNumericsTest(unittest.TestCase):
    def test_rotary_embedding(self) -> None:
        q = torch.tensor([[[[1.0, 2.0], [-1.0, 0.5]]]])
        k = torch.tensor([[[[0.25, -0.75], [1.5, -2.0]]]])
        cos = torch.tensor([[1.0], [0.5], [-0.25], [0.75]])
        sin = torch.tensor([[0.0], [0.25], [0.5], [-0.5]])
        outputs = run_operator(
            "rotary_embedding",
            {"q": q, "k": k, "cos": cos, "sin": sin},
            outputs=["out_q", "out_k"],
            attrs={"start_pos": 1},
        )

        def rotate(x: torch.Tensor) -> torch.Tensor:
            out = x.clone()
            c = cos[1:3, 0].reshape(1, 1, 2)
            s = sin[1:3, 0].reshape(1, 1, 2)
            out[..., 0] = x[..., 0] * c - x[..., 1] * s
            out[..., 1] = x[..., 0] * s + x[..., 1] * c
            return out

        assert_close(self, outputs["out_q"], rotate(q))
        assert_close(self, outputs["out_k"], rotate(k))


if __name__ == "__main__":
    unittest.main()
