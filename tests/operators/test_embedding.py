from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class EmbeddingOperatorNumericsTest(unittest.TestCase):
    def test_embedding(self) -> None:
        ids = torch.tensor([[2, 1], [0, 3]], dtype=torch.int64)
        table = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10.0
        assert_close(
            self, run_operator("embedding", {"ids": ids, "table": table})["output"], table[ids]
        )


if __name__ == "__main__":
    unittest.main()
