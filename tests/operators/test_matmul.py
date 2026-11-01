"""Numeric test comparing the DLI matmul operator with PyTorch."""

from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class MatmulOperatorNumericsTest(unittest.TestCase):
    """Checks the matmul operator against its PyTorch reference."""

    def test_matmul(self) -> None:
        """Verifies the matmul operator output matches PyTorch within tolerance."""
        a = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]])
        b = torch.tensor(
            [[0.5, 1.0, -0.75], [1.5, -0.5, 0.25], [-1.0, 0.0, 2.0], [0.25, 0.75, -1.5]]
        )
        assert_close(self, run_operator("matmul", {"a": a, "b": b})["output"], a @ b)


if __name__ == "__main__":
    unittest.main()
