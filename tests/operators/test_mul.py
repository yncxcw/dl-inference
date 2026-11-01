"""Numeric test comparing the DLI mul operator with PyTorch."""

from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class MulOperatorNumericsTest(unittest.TestCase):
    """Checks the mul operator against its PyTorch reference."""

    def test_mul(self) -> None:
        """Verifies the mul operator output matches PyTorch within tolerance."""
        x = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        b = torch.tensor([0.5, -1.0, 2.0])
        assert_close(self, run_operator("mul", {"x": x, "b": b})["output"], x * b)


if __name__ == "__main__":
    unittest.main()
