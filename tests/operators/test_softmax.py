"""Numeric test comparing the DLI softmax operator with PyTorch."""

from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class SoftmaxOperatorNumericsTest(unittest.TestCase):
    """Checks the softmax operator against its PyTorch reference."""

    def test_softmax(self) -> None:
        """Verifies the softmax operator output matches PyTorch within tolerance."""
        x = torch.tensor([[1.0, 0.0, -1.0, 2.0, 0.5], [-0.5, 1.5, 2.5, -1.0, 0.0]])
        assert_close(self, run_operator("softmax", {"x": x})["output"], torch.softmax(x, dim=-1))


if __name__ == "__main__":
    unittest.main()
