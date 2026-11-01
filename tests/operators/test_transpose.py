"""Numeric test comparing the DLI transpose operator with PyTorch."""

from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class TransposeOperatorNumericsTest(unittest.TestCase):
    """Checks the transpose operator against its PyTorch reference."""

    def test_transpose(self) -> None:
        """Verifies the transpose operator output matches PyTorch within tolerance."""
        x = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        actual = run_operator("transpose", {"x": x}, attrs={"perm": [1, 0]})["output"]
        assert_close(self, actual, x.t())


if __name__ == "__main__":
    unittest.main()
