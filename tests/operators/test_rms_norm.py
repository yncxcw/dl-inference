"""Numeric test comparing the DLI rms norm operator with PyTorch."""

from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class RmsNormOperatorNumericsTest(unittest.TestCase):
    """Checks the rms norm operator against its PyTorch reference."""

    def test_rms_norm(self) -> None:
        """Verifies the rms norm operator output matches PyTorch within tolerance."""
        x = torch.tensor([[1.0, 2.0, -1.0, 0.5], [-0.5, 3.0, 1.5, 2.0]])
        weight = torch.tensor([0.5, 1.0, -1.5, 2.0])
        eps = 1e-6
        actual = run_operator("rms_norm", {"x": x, "weight": weight}, attrs={"eps": eps})["output"]
        expected = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * weight
        assert_close(self, actual, expected)


if __name__ == "__main__":
    unittest.main()
