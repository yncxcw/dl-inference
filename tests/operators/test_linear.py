from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class LinearOperatorNumericsTest(unittest.TestCase):
    def test_linear(self) -> None:
        x = torch.tensor([[1.0, 2.0, -1.0, 0.5], [-0.5, 3.0, 1.5, 2.0]])
        weight = torch.tensor([[0.5, -1.0, 0.25, 2.0], [1.5, 0.0, -0.5, 0.75], [-1.0, 1.0, 0.5, 0.0]])
        bias = torch.tensor([0.1, -0.2, 0.3])
        actual = run_operator("linear", {"x": x, "weight": weight, "bias": bias})["output"]
        assert_close(self, actual, F.linear(x, weight, bias))


if __name__ == "__main__":
    unittest.main()
