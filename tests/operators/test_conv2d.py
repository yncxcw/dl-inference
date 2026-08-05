from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class Conv2dOperatorNumericsTest(unittest.TestCase):
    def test_conv2d(self) -> None:
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 8.0
        weight = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]], [[[0.5, -0.25], [0.75, 0.125]]]])
        bias = torch.tensor([0.25, -0.5])
        actual = run_operator(
            "conv2d",
            {"x": x, "weight": weight, "bias": bias},
            attrs={"stride": [1, 1], "padding": [0, 0]},
        )["output"]
        assert_close(self, actual, F.conv2d(x, weight, bias=bias))


if __name__ == "__main__":
    unittest.main()
