from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class MaxPool2dOperatorNumericsTest(unittest.TestCase):
    def test_max_pool2d(self) -> None:
        x = torch.tensor([[[[1.0, -1.0, 3.0, 2.0], [0.5, 4.0, -2.0, 1.5],
                           [2.5, 0.0, 5.0, -0.5], [1.0, 3.5, 2.0, 4.5]]]])
        actual = run_operator(
            "max_pool2d",
            {"x": x},
            attrs={"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]},
        )["output"]
        assert_close(self, actual, F.max_pool2d(x, kernel_size=2, stride=2))


if __name__ == "__main__":
    unittest.main()
