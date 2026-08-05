from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class ReluOperatorNumericsTest(unittest.TestCase):
    def test_relu(self) -> None:
        x = torch.tensor([[-1.0, 0.0, 2.0], [3.5, -4.0, 5.0]])
        assert_close(self, run_operator("relu", {"x": x})["output"], torch.relu(x))


if __name__ == "__main__":
    unittest.main()
