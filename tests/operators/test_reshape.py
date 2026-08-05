from __future__ import annotations

import unittest

import torch

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class ReshapeOperatorNumericsTest(unittest.TestCase):
    def test_reshape(self) -> None:
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        actual = run_operator("reshape", {"x": x}, attrs={"shape": [3, 2]})["output"]
        assert_close(self, actual, x.reshape(3, 2))


if __name__ == "__main__":
    unittest.main()
