from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from operator_test_support import CUDA_AVAILABLE, assert_close, run_operator


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA device is not available")
class BatchNorm2dOperatorNumericsTest(unittest.TestCase):
    def test_batch_norm2d(self) -> None:
        x = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4) / 8.0
        weight = torch.tensor([1.25, -0.5, 2.0])
        bias = torch.tensor([0.25, 1.0, -0.75])
        running_mean = torch.tensor([0.5, 1.5, 2.5])
        running_var = torch.tensor([0.25, 1.0, 4.0])
        eps = 1e-4
        actual = run_operator(
            "batch_norm2d",
            {
                "x": x,
                "weight": weight,
                "bias": bias,
                "running_mean": running_mean,
                "running_var": running_var,
            },
            attrs={"eps": eps},
        )["output"]
        expected = F.batch_norm(x, running_mean, running_var, weight, bias, training=False, eps=eps)
        assert_close(self, actual, expected)


if __name__ == "__main__":
    unittest.main()
