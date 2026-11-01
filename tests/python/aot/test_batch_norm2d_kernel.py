"""Tests for the batch norm2d AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.batch_norm2d import kernel
from test_support import assert_kernel_args


def test_batch_norm2d_kernel_signature() -> None:
    """Verifies the batch norm2d kernel exposes the expected argument names."""
    assert_kernel_args(
        kernel.batch_norm2d_kernel,
        [
            "x",
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "out",
            "total",
            "channels",
            "spatial",
            "eps",
        ],
    )


if __name__ == "__main__":
    test_batch_norm2d_kernel_signature()
