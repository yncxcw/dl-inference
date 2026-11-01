"""Tests for the max pool2d AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.max_pool2d import kernel
from test_support import assert_kernel_args


def test_max_pool2d_kernel_signature() -> None:
    """Verifies the max pool2d kernel exposes the expected argument names."""
    assert_kernel_args(
        kernel.max_pool2d_kernel, ["x", "out", "total", "n_dim", "c_dim", "k_h", "k_w"]
    )


if __name__ == "__main__":
    test_max_pool2d_kernel_signature()
