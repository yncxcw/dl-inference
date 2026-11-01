"""Tests for the conv2d AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.conv2d import kernel
from test_support import assert_kernel_args


def test_conv2d_kernel_signature() -> None:
    """Verifies the conv2d kernel exposes the expected argument names."""
    assert_kernel_args(
        kernel.conv2d_kernel, ["x", "w", "bias", "out", "total", "in_c", "k_h", "k_w"]
    )


if __name__ == "__main__":
    test_conv2d_kernel_signature()
