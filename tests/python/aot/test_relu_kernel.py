"""Tests for the relu AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.relu import kernel
from test_support import assert_kernel_args


def test_relu_kernel_signature() -> None:
    """Verifies the relu kernel exposes the expected argument names."""
    assert_kernel_args(kernel.relu_kernel, ["x", "out", "total", "BLOCK"])


if __name__ == "__main__":
    test_relu_kernel_signature()
