"""Tests for the mul AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.mul import kernel
from test_support import assert_kernel_args


def test_mul_kernel_signature() -> None:
    """Verifies the mul kernel exposes the expected argument names."""
    assert_kernel_args(kernel.mul_kernel, ["a", "b", "out", "total", "width", "BLOCK"])


if __name__ == "__main__":
    test_mul_kernel_signature()
