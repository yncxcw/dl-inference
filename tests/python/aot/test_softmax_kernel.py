"""Tests for the softmax AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.softmax import kernel
from test_support import assert_kernel_args


def test_softmax_kernel_signature() -> None:
    """Verifies the softmax kernel exposes the expected argument names."""
    assert_kernel_args(kernel.softmax_kernel, ["x", "out", "rows", "width", "BLOCK"])


if __name__ == "__main__":
    test_softmax_kernel_signature()
