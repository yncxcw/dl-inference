"""Tests for the linear AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.linear import kernel
from test_support import assert_kernel_args


def test_linear_kernel_signature() -> None:
    """Verifies the linear kernel exposes the expected argument names."""
    assert_kernel_args(kernel.linear_kernel, ["x", "weight", "bias", "out", "m", "n", "k"])


if __name__ == "__main__":
    test_linear_kernel_signature()
