"""Tests for the matmul AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.matmul import kernel
from test_support import assert_kernel_args


def test_matmul_kernel_signature() -> None:
    """Verifies the matmul kernel exposes the expected argument names."""
    assert_kernel_args(kernel.matmul_kernel, ["a", "b", "out", "m", "n", "k"])


if __name__ == "__main__":
    test_matmul_kernel_signature()
