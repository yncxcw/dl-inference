"""Tests for the transpose AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.transpose import kernel
from test_support import assert_kernel_args


def test_transpose2d_kernel_signature() -> None:
    """Verifies the transpose kernel exposes the expected argument names."""
    assert_kernel_args(kernel.transpose2d_kernel, ["x", "out", "rows", "cols"])


if __name__ == "__main__":
    test_transpose2d_kernel_signature()
