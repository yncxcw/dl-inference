"""Tests for the silu AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.silu import kernel
from test_support import assert_kernel_args


def test_silu_kernel_signature() -> None:
    """Verifies the silu kernel exposes the expected argument names."""
    assert_kernel_args(kernel.silu_kernel, ["x", "out", "total", "BLOCK"])


if __name__ == "__main__":
    test_silu_kernel_signature()
