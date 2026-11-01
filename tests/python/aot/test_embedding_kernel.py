"""Tests for the embedding AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.embedding import kernel
from test_support import assert_kernel_args


def test_embedding_kernel_signature() -> None:
    """Verifies the embedding kernel exposes the expected argument names."""
    assert_kernel_args(kernel.embedding_kernel, ["ids", "table", "out", "total", "hidden"])


if __name__ == "__main__":
    test_embedding_kernel_signature()
