"""Tests for the attention AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.attention import kernel
from test_support import assert_kernel_args


def test_attention_kernel_signature() -> None:
    """Verifies the attention kernel exposes the expected argument names."""
    assert_kernel_args(
        kernel.attention_kernel, ["q", "k", "v", "out", "seq_q", "seq_k", "head_dim"]
    )


if __name__ == "__main__":
    test_attention_kernel_signature()
