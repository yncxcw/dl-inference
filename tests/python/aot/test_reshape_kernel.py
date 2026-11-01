"""Tests for the reshape AOT Triton kernel module."""

from __future__ import annotations

from dli_ops.aot.reshape import kernel


def test_reshape_kernel_module_documents_cpp_registration() -> None:
    """Verifies the reshape kernel module documents its C++ registration."""
    assert "C++" in (kernel.__doc__ or "")


if __name__ == "__main__":
    test_reshape_kernel_module_documents_cpp_registration()
