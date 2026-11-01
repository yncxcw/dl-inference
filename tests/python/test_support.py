"""Shared helpers for kernel signature tests."""

from __future__ import annotations

from collections.abc import Iterable


def assert_kernel_args(kernel: object, names: Iterable[str]) -> None:
    """Asserts a Triton kernel declares the given argument names.

    Args:
        kernel: Kernel object expected to expose an ``arg_names`` attribute.
        names: Argument names that must all be present.
    """
    arg_names = getattr(kernel, "arg_names", None)
    assert arg_names is not None
    for name in names:
        assert name in arg_names
