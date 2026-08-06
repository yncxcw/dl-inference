from __future__ import annotations

from collections.abc import Iterable


def assert_kernel_args(kernel: object, names: Iterable[str]) -> None:
    arg_names = getattr(kernel, "arg_names", None)
    assert arg_names is not None
    for name in names:
        assert name in arg_names
