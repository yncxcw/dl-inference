from __future__ import annotations

from dli_ops.aot.add import kernel
from test_support import assert_kernel_args


def test_add_kernel_signature() -> None:
    assert_kernel_args(kernel.add_kernel, ["a", "b", "out", "total", "width", "BLOCK"])


if __name__ == "__main__":
    test_add_kernel_signature()
