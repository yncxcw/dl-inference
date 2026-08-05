from __future__ import annotations

from dli_ops.aot.conv2d import kernel
from test_support import assert_kernel_args


def test_conv2d_kernel_signature() -> None:
    assert_kernel_args(kernel.conv2d_kernel, ["x", "w", "bias", "out", "total", "in_c", "k_h", "k_w"])


if __name__ == "__main__":
    test_conv2d_kernel_signature()

