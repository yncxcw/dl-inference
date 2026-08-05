from __future__ import annotations

from dli_ops.aot.batch_norm2d import kernel
from test_support import assert_kernel_args


def test_batch_norm2d_kernel_signature() -> None:
    assert_kernel_args(
        kernel.batch_norm2d_kernel,
        ["x", "weight", "bias", "running_mean", "running_var", "out", "total", "channels", "spatial", "eps"],
    )


if __name__ == "__main__":
    test_batch_norm2d_kernel_signature()
