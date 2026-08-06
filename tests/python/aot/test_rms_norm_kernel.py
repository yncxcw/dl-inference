from __future__ import annotations

from dli_ops.aot.rms_norm import kernel
from test_support import assert_kernel_args


def test_rms_norm_kernel_signature() -> None:
    assert_kernel_args(kernel.rms_norm_kernel, ["x", "weight", "out", "rows", "eps", "hidden"])


if __name__ == "__main__":
    test_rms_norm_kernel_signature()
