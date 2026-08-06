from __future__ import annotations

from dli_ops.aot.rotary import kernel
from test_support import assert_kernel_args


def test_rotary_kernel_signature() -> None:
    assert_kernel_args(
        kernel.rotary_kernel, ["q", "k", "cos", "sin", "out_q", "out_k", "total_pairs"]
    )


if __name__ == "__main__":
    test_rotary_kernel_signature()
