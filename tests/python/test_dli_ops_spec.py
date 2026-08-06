from __future__ import annotations

from dli_ops import OperatorSpec


def test_operator_spec_dataclass_defaults() -> None:
    spec = OperatorSpec(
        name="linear",
        type_name="linear",
        module="dli_ops.aot.linear.kernel",
        function="linear_kernel",
        description="test operator",
    )
    assert spec.name == "linear"
    assert spec.type_name == "linear"
    assert spec.module == "dli_ops.aot.linear.kernel"
    assert spec.function == "linear_kernel"
    assert spec.description == "test operator"
    assert spec.triton_source is None
    assert spec.cxx_source is None


if __name__ == "__main__":
    test_operator_spec_dataclass_defaults()
