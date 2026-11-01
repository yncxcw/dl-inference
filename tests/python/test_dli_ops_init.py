"""Tests for the ``dli_ops`` package namespace."""

from __future__ import annotations

import dli_ops
from dli_ops.spec import OperatorSpec


def test_operator_spec_is_reexported() -> None:
    """Verifies ``dli_ops`` re-exports ``OperatorSpec``."""
    assert dli_ops.OperatorSpec is OperatorSpec


if __name__ == "__main__":
    test_operator_spec_is_reexported()
