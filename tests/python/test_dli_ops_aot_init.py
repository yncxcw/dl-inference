"""Tests for the ``dli_ops.aot`` package."""

from __future__ import annotations

import dli_ops.aot as aot


def test_aot_package_imports() -> None:
    """Verifies the ``dli_ops.aot`` package imports and has a docstring."""
    assert aot.__doc__


if __name__ == "__main__":
    test_aot_package_imports()
