from __future__ import annotations

import dli_ops.aot as aot


def test_aot_package_imports() -> None:
    assert aot.__doc__


if __name__ == "__main__":
    test_aot_package_imports()

