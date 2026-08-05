from __future__ import annotations

import dli_export
import dli_export.export as export_impl


def test_export_module_is_reexported() -> None:
    assert dli_export.export_module is export_impl.export_module


if __name__ == "__main__":
    test_export_module_is_reexported()

