from __future__ import annotations

import subprocess
import sys

import dli_export
import dli_export.export as export_impl
import dli_export.torch_export as torch_export_impl


def test_export_module_is_reexported() -> None:
    assert dli_export.export_module is export_impl.export_module
    assert dli_export.exported_program_to_dli is torch_export_impl.exported_program_to_dli


def test_qwen3_5_exports_are_loaded_lazily() -> None:
    script = """
import sys

import dli_export

assert "dli_export.qwen3_5" not in sys.modules
qwen3_5_export = dli_export.export_qwen3_5_to_dli
qwen3_5_impl = sys.modules["dli_export.qwen3_5"]
assert qwen3_5_export is qwen3_5_impl.export_qwen3_5_to_dli
assert dli_export.export_qwen3_5_static_cache is qwen3_5_impl.export_qwen3_5_static_cache
assert "export_qwen3_5_to_dli" in dir(dli_export)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_qwen3_5_exports_support_from_import() -> None:
    from dli_export import export_qwen3_5_to_dli

    qwen3_5_impl = sys.modules["dli_export.qwen3_5"]
    assert dli_export.export_qwen3_5_to_dli is qwen3_5_impl.export_qwen3_5_to_dli
    assert export_qwen3_5_to_dli is dli_export.export_qwen3_5_to_dli


if __name__ == "__main__":
    test_export_module_is_reexported()
    test_qwen3_5_exports_are_loaded_lazily()
    test_qwen3_5_exports_support_from_import()
