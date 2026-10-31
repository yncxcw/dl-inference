from __future__ import annotations

import subprocess
import sys

import dli_export
import dli_export.export as export_impl
import dli_export.torch_export as torch_export_impl


def test_export_module_is_reexported() -> None:
    assert dli_export.export_module is export_impl.export_module
    assert dli_export.exported_program_to_dli is torch_export_impl.exported_program_to_dli


def test_model_exporters_are_loaded_lazily() -> None:
    script = """
import sys

import dli_export

assert "dli_export.causal_lm" not in sys.modules
assert "dli_export.qwen3_5" not in sys.modules
export = dli_export.export_causal_lm_to_dli
causal_lm = sys.modules["dli_export.causal_lm"]
assert export is causal_lm.export_causal_lm_to_dli
assert dli_export.export_causal_lm is causal_lm.export_causal_lm
assert dli_export.export_qwen3_5_to_dli is sys.modules["dli_export.qwen3_5"].export_qwen3_5_to_dli
assert {"export_causal_lm", "export_qwen3_5_to_dli"} <= set(dir(dli_export))
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_model_exporters_support_from_import() -> None:
    from dli_export import export_causal_lm_to_dli, export_qwen3_5_to_dli

    assert export_causal_lm_to_dli is sys.modules["dli_export.causal_lm"].export_causal_lm_to_dli
    assert export_qwen3_5_to_dli is sys.modules["dli_export.qwen3_5"].export_qwen3_5_to_dli


if __name__ == "__main__":
    test_export_module_is_reexported()
    test_model_exporters_are_loaded_lazily()
    test_model_exporters_support_from_import()
