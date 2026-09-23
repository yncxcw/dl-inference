from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from dli_export.weights import WeightWriter, dtype_name


def test_dtype_name() -> None:
    assert dtype_name(torch.zeros(1, dtype=torch.bool)) == "bool"
    assert dtype_name(torch.zeros(1, dtype=torch.float32)) == "float32"
    assert dtype_name(torch.zeros(1, dtype=torch.int64)) == "int64"
    try:
        dtype_name(torch.zeros(1, dtype=torch.int32))
    except TypeError:
        pass
    else:
        raise AssertionError("dtype_name should reject unsupported dtypes")


def test_weight_writer_manifest_and_payload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writer = WeightWriter(tmp, "weights")
        assert writer.manifest_name == "weights.dli.weights.json"
        assert writer.data_name == "weights.dli.weights.bin"
        writer.add("float", torch.tensor([[1.0, 2.0]], dtype=torch.float64))
        writer.add("ids", torch.tensor([3, 4], dtype=torch.int64))
        writer.add("mask", torch.tensor([True, False], dtype=torch.bool))
        writer.add("mask", torch.tensor([True, False], dtype=torch.bool))
        manifest_path = writer.write()

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert manifest["format"] == "dli.weights.v1"
        assert manifest["data"] == "weights.dli.weights.bin"
        assert manifest["tensors"]["float"]["dtype"] == "float32"
        assert manifest["tensors"]["float"]["shape"] == [1, 2]
        assert manifest["tensors"]["float"]["offset"] == 0
        assert manifest["tensors"]["ids"]["dtype"] == "int64"
        assert manifest["tensors"]["ids"]["offset"] == 8
        assert manifest["tensors"]["mask"]["dtype"] == "bool"
        assert manifest["tensors"]["mask"]["offset"] == 24
        assert (Path(tmp) / "weights.dli.weights.bin").stat().st_size == 26

        try:
            writer.add("mask", torch.tensor([False, True], dtype=torch.bool))
        except ValueError as error:
            assert "different values" in str(error)
        else:
            raise AssertionError("conflicting duplicate weight was accepted")


if __name__ == "__main__":
    test_dtype_name()
    test_weight_writer_manifest_and_payload()
