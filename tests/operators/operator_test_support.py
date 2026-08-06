from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch


CUDA_AVAILABLE = torch.cuda.is_available()


def _runner() -> Path:
    value = os.environ.get("DLI_OP_TEST_RUNNER")
    if not value:
        raise unittest.SkipTest("DLI_OP_TEST_RUNNER is not set")
    path = Path(value)
    if not path.exists():
        raise unittest.SkipTest(f"DLI_OP_TEST_RUNNER does not exist: {path}")
    return path


def _plugin() -> Path:
    value = os.environ.get("DLI_AOT_PLUGIN")
    if not value:
        raise unittest.SkipTest("DLI_AOT_PLUGIN is not set")
    path = Path(value)
    if not path.exists():
        raise unittest.SkipTest(f"DLI_AOT_PLUGIN does not exist: {path}")
    return path


def _dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.float32:
        return "float32"
    if tensor.dtype == torch.int64:
        return "int64"
    raise TypeError(f"unsupported dtype: {tensor.dtype}")


def _write_weights(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    data_path = path / "inputs.bin"
    manifest_path = path / "inputs.json"
    offset = 0
    manifest: dict[str, object] = {
        "format": "dli.weights.v1",
        "data": data_path.name,
        "tensors": {},
    }
    with data_path.open("wb") as data:
        for name, tensor in tensors.items():
            host = tensor.detach().cpu().contiguous()
            payload = host.numpy().tobytes()
            data.write(payload)
            manifest["tensors"][name] = {
                "dtype": _dtype_name(host),
                "shape": list(host.shape),
                "offset": offset,
                "nbytes": len(payload),
            }
            offset += len(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_graph(
    path: Path,
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    attrs: dict[str, object] | None = None,
) -> Path:
    graph_path = path / "graph.json"
    graph = {
        "format": "dli.graph.v1",
        "model_type": "operator_numeric_test",
        "inputs": inputs,
        "outputs": outputs,
        "nodes": [
            {
                "name": op_type,
                "op": op_type,
                "inputs": inputs,
                "outputs": outputs,
                "attrs": attrs or {},
            }
        ],
    }
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return graph_path


def _read_outputs(path: Path) -> dict[str, torch.Tensor]:
    manifest = json.loads((path / "outputs.json").read_text(encoding="utf-8"))
    outputs = {}
    for name, metadata in manifest["tensors"].items():
        dtype = torch.float32 if metadata["dtype"] == "float32" else torch.int64
        data = (path / metadata["file"]).read_bytes()
        outputs[name] = (
            torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(metadata["shape"])
        )
    return outputs


def run_operator(
    op_type: str,
    inputs: dict[str, torch.Tensor],
    outputs: list[str] | None = None,
    attrs: dict[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    outputs = outputs or ["output"]
    with tempfile.TemporaryDirectory(prefix=f"dli_{op_type}_") as temp:
        root = Path(temp)
        graph_path = _write_graph(root, op_type, list(inputs), outputs, attrs)
        input_path = _write_weights(root, inputs)
        output_dir = root / "out"
        subprocess.run(
            [
                str(_runner()),
                "--graph",
                str(graph_path),
                "--inputs",
                str(input_path),
                "--plugin",
                str(_plugin()),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return _read_outputs(output_dir)


def assert_close(test: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor) -> None:
    test.assertEqual(list(actual.shape), list(expected.shape))
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)
