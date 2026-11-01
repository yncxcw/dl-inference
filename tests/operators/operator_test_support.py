"""Shared helpers for operator numeric tests that run the DLI engine runner."""

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
    """Locates the operator test runner binary.

    Returns:
        Path to the executable named by ``DLI_OP_TEST_RUNNER``.

    Raises:
        unittest.SkipTest: If the variable is unset or the path is missing.
    """
    value = os.environ.get("DLI_OP_TEST_RUNNER")
    if not value:
        raise unittest.SkipTest("DLI_OP_TEST_RUNNER is not set")
    path = Path(value)
    if not path.exists():
        raise unittest.SkipTest(f"DLI_OP_TEST_RUNNER does not exist: {path}")
    return path


def _plugin() -> Path:
    """Locates the AOT plugin shared library.

    Returns:
        Path to the library named by ``DLI_AOT_PLUGIN``.

    Raises:
        unittest.SkipTest: If the variable is unset or the path is missing.
    """
    value = os.environ.get("DLI_AOT_PLUGIN")
    if not value:
        raise unittest.SkipTest("DLI_AOT_PLUGIN is not set")
    path = Path(value)
    if not path.exists():
        raise unittest.SkipTest(f"DLI_AOT_PLUGIN does not exist: {path}")
    return path


def _dtype_name(tensor: torch.Tensor) -> str:
    """Returns the DLI manifest dtype name for a tensor.

    Args:
        tensor: Tensor whose dtype is inspected.

    Returns:
        ``"float32"`` or ``"int64"``.

    Raises:
        TypeError: If the dtype is not supported.
    """
    if tensor.dtype == torch.float32:
        return "float32"
    if tensor.dtype == torch.int64:
        return "int64"
    raise TypeError(f"unsupported dtype: {tensor.dtype}")


def _write_weights(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    """Writes tensors as a DLI weights file (binary payload plus JSON manifest).

    Args:
        path: Directory that receives ``inputs.bin`` and ``inputs.json``.
        tensors: Tensors to serialize, keyed by name.

    Returns:
        Path to the written manifest.
    """
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
    """Writes a single-node DLI graph JSON file.

    Args:
        path: Directory that receives ``graph.json``.
        op_type: Operator type used for the node's name and op.
        inputs: Graph and node input tensor names.
        outputs: Graph and node output tensor names.
        attrs: Optional operator attributes.

    Returns:
        Path to the written graph file.
    """
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
    """Reads output tensors written by the operator runner.

    Args:
        path: Output directory containing ``outputs.json`` and tensor files.

    Returns:
        Output tensors keyed by name.
    """
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
    """Runs one operator through the DLI engine runner.

    Args:
        op_type: Operator type to execute.
        inputs: Input tensors keyed by name.
        outputs: Output tensor names; defaults to ``["output"]``.
        attrs: Optional operator attributes.

    Returns:
        Output tensors keyed by name.

    Raises:
        unittest.SkipTest: If the runner or plugin is not configured.
        subprocess.CalledProcessError: If the runner exits with an error.
    """
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
    """Asserts two tensors have equal shape and close values.

    Args:
        test: Test case used for the shape assertion.
        actual: Tensor produced by the operator under test.
        expected: Reference tensor, moved to CPU before comparison.
    """
    test.assertEqual(list(actual.shape), list(expected.shape))
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)
