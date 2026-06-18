from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


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
    manifest: dict[str, object] = {"format": "dli.weights.v1", "data": data_path.name, "tensors": {}}
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


def _write_graph(path: Path, op_type: str, inputs: list[str], outputs: list[str],
                 attrs: dict[str, object] | None = None) -> Path:
    graph_path = path / "graph.json"
    graph = {
        "format": "dli.graph.v1",
        "model_type": "operator_numeric_test",
        "inputs": inputs,
        "outputs": outputs,
        "nodes": [{"name": op_type, "op": op_type, "inputs": inputs, "outputs": outputs, "attrs": attrs or {}}],
    }
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return graph_path


def _read_outputs(path: Path) -> dict[str, torch.Tensor]:
    manifest = json.loads((path / "outputs.json").read_text(encoding="utf-8"))
    outputs = {}
    for name, metadata in manifest["tensors"].items():
        dtype = torch.float32 if metadata["dtype"] == "float32" else torch.int64
        data = (path / metadata["file"]).read_bytes()
        outputs[name] = torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(metadata["shape"])
    return outputs


def _run_operator(op_type: str, inputs: dict[str, torch.Tensor],
                  outputs: list[str] | None = None,
                  attrs: dict[str, object] | None = None) -> dict[str, torch.Tensor]:
    outputs = outputs or ["output"]
    with tempfile.TemporaryDirectory(prefix=f"dli_{op_type}_") as temp:
        root = Path(temp)
        graph_path = _write_graph(root, op_type, list(inputs), outputs, attrs)
        input_path = _write_weights(root, inputs)
        output_dir = root / "out"
        subprocess.run(
            [str(_runner()), "--graph", str(graph_path), "--inputs", str(input_path),
             "--plugin", str(_plugin()), "--output-dir", str(output_dir)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return _read_outputs(output_dir)


def _assert_close(test: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor) -> None:
    test.assertEqual(list(actual.shape), list(expected.shape))
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device is not available")
class AotOperatorNumericsTest(unittest.TestCase):
    def test_add(self):
        x = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        b = torch.tensor([0.5, -1.0, 2.0])
        _assert_close(self, _run_operator("add", {"x": x, "b": b})["output"], x + b)

    def test_attention(self):
        q = torch.tensor([[[[0.2, -0.4], [1.0, 0.5]], [[-0.3, 0.7], [0.8, -0.6]]]])
        k = torch.tensor([[[[0.1, 0.9], [-0.5, 0.2]], [[0.6, -0.1], [0.3, 0.4]]]])
        v = torch.tensor([[[[1.0, -1.0], [0.5, 0.25]], [[-0.2, 0.8], [1.2, -0.7]]]])
        actual = _run_operator("attention", {"q": q, "k": k, "v": v}, attrs={"causal": True})["output"]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool))
        expected = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        _assert_close(self, actual, expected)

    def test_conv2d(self):
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 8.0
        weight = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]], [[[0.5, -0.25], [0.75, 0.125]]]])
        bias = torch.tensor([0.25, -0.5])
        actual = _run_operator("conv2d", {"x": x, "weight": weight, "bias": bias},
                               attrs={"stride": [1, 1], "padding": [0, 0]})["output"]
        _assert_close(self, actual, F.conv2d(x, weight, bias=bias))

    def test_embedding(self):
        ids = torch.tensor([[2, 1], [0, 3]], dtype=torch.int64)
        table = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10.0
        _assert_close(self, _run_operator("embedding", {"ids": ids, "table": table})["output"], table[ids])

    def test_linear(self):
        x = torch.tensor([[1.0, 2.0, -1.0, 0.5], [-0.5, 3.0, 1.5, 2.0]])
        weight = torch.tensor([[0.5, -1.0, 0.25, 2.0], [1.5, 0.0, -0.5, 0.75], [-1.0, 1.0, 0.5, 0.0]])
        bias = torch.tensor([0.1, -0.2, 0.3])
        _assert_close(self, _run_operator("linear", {"x": x, "weight": weight, "bias": bias})["output"],
                      F.linear(x, weight, bias))

    def test_matmul(self):
        a = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]])
        b = torch.tensor([[0.5, 1.0, -0.75], [1.5, -0.5, 0.25], [-1.0, 0.0, 2.0], [0.25, 0.75, -1.5]])
        _assert_close(self, _run_operator("matmul", {"a": a, "b": b})["output"], a @ b)

    def test_max_pool2d(self):
        x = torch.tensor([[[[1.0, -1.0, 3.0, 2.0], [0.5, 4.0, -2.0, 1.5],
                           [2.5, 0.0, 5.0, -0.5], [1.0, 3.5, 2.0, 4.5]]]])
        actual = _run_operator("max_pool2d", {"x": x},
                               attrs={"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]})["output"]
        _assert_close(self, actual, F.max_pool2d(x, kernel_size=2, stride=2))

    def test_mul(self):
        x = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        b = torch.tensor([0.5, -1.0, 2.0])
        _assert_close(self, _run_operator("mul", {"x": x, "b": b})["output"], x * b)

    def test_relu(self):
        x = torch.tensor([[-1.0, 0.0, 2.0], [3.5, -4.0, 5.0]])
        _assert_close(self, _run_operator("relu", {"x": x})["output"], torch.relu(x))

    def test_reshape(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        _assert_close(self, _run_operator("reshape", {"x": x}, attrs={"shape": [3, 2]})["output"], x.reshape(3, 2))

    def test_rms_norm(self):
        x = torch.tensor([[1.0, 2.0, -1.0, 0.5], [-0.5, 3.0, 1.5, 2.0]])
        weight = torch.tensor([0.5, 1.0, -1.5, 2.0])
        eps = 1e-6
        actual = _run_operator("rms_norm", {"x": x, "weight": weight}, attrs={"eps": eps})["output"]
        expected = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * weight
        _assert_close(self, actual, expected)

    def test_rotary_embedding(self):
        q = torch.tensor([[[[1.0, 2.0], [-1.0, 0.5]]]])
        k = torch.tensor([[[[0.25, -0.75], [1.5, -2.0]]]])
        cos = torch.tensor([[1.0], [0.5], [-0.25], [0.75]])
        sin = torch.tensor([[0.0], [0.25], [0.5], [-0.5]])
        outputs = _run_operator("rotary_embedding", {"q": q, "k": k, "cos": cos, "sin": sin},
                                outputs=["out_q", "out_k"], attrs={"start_pos": 1})

        def rotate(x: torch.Tensor) -> torch.Tensor:
            out = x.clone()
            c = cos[1:3, 0].reshape(1, 1, 2)
            s = sin[1:3, 0].reshape(1, 1, 2)
            out[..., 0] = x[..., 0] * c - x[..., 1] * s
            out[..., 1] = x[..., 0] * s + x[..., 1] * c
            return out

        _assert_close(self, outputs["out_q"], rotate(q))
        _assert_close(self, outputs["out_k"], rotate(k))

    def test_silu(self):
        x = torch.tensor([[-1.0, 0.0, 2.0], [3.5, -4.0, 5.0]])
        _assert_close(self, _run_operator("silu", {"x": x})["output"], F.silu(x))

    def test_softmax(self):
        x = torch.tensor([[1.0, 0.0, -1.0, 2.0, 0.5], [-0.5, 1.5, 2.5, -1.0, 0.0]])
        _assert_close(self, _run_operator("softmax", {"x": x})["output"], torch.softmax(x, dim=-1))

    def test_transpose(self):
        x = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        _assert_close(self, _run_operator("transpose", {"x": x}, attrs={"perm": [1, 0]})["output"], x.t())


if __name__ == "__main__":
    unittest.main()
