from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

import dli_export.export as export_impl
from alexnet import create_model as create_alexnet
from dli_export import export_module
from qwen2 import create_model as create_qwen2


class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_private_parsers() -> None:
    assert export_impl._parse_shape("1,3,32,32") == (1, 3, 32, 32)
    assert export_impl._parse_model_kwargs(["local_files_only=true", "hidden=4", "name=qwen2"]) == {
        "local_files_only": True,
        "hidden": 4,
        "name": "qwen2",
    }
    assert callable(export_impl._load_factory("alexnet:create_model"))


def test_export_linear_fx_graph() -> None:
    model = LinearModel().eval()
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, weights_path = export_module(
            model, (torch.zeros(2, 4),), tmp, model_type="linear_test", stem="linear"
        )
        graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        assert graph["format"] == "dli.graph.v1"
        assert graph["model_type"] == "linear_test"
        assert graph["weights"] == "linear.dli.weights.json"
        assert graph["nodes"][0]["op"] == "linear"
        manifest = json.loads(Path(weights_path).read_text(encoding="utf-8"))
        assert manifest["tensors"]["linear.weight"]["shape"] == [3, 4]
        assert manifest["tensors"]["linear.bias"]["shape"] == [3]


def test_export_alexnet_fx_graph() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, weights_path = export_module(
            create_alexnet(),
            (torch.zeros(1, 3, 32, 32),),
            tmp,
            model_type="alexnet_tiny",
            stem="alexnet",
        )
        graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        ops = [node["op"] for node in graph["nodes"]]
        assert graph["model_type"] == "alexnet_tiny"
        assert ops == [
            "conv2d",
            "relu",
            "max_pool2d",
            "conv2d",
            "relu",
            "max_pool2d",
            "reshape",
            "linear",
            "relu",
            "linear",
        ]
        manifest = json.loads(Path(weights_path).read_text(encoding="utf-8"))
        assert manifest["tensors"]["features.0.weight"]["shape"] == [8, 3, 3, 3]
        assert manifest["tensors"]["classifier.1.weight"]["shape"] == [8, 1024]


def test_export_qwen2_graph() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, weights_path = export_module(create_qwen2(), (), tmp, model_type="qwen2", stem="qwen2")
        graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        ops = [node["op"] for node in graph["nodes"]]
        assert graph["model_type"] == "qwen2"
        assert graph["inputs"] == ["input_ids"]
        assert graph["outputs"] == ["logits"]
        assert ops == [
            "embedding", "rms_norm", "linear", "linear", "linear",
            "reshape", "reshape", "reshape", "rotary_embedding",
            "attention", "reshape", "linear", "add", "rms_norm",
            "linear", "linear", "silu", "mul", "linear", "add",
            "rms_norm", "linear",
        ]
        manifest = json.loads(Path(weights_path).read_text(encoding="utf-8"))
        assert manifest["tensors"]["rotary_cos"]["shape"] == [16, 1]
        assert manifest["tensors"]["rotary_sin"]["shape"] == [16, 1]
        assert manifest["tensors"]["model.embed_tokens.weight"]["shape"] == [8, 4]


if __name__ == "__main__":
    test_private_parsers()
    test_export_linear_fx_graph()
    test_export_alexnet_fx_graph()
    test_export_qwen2_graph()

