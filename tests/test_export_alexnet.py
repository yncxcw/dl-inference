from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from alexnet import create_model
from dli_export import export_module


def main() -> None:
    model = create_model()
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, weights_path = export_module(
            model, (torch.zeros(1, 3, 32, 32),), tmp, model_type="alexnet_tiny", stem="alexnet"
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


if __name__ == "__main__":
    main()
