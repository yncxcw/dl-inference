from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from dli_export import export_module


class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> None:
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


if __name__ == "__main__":
    main()
