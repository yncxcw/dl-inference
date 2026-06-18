from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dli_export import export_module
from qwen2 import create_model


def main() -> None:
    model = create_model()
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, weights_path = export_module(model, (), tmp, model_type="qwen2", stem="qwen2")
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
    main()
