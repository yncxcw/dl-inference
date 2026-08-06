from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch

import dli
from dli_export import export_module


def _plugin() -> Path:
    value = os.environ.get("DLI_AOT_PLUGIN")
    if not value:
        raise RuntimeError("DLI_AOT_PLUGIN is not set")
    path = Path(value)
    if not path.exists():
        raise RuntimeError(f"DLI_AOT_PLUGIN does not exist: {path}")
    return path


def _run_graph(
    graph_path: Path, weights_path: Path, inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    engine = dli.Engine()
    engine.load_library(str(_plugin()))
    graph_inputs = dli.load_weights(str(weights_path), "cuda")
    graph_inputs.update(inputs)
    outputs = engine.run(dli.Graph.from_json_file(str(graph_path)), graph_inputs)
    torch.cuda.synchronize()
    return outputs


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 1e-3, atol: float = 1e-3
) -> None:
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=rtol, atol=atol)


def test_alexnet() -> None:
    from alexnet import create_model

    model = create_model().eval()
    x = (torch.arange(1 * 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32) % 29 - 14) / 29.0
    with tempfile.TemporaryDirectory(prefix="dli_e2e_alexnet_") as tmp:
        graph_path, weights_path = export_module(
            model,
            (torch.zeros(1, 3, 32, 32),),
            tmp,
            model_type="alexnet_e2e",
            stem="alexnet",
        )
        expected = model.cuda()(x.cuda())
        outputs = _run_graph(graph_path, weights_path, {"x": x.cuda()})
        actual = next(iter(outputs.values()))
        _assert_close(actual, expected, rtol=5e-3, atol=2e-2)


def test_qwen2() -> None:
    from qwen2 import create_model

    model = create_model().eval()
    input_ids = torch.tensor([2], dtype=torch.int64)
    with tempfile.TemporaryDirectory(prefix="dli_e2e_qwen2_") as tmp:
        graph_path, weights_path = export_module(model, (), tmp, model_type="qwen2", stem="qwen2")
        model.cuda()
        expected = model(input_ids.reshape(1, 1).cuda()).logits[:, -1, :]
        actual = _run_graph(graph_path, weights_path, {"input_ids": input_ids.cuda()})["logits"]
        _assert_close(actual, expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("alexnet", "qwen2"), required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("skipped: CUDA device is not available")
        return 0
    if args.model == "alexnet":
        test_alexnet()
    else:
        test_qwen2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
