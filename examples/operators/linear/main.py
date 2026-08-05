from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "CMakeLists.txt").exists() and (path / "python").exists():
            return path
    raise RuntimeError("failed to locate repository root")


ROOT = _repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "build" / "python"))
sys.path.insert(1, str(ROOT / "python"))

import torch
from torch import nn

import dli
from dli_export import export_module


class LinearExample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    dtype=torch.float32,
                )
            )
            self.linear.bias.copy_(torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.build_dir / "examples" / "operators" / "linear"
    plugin = args.plugin or args.build_dir / "operators" / "libdli_triton_aot_ops.so"

    graph_path, weights_path = export_module(
        LinearExample().eval(),
        (torch.zeros((2, 4), dtype=torch.float32),),
        output_dir,
        model_type="operator_linear_example",
        stem="linear",
    )
    print(f"graph: {graph_path}")
    print(f"weights: {weights_path}")
    if args.export_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("linear example requires CUDA because the AOT operators run on GPU")

    inputs = dli.load_weights(str(weights_path), "cuda")
    inputs["x"] = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float32,
        device="cuda",
    )

    engine = dli.Engine()
    engine.load_library(str(plugin))
    outputs = engine.run(dli.Graph.from_json_file(str(graph_path)), inputs)
    torch.cuda.synchronize()
    values = outputs["linear"].detach().cpu().flatten().tolist()
    print("output:", " ".join(str(value) for value in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
