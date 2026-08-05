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

import dli
from dli_export import export_module
from qwen2 import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.build_dir / "examples" / "qwen2"
    plugin = args.plugin or args.build_dir / "operators" / "libdli_triton_aot_ops.so"

    graph_path, weights_path = export_module(
        create_model(model_id=args.model_id, local_files_only=args.local_files_only),
        (),
        output_dir,
        model_type="qwen2",
        stem="qwen2",
    )
    print(f"graph: {graph_path}")
    print(f"weights: {weights_path}")
    if args.export_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("qwen2 example requires CUDA because the AOT operators run on GPU")

    inputs = dli.load_weights(str(weights_path), "cuda")
    inputs["input_ids"] = torch.tensor([2], dtype=torch.int64, device="cuda")

    engine = dli.Engine()
    engine.load_library(str(plugin))
    outputs = engine.run(dli.Graph.from_json_file(str(graph_path)), inputs)
    torch.cuda.synchronize()
    logits = outputs["logits"].detach().cpu().flatten()[:16].tolist()
    print("logits:", " ".join(str(value) for value in logits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
