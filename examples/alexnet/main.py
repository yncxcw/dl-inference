"""Export a tiny AlexNet to DLI and run it with the Triton AOT operators."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root(start: Path) -> Path:
    """Finds the repository root by walking up from ``start``.

    Args:
        start: File or directory to start the search from.

    Returns:
        The first ancestor (or ``start`` itself) containing both ``CMakeLists.txt``
        and a ``python`` directory.

    Raises:
        RuntimeError: If no such directory exists.
    """
    for path in [start, *start.parents]:
        if (path / "CMakeLists.txt").exists() and (path / "python").exists():
            return path
    raise RuntimeError("failed to locate repository root")


# Make the built ``dli`` extension and the in-tree Python packages importable when the
# example is run directly from a checkout.
ROOT = _repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "build" / "python"))
sys.path.insert(1, str(ROOT / "python"))

import torch

import dli
from alexnet import create_model
from dli_export import export_module


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def example_input() -> torch.Tensor:
    """Builds a deterministic 1x3x32x32 input.

    Returns:
        A float32 tensor with values in roughly [-0.5, 0.5).
    """
    values = torch.arange(1 * 3 * 32 * 32, dtype=torch.float32)
    return ((values % 23) - 11).reshape(1, 3, 32, 32) / 23.0


def main() -> int:
    """Exports the model, then runs it on the GPU unless ``--export-only`` is given.

    Returns:
        Process exit code (0 on success).

    Raises:
        RuntimeError: If CUDA is unavailable when execution is requested.
    """
    args = parse_args()
    output_dir = args.output_dir or args.build_dir / "examples" / "alexnet"
    plugin = args.plugin or args.build_dir / "operators" / "libdli_triton_aot_ops.so"

    graph_path, weights_path = export_module(
        create_model(),
        (torch.zeros((1, 3, 32, 32), dtype=torch.float32),),
        output_dir,
        model_type="alexnet_tiny",
        stem="alexnet",
    )
    print(f"graph: {graph_path}")
    print(f"weights: {weights_path}")
    if args.export_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("alexnet example requires CUDA because the AOT operators run on GPU")

    inputs = dli.load_weights(str(weights_path), "cuda")
    inputs["x"] = example_input().to("cuda")

    engine = dli.Engine()
    engine.load_library(str(plugin))
    outputs = engine.run(dli.Graph.from_json_file(str(graph_path)), inputs)
    torch.cuda.synchronize()
    values = next(iter(outputs.values())).detach().cpu().flatten()[:8].tolist()
    print("output:", " ".join(str(value) for value in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
