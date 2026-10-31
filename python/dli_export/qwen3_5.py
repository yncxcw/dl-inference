"""Qwen3.5 export, kept as a thin alias of the generic causal-LM exporter.

Qwen3.5 needs no model-specific code: `dli_export.causal_lm` exports any
Hugging Face causal LM, including Qwen3.5's hybrid linear-attention layers.
This module keeps `python -m dli_export.qwen3_5 ...` and
`export_qwen3_5_to_dli` working with their historical `qwen3_5` file stem.
New code should use `python -m dli_export.causal_lm`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from .causal_lm import export_causal_lm_to_dli
from .causal_lm import main as _causal_lm_main


def export_qwen3_5_to_dli(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    max_cache_len: int,
    stem: str = "qwen3_5",
    model_id: str | None = None,
    batch_size: int = 1,
    strict: bool = False,
) -> Path:
    """Export an FP32 Qwen3.5 model as a deployable autoregressive bundle."""

    if batch_size != 1:
        raise ValueError("DLI autoregressive bundles currently require batch_size=1")
    return export_causal_lm_to_dli(
        model,
        output_dir,
        max_cache_len=max_cache_len,
        stem=stem,
        model_id=model_id,
        strict=strict,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--stem" or arg.startswith("--stem=") for arg in args):
        args += ["--stem", "qwen3_5"]
    return _causal_lm_main(args)


__all__ = ["export_qwen3_5_to_dli"]


if __name__ == "__main__":
    raise SystemExit(_main())
