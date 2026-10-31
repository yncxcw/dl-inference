"""Serve a DLI bundle: `python -m dli_serve --artifacts BUNDLE.json`."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("dli_serve")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dli-serve",
        description="Serve a DLI causal-LM bundle with an OpenAI-compatible HTTP API.",
    )
    parser.add_argument(
        "--artifacts", type=Path, required=True, help="dli.autoregressive.v1 bundle JSON"
    )
    parser.add_argument("--model-id", help="tokenizer ID or path (default: the bundle's model_id)")
    parser.add_argument(
        "--served-model-name", help="model name clients request (default: the model ID)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--max-num-seqs", type=int, default=4, help="sequences stepped concurrently"
    )
    parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=8,
        help="request-state snapshots kept for prefix reuse; 0 disables the cache",
    )
    parser.add_argument("--max-tokens", type=int, help="reply limit for requests that set none")
    parser.add_argument("--api-key", help="require this bearer token on /v1 endpoints")
    parser.add_argument(
        "--allow-origin", action="append", default=[], help="CORS origin to allow (repeatable)"
    )
    parser.add_argument(
        "--enable-thinking", action="store_true", help="enable Qwen thinking mode by default"
    )
    parser.add_argument(
        "--local-files-only", action="store_true", help="never contact the Hugging Face Hub"
    )
    parser.add_argument(
        "--log-level", default="info", choices=("debug", "info", "warning", "error")
    )
    return parser.parse_args(argv)


def sampling_defaults(model_id: str, *, local_files_only: bool):
    """Read default sampling values from the model's generation_config.json, as vLLM does."""

    from .api import SamplingDefaults

    try:
        transformers = importlib.import_module("transformers")
        config = transformers.GenerationConfig.from_pretrained(
            model_id, local_files_only=local_files_only
        )
    except Exception as error:  # a missing file just means OpenAI defaults
        logger.info("no generation config for %s (%s); using OpenAI defaults", model_id, error)
        return SamplingDefaults()
    if not getattr(config, "do_sample", False):
        return SamplingDefaults(temperature=0.0)
    return SamplingDefaults(
        temperature=float(1.0 if config.temperature is None else config.temperature),
        top_p=float(1.0 if config.top_p is None else config.top_p),
        top_k=int(config.top_k or 0),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        import torch
        import uvicorn

        from .api import create_app
        from .runner import DLIModelRunner, HFTokenizer, load_bundle
        from .scheduler import Scheduler
    except ImportError as error:
        print(
            f"error: {error}. Install the server dependencies with "
            "`python -m pip install fastapi uvicorn transformers`.",
            file=sys.stderr,
        )
        return 2

    try:
        bundle = load_bundle(args.artifacts)
        model_id = args.model_id or bundle.model_id
        if model_id is None:
            raise ValueError("the bundle has no model_id; pass --model-id")
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading tokenizer %s", model_id)
        tokenizer = HFTokenizer.from_pretrained(model_id, local_files_only=args.local_files_only)
        defaults = sampling_defaults(model_id, local_files_only=args.local_files_only)
        logger.info("loading %s on %s", args.artifacts, device)
        runner = DLIModelRunner(bundle, device=device)
        scheduler = Scheduler(
            runner,
            tokenizer.decode,
            max_num_seqs=args.max_num_seqs,
            prefix_cache_size=args.prefix_cache_size,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    model_name = args.served_model_name or model_id
    logger.info(
        "serving %s (context %d tokens, sampling defaults %s) on http://%s:%d",
        model_name,
        bundle.max_context_tokens,
        defaults,
        args.host,
        args.port,
    )
    app = create_app(
        scheduler,
        tokenizer,
        model_name=model_name,
        api_key=args.api_key,
        allow_origins=args.allow_origin,
        defaults=defaults,
        default_max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
