"""OpenAI-compatible HTTP serving for exported DLI causal-LM bundles.

The scheduler and prefix cache depend only on the DLI runtime and PyTorch.
The HTTP layer (`dli_serve.api`) additionally needs FastAPI and uvicorn, which
are imported only when it is used.
"""

from __future__ import annotations

from .kv_cache import CacheMatch, PrefixCache
from .scheduler import (
    Failed,
    Finished,
    GenerationRequest,
    RequestHandle,
    SamplingParams,
    Scheduler,
    TextDelta,
)

__all__ = [
    "CacheMatch",
    "Failed",
    "Finished",
    "GenerationRequest",
    "PrefixCache",
    "RequestHandle",
    "SamplingParams",
    "Scheduler",
    "TextDelta",
]
