"""Python bindings for the dl-inference (DLI) runtime.

Re-exports the native ``Engine``, ``Graph`` and ``load_weights`` entry points
from the compiled ``_dli_native`` extension.
"""

from __future__ import annotations

# Import torch first so its shared libraries are loaded before the native
# extension, which links against them.
import torch as _torch  # noqa: F401

from ._dli_native import Engine, Graph, load_weights

__all__ = ["Engine", "Graph", "load_weights"]
