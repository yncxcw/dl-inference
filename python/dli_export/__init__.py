from importlib import import_module
from typing import Any

from .export import export_module
from .torch_export import TorchExportError, export_program, exported_program_to_dli


_QWEN3_5_EXPORTS = frozenset(
    (
        "export_qwen3_5_static_cache",
        "export_qwen3_5_to_dli",
    )
)


def __getattr__(name: str) -> Any:
    if name not in _QWEN3_5_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".qwen3_5", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_QWEN3_5_EXPORTS))


__all__ = [
    "TorchExportError",
    "export_module",
    "export_program",
    "export_qwen3_5_static_cache",
    "export_qwen3_5_to_dli",
    "exported_program_to_dli",
]
