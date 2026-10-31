from importlib import import_module
from typing import Any

from .export import export_module
from .torch_export import TorchExportError, export_program, exported_program_to_dli


# Model exporters load on first use, keeping `import dli_export` light.
_LAZY_EXPORTS = {
    "export_causal_lm": ".causal_lm",
    "export_causal_lm_to_dli": ".causal_lm",
    "export_qwen3_5_to_dli": ".qwen3_5",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_LAZY_EXPORTS))


__all__ = [
    "TorchExportError",
    "export_causal_lm",
    "export_causal_lm_to_dli",
    "export_module",
    "export_program",
    "export_qwen3_5_to_dli",
    "exported_program_to_dli",
]
