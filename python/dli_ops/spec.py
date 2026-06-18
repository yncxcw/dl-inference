from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OperatorSpec:
    """A Python/Triton operator exported through the C++ runtime plugin ABI."""

    name: str
    type_name: str
    module: str
    function: str
    description: str = ""
    triton_source: Optional[str] = None
    cxx_source: Optional[str] = None
