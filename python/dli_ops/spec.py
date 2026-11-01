"""Operator specification dataclass for Python/Triton operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OperatorSpec:
    """A Python/Triton operator exported through the C++ runtime plugin ABI.

    Attributes:
        name: Unique operator instance name.
        type_name: Operator type name registered with the C++ runtime.
        module: Python module that defines the operator.
        function: Name of the function inside ``module`` implementing the operator.
        description: Optional human-readable description.
        triton_source: Optional path to the Triton kernel source.
        cxx_source: Optional path to the C++ plugin source.
    """

    name: str
    type_name: str
    module: str
    function: str
    description: str = ""
    triton_source: Optional[str] = None
    cxx_source: Optional[str] = None
