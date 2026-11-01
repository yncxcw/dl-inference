"""Serialization of tensors into the ``dli.weights.v1`` manifest and binary blob."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def dtype_name(tensor: torch.Tensor) -> str:
    """Returns the DLI manifest name for a tensor's dtype.

    Args:
        tensor: Tensor whose dtype is inspected.

    Returns:
        ``"float32"`` or ``"int64"``.

    Raises:
        TypeError: If the dtype is not supported by the weights format.
    """
    if tensor.dtype == torch.float32:
        return "float32"
    if tensor.dtype == torch.int64:
        return "int64"
    raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")


class WeightWriter:
    """Accumulates tensors and writes them as a JSON manifest plus a binary file.

    Tensors are packed back to back in insertion order; the manifest records
    each tensor's dtype, shape, byte offset and size.

    Attributes:
        output_dir: Directory the manifest and data files are written to.
        stem: Common file name prefix for the output files.
        tensors: Manifest entries keyed by tensor name.
        payloads: Raw tensor bytes, in the order they were added.
        offset: Total number of payload bytes added so far.
    """

    def __init__(self, output_dir: str | Path, stem: str) -> None:
        """Initializes an empty writer.

        Args:
            output_dir: Directory to write the output files to.
            stem: File name prefix, e.g. ``"model"``.
        """
        self.output_dir = Path(output_dir)
        self.stem = stem
        self.tensors: dict[str, dict[str, Any]] = {}
        self.payloads: list[bytes] = []
        self.offset = 0

    @property
    def manifest_name(self) -> str:
        """File name of the JSON manifest."""
        return f"{self.stem}.dli.weights.json"

    @property
    def data_name(self) -> str:
        """File name of the binary tensor data."""
        return f"{self.stem}.dli.weights.bin"

    def add(self, name: str, tensor: torch.Tensor) -> str:
        """Queues a tensor for writing.

        Tensors that are neither float32 nor int64 are converted to float32.

        Args:
            name: Key under which the tensor is recorded in the manifest.
            tensor: Tensor to store; it is detached and copied to host memory.

        Returns:
            ``name``, so callers can use the result as a graph input name.
        """
        host = tensor.detach().cpu().contiguous()
        if host.dtype not in (torch.float32, torch.int64):
            host = host.to(torch.float32)
        payload = host.numpy().tobytes()
        self.tensors[name] = {
            "dtype": dtype_name(host),
            "shape": list(host.shape),
            "offset": self.offset,
            "nbytes": len(payload),
        }
        self.offset += len(payload)
        self.payloads.append(payload)
        return name

    def write(self) -> Path:
        """Writes the data file and the manifest to ``output_dir``.

        Returns:
            Path of the written manifest file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data_path = self.output_dir / self.data_name
        with data_path.open("wb") as data:
            for payload in self.payloads:
                data.write(payload)
        manifest = {
            "format": "dli.weights.v1",
            "data": self.data_name,
            "tensors": self.tensors,
        }
        manifest_path = self.output_dir / self.manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path
