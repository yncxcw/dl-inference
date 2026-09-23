from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.bool:
        return "bool"
    if tensor.dtype == torch.float32:
        return "float32"
    if tensor.dtype == torch.int64:
        return "int64"
    raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")


class WeightWriter:
    def __init__(self, output_dir: str | Path, stem: str):
        self.output_dir = Path(output_dir)
        self.stem = stem
        self.tensors: dict[str, dict[str, Any]] = {}
        self.payloads: list[bytes] = []
        self._payloads_by_name: dict[str, bytes] = {}
        self.offset = 0

    @property
    def manifest_name(self) -> str:
        return f"{self.stem}.dli.weights.json"

    @property
    def data_name(self) -> str:
        return f"{self.stem}.dli.weights.bin"

    def add(self, name: str, tensor: torch.Tensor) -> str:
        host = tensor.detach().cpu().contiguous()
        if host.dtype not in (torch.bool, torch.float32, torch.int64):
            host = host.to(torch.float32)
        payload = host.numpy().tobytes()
        metadata = {
            "dtype": dtype_name(host),
            "shape": list(host.shape),
        }
        if name in self.tensors:
            existing = self.tensors[name]
            if (
                existing["dtype"] != metadata["dtype"]
                or existing["shape"] != metadata["shape"]
                or self._payloads_by_name[name] != payload
            ):
                raise ValueError(f"weight {name!r} was added twice with different values")
            return name
        self.tensors[name] = {
            **metadata,
            "offset": self.offset,
            "nbytes": len(payload),
        }
        self._payloads_by_name[name] = payload
        self.offset += len(payload)
        self.payloads.append(payload)
        return name

    def write(self) -> Path:
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
