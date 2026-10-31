"""Export any Hugging Face causal LM to a DLI autoregressive bundle.

The exporter knows no model architectures. It relies on two generic
Transformers facilities instead:

* ``StaticCache`` builds the cache layer each layer type in the config needs
  (full attention, linear attention, or a hybrid) and keeps all per-step state
  in tensors. A warm-up step materializes the layers, and every tensor
  reachable from a layer (attributes, and dict or list entries) becomes named
  request state that the graph takes as an input and returns updated.
* Functions wrapped by Transformers' Hub-kernel loader are swapped for their
  PyTorch reference implementations while tracing, so optional fused kernels
  never reach the graph.

The result is one fixed-capacity graph that consumes one token per run and
serves both prefill and decode. Python-level cache flags are fixed after the
warm-up, so tracing takes the continuation path; starting that path from zero
state is equivalent to a fresh cache. Export fails if a cache layer keeps
per-step state outside tensors.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ARTIFACT_FORMAT = "dli.autoregressive.v1"
_CURSOR = "cursor"


@dataclass(frozen=True)
class StateSpec:
    """One request-state tensor and where it lives in the Transformers cache."""

    name: str
    path: tuple[Any, ...]  # (layer index, attribute[, dict key or list index])
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class CausalLMExport:
    """A one-token step program plus the metadata the DLI converter needs."""

    program: torch.export.ExportedProgram
    step_module: torch.nn.Module
    model_type: str
    max_cache_len: int
    state_specs: tuple[StateSpec, ...]
    state_initializers: dict[str, torch.Tensor]
    output_names: tuple[str, ...]
    state_inputs: dict[str, str]
    state_outputs: dict[str, str]

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.state_specs)

    def initial_state(self) -> tuple[torch.Tensor, ...]:
        """Fresh zero state in the step module's positional order."""

        return tuple(self.state_initializers[name].clone() for name in self.state_names)


def _transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ImportError as error:
        raise ImportError("causal-LM export requires Hugging Face Transformers") from error


@contextmanager
def hub_kernel_fallbacks(model: torch.nn.Module) -> Iterator[None]:
    """Use the PyTorch reference for every Hub-kernel function the model calls."""

    modules = {sys.modules[type(module).__module__] for module in model.modules()}
    patched: list[tuple[Any, str, Any]] = []
    try:
        for module in modules:
            for name, value in list(vars(module).items()):
                code = getattr(value, "__code__", None)
                filename = getattr(code, "co_filename", "").replace("\\", "/")
                if hasattr(value, "__wrapped__") and filename.endswith(
                    "integrations/hub_kernels.py"
                ):
                    patched.append((module, name, value))
                    setattr(module, name, value.__wrapped__)
        yield
    finally:
        for module, name, value in reversed(patched):
            setattr(module, name, value)


def _entries(value: Any) -> Iterator[tuple[Any, Any]]:
    if isinstance(value, Mapping):
        yield from value.items()
    elif isinstance(value, (list, tuple)):
        yield from enumerate(value)


def _cache_tensors(cache: Any) -> list[tuple[tuple[Any, ...], torch.Tensor]]:
    """Every tensor a cache layer holds, with its (layer, attribute[, key]) path."""

    found = []
    for index, layer in enumerate(cache.layers):
        for attribute, value in vars(layer).items():
            if isinstance(value, torch.Tensor):
                found.append(((index, attribute), value))
            for key, item in _entries(value):
                if isinstance(item, torch.Tensor):
                    found.append(((index, attribute, key), item))
    return found


def _python_state(cache: Any) -> dict[tuple[Any, ...], Any]:
    """Non-tensor layer state, used to detect state that tracing would freeze."""

    state = {}
    for index, layer in enumerate(cache.layers):
        for attribute, value in vars(layer).items():
            if isinstance(value, torch.Tensor):
                continue
            entries = list(_entries(value))
            if entries:
                for key, item in entries:
                    if not isinstance(item, torch.Tensor):
                        state[(index, attribute, key)] = item
            else:
                state[(index, attribute)] = value
    return state


def _get(cache: Any, path: tuple[Any, ...]) -> torch.Tensor:
    value = getattr(cache.layers[path[0]], path[1])
    return value if len(path) == 2 else value[path[2]]


def _set(cache: Any, path: tuple[Any, ...], tensor: torch.Tensor) -> None:
    layer = cache.layers[path[0]]
    if len(path) == 2:
        setattr(layer, path[1], tensor)
    else:
        getattr(layer, path[1])[path[2]] = tensor


def _copy_cache(template: Any) -> Any:
    """Copy the cache structure so a step can swap tensors without touching it."""

    cache = copy.copy(template)
    layers = []
    for layer in template.layers:
        clone = copy.copy(layer)
        for attribute, value in vars(layer).items():
            if isinstance(value, (dict, list)):
                setattr(clone, attribute, copy.copy(value))
        layers.append(clone)
    cache.layers = layers
    return cache


def _state_name(path: tuple[Any, ...]) -> str:
    return "layer." + ".".join(str(part) for part in path)


def _model_step(
    model: torch.nn.Module,
    cache: Any,
    input_ids: torch.Tensor,
    cursor: torch.Tensor,
    max_cache_len: int,
) -> torch.Tensor:
    """Run one token at position `cursor` against a fixed-capacity cache."""

    batch = input_ids.shape[0]
    visible = torch.arange(max_cache_len, device=input_ids.device) <= cursor
    return model(
        input_ids=input_ids.unsqueeze(-1),
        position_ids=cursor.reshape(1, 1).expand(batch, 1),
        cache_position=cursor.reshape(1),
        attention_mask=visible.unsqueeze(0).expand(batch, -1),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    ).logits


class CausalLMStep(torch.nn.Module):
    """`(input_ids, *state, cursor) -> (logits, *next_state, next_cursor)`."""

    def __init__(
        self,
        model: torch.nn.Module,
        template_cache: Any,
        state_specs: Sequence[StateSpec],
        max_cache_len: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_cache_len = max_cache_len
        self._template = template_cache
        self._paths = tuple(spec.path for spec in state_specs if spec.name != _CURSOR)

    def forward(self, input_ids: torch.Tensor, *state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cache = _copy_cache(self._template)
        for path, tensor in zip(self._paths, state):
            # Clone so in-place cache updates never alias a state input.
            _set(cache, path, tensor.clone())
        cursor = state[-1]
        logits = _model_step(self.model, cache, input_ids, cursor, self.max_cache_len)
        return (logits, *(_get(cache, path) for path in self._paths), cursor + 1)


def _materialize_cache(model: torch.nn.Module, max_cache_len: int, device: torch.device) -> Any:
    """Build a StaticCache and run warm-up steps so every layer holds its tensors."""

    try:
        cache = _transformers().StaticCache(config=model.config, max_cache_len=max_cache_len)
    except Exception as error:
        raise ValueError(f"{type(model).__name__} does not support a static cache") from error
    token = torch.zeros(1, dtype=torch.int64, device=device)
    _model_step(model, cache, token, torch.tensor(0, device=device), max_cache_len)
    frozen = _python_state(cache)
    if max_cache_len > 1:
        _model_step(model, cache, token, torch.tensor(1, device=device), max_cache_len)
        after = _python_state(cache)
        changed = sorted(
            _state_name(path)
            for path in frozen.keys() | after.keys()
            if frozen.get(path) != after.get(path)
        )
        if changed:
            raise ValueError(
                "the cache keeps per-step state outside tensors, which a traced graph "
                f"would freeze: {', '.join(changed)}"
            )
    return cache


def _user_names(specs: Sequence[Any], kind: str) -> list[str]:
    return [
        spec.arg.name
        for spec in specs
        if getattr(spec.kind, "name", str(spec.kind)).rsplit(".", 1)[-1] == kind
    ]


def export_causal_lm(
    model: torch.nn.Module, *, max_cache_len: int, strict: bool = False
) -> CausalLMExport:
    """Export a one-token, fixed-capacity step program for a causal LM.

    The model is switched to evaluation mode and FP32 (the DLI tensor types),
    with gradients disabled.
    """

    if max_cache_len < 1:
        raise ValueError("max_cache_len must be positive")
    config = getattr(model, "config", None)
    if config is None or getattr(config, "is_encoder_decoder", False):
        raise TypeError("model must be a decoder-only Hugging Face causal LM")
    model.eval()
    model.to(dtype=torch.float32)
    model.requires_grad_(False)
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters") from error

    with torch.no_grad(), hub_kernel_fallbacks(model):
        template = _materialize_cache(model, max_cache_len, device)
        specs = tuple(
            StateSpec(_state_name(path), path, tuple(tensor.shape), tensor.dtype)
            for path, tensor in _cache_tensors(template)
        ) + (
            StateSpec(_CURSOR, (), (), torch.int64),
        )
        initializers = {
            spec.name: torch.zeros(spec.shape, dtype=spec.dtype, device=device) for spec in specs
        }
        step = CausalLMStep(model, template, specs, max_cache_len).eval()
        example = (
            torch.zeros(1, dtype=torch.int64, device=device),
            *(initializers[spec.name] for spec in specs),
        )
        program = torch.export.export(step, example, strict=strict)
        # Keep high-level ATen operators while functionalizing mutations.
        program = program.run_decompositions({})
    program.validate()

    input_names = _user_names(program.graph_signature.input_specs, "USER_INPUT")
    output_count = len(_user_names(program.graph_signature.output_specs, "USER_OUTPUT"))
    if len(input_names) != len(specs) + 1 or output_count != len(specs) + 1:
        raise RuntimeError("the exported step signature does not match the request state")
    output_names = ("logits", *(f"next_state.{spec.name}" for spec in specs))
    return CausalLMExport(
        program=program,
        step_module=step,
        model_type=f"{getattr(config, 'model_type', 'causal_lm')}_step",
        max_cache_len=max_cache_len,
        state_specs=specs,
        state_initializers=initializers,
        output_names=output_names,
        state_inputs={name: spec.name for name, spec in zip(input_names[1:], specs)},
        state_outputs={output: spec.name for output, spec in zip(output_names[1:], specs)},
    )


def causal_lm_to_dli(
    exported: CausalLMExport,
    output_dir: str | Path,
    *,
    stem: str,
    model_id: str | None = None,
) -> Path:
    """Write the step graph, weights, and `dli.autoregressive.v1` bundle manifest."""

    converter = importlib.import_module("dli_export.torch_export").exported_program_to_dli
    weight_writer = importlib.import_module("dli_export.weights").WeightWriter(output_dir, stem)
    graph_path, weights_path = converter(
        exported.program,
        output_dir,
        model_type=exported.model_type,
        stem=f"{stem}_step",
        output_names=exported.output_names,
        state_inputs=exported.state_inputs,
        state_outputs=exported.state_outputs,
        state_initializers=exported.state_initializers,
        weight_writer=weight_writer,
    )
    bundle: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "prefill_graph": graph_path.name,
        "decode_graph": graph_path.name,
        "weights": weights_path.name,
        "max_context_tokens": exported.max_cache_len,
        "input_shape": [1],
    }
    if model_id is not None:
        bundle["model_id"] = model_id
    bundle_path = Path(output_dir) / f"{stem}.dli.bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def export_causal_lm_to_dli(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    max_cache_len: int,
    stem: str = "model",
    model_id: str | None = None,
    strict: bool = False,
) -> Path:
    """Export an FP32 causal LM as a deployable autoregressive bundle."""

    exported = export_causal_lm(model, max_cache_len=max_cache_len, strict=strict)
    return causal_lm_to_dli(exported, output_dir, stem=stem, model_id=model_id)


def default_stem(model_id: str) -> str:
    """A file-name stem from a Hub ID or path, e.g. `Qwen/Qwen3-0.6B` -> `qwen3-0.6b`."""

    return re.sub(r"[^a-z0-9._-]+", "_", Path(model_id.rstrip("/")).name.lower()) or "model"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dli_export.causal_lm",
        description="Export a Hugging Face causal LM to a DLI autoregressive bundle.",
    )
    parser.add_argument("--model-id", required=True, help="Hub ID or local checkpoint path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-context-tokens", required=True, type=int)
    parser.add_argument("--stem", help="output file stem (default: derived from the model ID)")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--strict", action="store_true", help="use strict torch.export tracing")
    args = parser.parse_args(argv)

    model = _transformers().AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    bundle_path = export_causal_lm_to_dli(
        model,
        args.output_dir,
        max_cache_len=args.max_context_tokens,
        stem=args.stem or default_stem(args.model_id),
        model_id=args.model_id,
        strict=args.strict,
    )
    print(bundle_path)
    return 0


__all__ = [
    "CausalLMExport",
    "CausalLMStep",
    "StateSpec",
    "causal_lm_to_dli",
    "default_stem",
    "export_causal_lm",
    "export_causal_lm_to_dli",
    "hub_kernel_fallbacks",
]


if __name__ == "__main__":
    raise SystemExit(main())
