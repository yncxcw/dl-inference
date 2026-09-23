from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch


_LINEAR_ATTENTION = "linear_attention"
_FULL_ATTENTION = "full_attention"


@dataclass(frozen=True)
class Qwen3_5StateSpec:
    """One tensor in the flattened Qwen3.5 inference state."""

    name: str
    layer_index: int | None
    kind: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class Qwen3_5StageExport:
    """One exported generation stage and its DLI state-edge metadata."""

    name: str
    program: torch.export.ExportedProgram
    eager_module: torch.nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    output_names: tuple[str, ...]
    state_inputs: dict[str, str]
    state_outputs: dict[str, str]
    state_initializers: dict[str, torch.Tensor]


@dataclass
class Qwen3_5StaticCacheExport:
    """Fixed-capacity one-token prefill/decode stages for Qwen3.5.

    Both stages use the same recurrent ``torch.export`` program. At runtime the
    prefill phase starts with the graph's zero state initializers and folds the
    prompt one token at a time; decode continues from the published state. DLI
    hides the flattened state edges from ordinary request inputs, so each stage
    publicly consumes only ``input_ids`` with shape ``[batch]``.
    """

    prefill: Qwen3_5StageExport
    decode: Qwen3_5StageExport
    state_specs: tuple[Qwen3_5StateSpec, ...]
    state_initializers: dict[str, torch.Tensor]
    max_cache_len: int
    batch_size: int

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.state_specs)

    def initial_state(self) -> dict[str, torch.Tensor]:
        """Return a new named zero state suitable for an independent request."""

        return {name: value.clone() for name, value in self.state_initializers.items()}

    def flat_initial_state(self) -> tuple[torch.Tensor, ...]:
        """Return zero state tensors in the decode program's positional order."""

        return tuple(self.state_initializers[name].clone() for name in self.state_names)


def _require_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ImportError as error:
        raise ImportError(
            "Qwen3.5 export requires Transformers with Qwen3.5 support"
        ) from error


def _validate_model(
    model: torch.nn.Module, max_cache_len: int, batch_size: int
) -> None:
    transformers = _require_transformers()
    model_class = getattr(transformers, "Qwen3_5ForCausalLM", None)
    if model_class is None or not isinstance(model, model_class):
        raise TypeError("model must be a Qwen3_5ForCausalLM")
    if max_cache_len < 1:
        raise ValueError("max_cache_len must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    config = getattr(model, "config", None)
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        raise ValueError("Qwen3.5 config does not define layer_types")
    unsupported = set(layer_types) - {_LINEAR_ATTENTION, _FULL_ATTENTION}
    if unsupported:
        raise ValueError(f"unsupported Qwen3.5 layer types: {sorted(unsupported)}")
    if _FULL_ATTENTION not in layer_types:
        raise ValueError(
            "Qwen3.5 static export requires at least one full-attention layer"
        )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("Qwen3.5 model has no parameters") from error


def _state_specs(
    model: torch.nn.Module, max_cache_len: int, batch_size: int
) -> tuple[Qwen3_5StateSpec, ...]:
    config = model.config
    specs: list[Qwen3_5StateSpec] = []
    for layer_index, layer_type in enumerate(config.layer_types):
        if layer_type == _LINEAR_ATTENTION:
            key_dim = int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
            value_dim = int(config.linear_num_value_heads) * int(
                config.linear_value_head_dim
            )
            conv_dim = 2 * key_dim + value_dim
            specs.extend(
                (
                    Qwen3_5StateSpec(
                        name=f"layer.{layer_index}.linear.conv",
                        layer_index=layer_index,
                        kind="linear_conv",
                        shape=(
                            batch_size,
                            conv_dim,
                            int(config.linear_conv_kernel_dim),
                        ),
                        dtype=torch.float32,
                    ),
                    Qwen3_5StateSpec(
                        name=f"layer.{layer_index}.linear.recurrent",
                        layer_index=layer_index,
                        kind="linear_recurrent",
                        shape=(
                            batch_size,
                            int(config.linear_num_value_heads),
                            int(config.linear_key_head_dim),
                            int(config.linear_value_head_dim),
                        ),
                        dtype=torch.float32,
                    ),
                )
            )
        else:
            head_dim = int(
                getattr(
                    config,
                    "head_dim",
                    int(config.hidden_size) // int(config.num_attention_heads),
                )
            )
            kv_shape = (
                batch_size,
                int(config.num_key_value_heads),
                max_cache_len,
                head_dim,
            )
            specs.extend(
                (
                    Qwen3_5StateSpec(
                        name=f"layer.{layer_index}.attention.key",
                        layer_index=layer_index,
                        kind="attention_key",
                        shape=kv_shape,
                        dtype=torch.float32,
                    ),
                    Qwen3_5StateSpec(
                        name=f"layer.{layer_index}.attention.value",
                        layer_index=layer_index,
                        kind="attention_value",
                        shape=kv_shape,
                        dtype=torch.float32,
                    ),
                )
            )
    specs.append(
        Qwen3_5StateSpec(
            name="cursor",
            layer_index=None,
            kind="cursor",
            shape=(),
            dtype=torch.int64,
        )
    )
    return tuple(specs)


def _zero_initializers(
    specs: Sequence[Qwen3_5StateSpec], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        spec.name: torch.zeros(spec.shape, dtype=spec.dtype, device=device)
        for spec in specs
    }


@contextmanager
def _reference_qwen3_5_kernels():
    """Force Transformers' traceable PyTorch fallbacks during export."""

    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    )
    originals: dict[str, Any] = {}
    try:
        for name in names:
            original = getattr(module, name)
            originals[name] = original
            setattr(module, name, getattr(original, "__wrapped__", original))
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


class _Qwen3_5StepBase(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, max_cache_len: int) -> None:
        super().__init__()
        self.model = model
        self.max_cache_len = max_cache_len
        self._static_cache_type = _require_transformers().StaticCache

    def _new_cache(self):
        return self._static_cache_type(
            config=self.model.config,
            max_cache_len=self.max_cache_len,
        )

    @staticmethod
    def _token_matrix(input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.unsqueeze(-1)

    def _flat_cache_state(self, cache: Any) -> tuple[torch.Tensor, ...]:
        state: list[torch.Tensor] = []
        cursor: torch.Tensor | None = None
        for layer_type, layer in zip(self.model.config.layer_types, cache.layers):
            if layer_type == _LINEAR_ATTENTION:
                state.extend((layer.conv_states[0], layer.recurrent_states[0]))
            else:
                state.extend((layer.keys, layer.values))
                if cursor is None:
                    cursor = layer.cumulative_length
        if cursor is None:
            raise RuntimeError("Qwen3.5 cache has no full-attention cursor")
        state.append(cursor)
        return tuple(state)

    def _model_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Any,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        return outputs.logits


class _Qwen3_5DecodeStep(_Qwen3_5StepBase):
    def forward(
        self, input_ids: torch.Tensor, *flat_state: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        cache = self._new_cache()
        cursor = flat_state[-1]
        state_index = 0

        for layer_type, layer in zip(self.model.config.layer_types, cache.layers):
            first = flat_state[state_index]
            second = flat_state[state_index + 1]
            state_index += 2
            if layer_type == _LINEAR_ATTENTION:
                conv_state = first.clone()
                recurrent_state = second.clone()
                layer.conv_states[0] = conv_state
                layer.recurrent_states[0] = recurrent_state
                layer.is_conv_states_initialized[0] = True
                layer.is_recurrent_states_initialized[0] = True
                layer.has_previous_state[0] = True
                layer.conv_kernel_size[0] = conv_state.shape[-1]
                layer.device = conv_state.device
                layer.dtype = conv_state.dtype
            else:
                key_state = first.clone()
                value_state = second.clone()
                layer.keys = key_state
                layer.values = value_state
                layer.cumulative_length = cursor.clone()
                layer.is_initialized = True
                layer.device = key_state.device
                layer.dtype = key_state.dtype
                layer.batch_size = key_state.shape[0]
                layer.num_heads = key_state.shape[1]
                layer.k_head_dim = key_state.shape[-1]
                layer.v_head_dim = value_state.shape[-1]

        token_ids = self._token_matrix(input_ids)
        position_ids = cursor.reshape(1, 1).expand(input_ids.shape[0], 1)
        cache_positions = torch.arange(self.max_cache_len, device=input_ids.device)
        attention_mask = (
            (cache_positions <= cursor).unsqueeze(0).expand(input_ids.shape[0], -1)
        )
        logits = self._model_forward(token_ids, position_ids, attention_mask, cache)
        return (logits, *self._flat_cache_state(cache))


def _kind_name(kind: Any) -> str:
    return getattr(kind, "name", str(kind).rsplit(".", 1)[-1])


def _argument_name(argument: Any) -> str:
    name = getattr(argument, "name", None)
    if not isinstance(name, str):
        raise RuntimeError("Qwen3.5 export produced a non-tensor graph argument")
    return name


def _user_argument_names(specs: Sequence[Any], expected_kind: str) -> tuple[str, ...]:
    return tuple(
        _argument_name(spec.arg)
        for spec in specs
        if _kind_name(spec.kind) == expected_kind
    )


def _export_stage(
    name: str,
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    state_specs: Sequence[Qwen3_5StateSpec],
    state_initializers: Mapping[str, torch.Tensor],
    *,
    strict: bool,
) -> Qwen3_5StageExport:
    with _reference_qwen3_5_kernels(), torch.no_grad():
        program = torch.export.export(module, example_inputs, strict=strict)
        # Keep the high-level ATen operators while functionalizing mutations.
        program = program.run_decompositions({})
    program.validate()

    input_names = _user_argument_names(
        program.graph_signature.input_specs, "USER_INPUT"
    )
    user_outputs = _user_argument_names(
        program.graph_signature.output_specs, "USER_OUTPUT"
    )
    if len(user_outputs) != len(state_specs) + 1:
        raise RuntimeError(
            f"{name} export returned {len(user_outputs)} tensors; "
            f"expected logits plus {len(state_specs)} state tensors"
        )

    if len(input_names) != len(state_specs) + 1:
        raise RuntimeError(
            f"{name} export input signature does not match the flattened state"
        )
    state_inputs = {
        input_name: spec.name
        for input_name, spec in zip(input_names[1:], state_specs)
    }
    stage_initializers = dict(state_initializers)

    output_names = (
        "logits",
        *(f"next_state.{spec.name}" for spec in state_specs),
    )
    state_outputs = {
        output_name: spec.name
        for output_name, spec in zip(output_names[1:], state_specs)
    }
    return Qwen3_5StageExport(
        name=name,
        program=program,
        eager_module=module,
        example_inputs=example_inputs,
        output_names=output_names,
        state_inputs=state_inputs,
        state_outputs=state_outputs,
        state_initializers=stage_initializers,
    )


def export_qwen3_5_static_cache(
    model: torch.nn.Module,
    *,
    max_cache_len: int,
    batch_size: int = 1,
    strict: bool = False,
) -> Qwen3_5StaticCacheExport:
    """Export a fixed-shape recurrent program for prefill and decode.

    The given model is switched to evaluation mode, converted to FP32, and has
    gradients disabled. The exported graph takes a flat state, but the returned
    state-edge maps let the DLI converter remove those tensors from the ordinary
    request input list and seed them after a request reset.

    Qwen3.5's empty-cache implementation selects its chunked DeltaNet path even
    for a single token, producing a much larger graph than the recurrent path.
    Supplying explicit zero state is numerically equivalent for token zero and
    lets one compact functional program serve both lifecycle stages.
    """

    _require_transformers()
    _validate_model(model, max_cache_len, batch_size)
    model.eval()
    model.to(dtype=torch.float32)
    model.requires_grad_(False)

    specs = _state_specs(model, max_cache_len, batch_size)
    initializers = _zero_initializers(specs, _model_device(model))
    example_ids = torch.zeros(
        batch_size, dtype=torch.int64, device=_model_device(model)
    )

    decode_module = _Qwen3_5DecodeStep(model, max_cache_len).eval()
    decode_inputs = (
        example_ids,
        *(initializers[spec.name] for spec in specs),
    )
    decode = _export_stage(
        "decode",
        decode_module,
        decode_inputs,
        specs,
        initializers,
        strict=strict,
    )
    # Prefill is a state-lifecycle phase, not a different compute kernel in the
    # correctness-first tokenwise implementation. Keep an explicit stage object
    # so callers can replace it with a future chunked program without changing
    # the bundle or generation APIs.
    prefill = replace(decode, name="prefill")
    return Qwen3_5StaticCacheExport(
        prefill=prefill,
        decode=decode,
        state_specs=specs,
        state_initializers=initializers,
        max_cache_len=max_cache_len,
        batch_size=batch_size,
    )


def qwen3_5_stage_to_dli(
    stage: Qwen3_5StageExport,
    output_dir: str | Path,
    *,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Write one prepared stage through the generic ExportedProgram converter.

    This helper is useful for inspecting one stage. Use
    :func:`qwen3_5_static_cache_to_dli` for a deployable bundle with shared
    lifecycle graph and model weights.
    """

    converter_module = importlib.import_module("dli_export.torch_export")
    converter = getattr(converter_module, "exported_program_to_dli", None)
    if converter is None:
        # Compatibility with the converter's development name.
        converter = converter_module.export_program
    return converter(
        stage.program,
        output_dir,
        model_type=f"qwen3_5_{stage.name}",
        stem=stem or f"qwen3_5_{stage.name}",
        output_names=stage.output_names,
        state_inputs=stage.state_inputs,
        state_outputs=stage.state_outputs,
        state_initializers=stage.state_initializers,
    )


def qwen3_5_static_cache_to_dli(
    exported: Qwen3_5StaticCacheExport,
    output_dir: str | Path,
    *,
    stem: str = "qwen3_5",
    model_id: str | None = None,
) -> tuple[dict[str, Path], Path]:
    """Write the shared step graph, weights, and autoregressive bundle manifest."""

    if exported.batch_size != 1:
        raise ValueError("Qwen3.5 chatbot bundles currently require batch_size=1")

    converter_module = importlib.import_module("dli_export.torch_export")
    converter = getattr(converter_module, "exported_program_to_dli", None)
    if converter is None:
        converter = converter_module.export_program
    weight_module = importlib.import_module("dli_export.weights")
    weight_writer = weight_module.WeightWriter(output_dir, stem)

    stage = exported.decode
    graph_path, weights_path = converter(
        stage.program,
        output_dir,
        model_type="qwen3_5_step",
        stem=f"{stem}_step",
        output_names=stage.output_names,
        state_inputs=stage.state_inputs,
        state_outputs=stage.state_outputs,
        state_initializers=stage.state_initializers,
        weight_writer=weight_writer,
    )
    graph_paths = {"prefill": graph_path, "decode": graph_path}

    bundle = {
        "format": "dli.autoregressive.v1",
        "prefill_graph": graph_paths["prefill"].name,
        "decode_graph": graph_paths["decode"].name,
        "weights": weights_path.name,
        "max_context_tokens": exported.max_cache_len,
        "input_shape": [exported.batch_size],
    }
    if model_id is not None:
        bundle["model_id"] = model_id
    bundle_path = Path(output_dir) / f"{stem}.dli.bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return graph_paths, weights_path


def export_qwen3_5_to_dli(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    max_cache_len: int,
    stem: str = "qwen3_5",
    model_id: str | None = None,
    batch_size: int = 1,
    strict: bool = False,
) -> Path:
    """Export an FP32 Qwen3.5 model as a deployable autoregressive bundle."""

    if batch_size != 1:
        raise ValueError("Qwen3.5 chatbot bundles currently require batch_size=1")

    exported = export_qwen3_5_static_cache(
        model,
        max_cache_len=max_cache_len,
        batch_size=batch_size,
        strict=strict,
    )
    qwen3_5_static_cache_to_dli(
        exported,
        output_dir,
        stem=stem,
        model_id=model_id,
    )
    return Path(output_dir) / f"{stem}.dli.bundle.json"


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Qwen3.5 to a fixed-capacity DLI autoregressive graph"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-context-tokens", required=True, type=int)
    parser.add_argument("--stem", default="qwen3_5")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    transformers = _require_transformers()
    model = transformers.Qwen3_5ForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        dtype=torch.float32,
    )
    bundle_path = export_qwen3_5_to_dli(
        model,
        args.output_dir,
        max_cache_len=args.max_context_tokens,
        stem=args.stem,
        model_id=args.model_id,
        strict=args.strict,
    )
    print(bundle_path)
    return 0


__all__ = [
    "Qwen3_5StageExport",
    "Qwen3_5StateSpec",
    "Qwen3_5StaticCacheExport",
    "export_qwen3_5_to_dli",
    "export_qwen3_5_static_cache",
    "qwen3_5_stage_to_dli",
    "qwen3_5_static_cache_to_dli",
]


if __name__ == "__main__":
    raise SystemExit(_main())
