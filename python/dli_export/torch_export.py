from __future__ import annotations

import json
import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.fx

from .weights import WeightWriter


class TorchExportError(RuntimeError):
    """Raised when an ExportedProgram cannot be represented by the DLI graph IR."""


StateInitializer = str | torch.Tensor
_SUPPORTED_TENSOR_DTYPES = frozenset((torch.bool, torch.float32, torch.int64))


def _require_supported_tensor_dtype(tensor: torch.Tensor, description: str) -> None:
    if tensor.dtype not in _SUPPORTED_TENSOR_DTYPES:
        supported = ", ".join(sorted(str(dtype) for dtype in _SUPPORTED_TENSOR_DTYPES))
        raise TorchExportError(
            f"{description} uses unsupported tensor dtype {tensor.dtype}; "
            f"DLI currently preserves only {supported}"
        )


@dataclass
class _ConversionContext:
    program: torch.export.ExportedProgram
    weights: WeightWriter
    graph_stem: str
    env: dict[torch.fx.Node, Any] = field(default_factory=dict)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    graph_inputs: list[str] = field(default_factory=list)
    tensor_anchors: list[tuple[str, torch.device]] = field(default_factory=list)
    literal_index: int = 0

    def add_weight(self, name: str, tensor: torch.Tensor) -> str:
        _require_supported_tensor_dtype(tensor, f"weight {name!r}")
        return self.weights.add(name, tensor)

    def add_tensor_input(self, inputs: list[str], name: str) -> int:
        inputs.append(name)
        return len(inputs) - 1


def _kind_name(kind: Any) -> str:
    return getattr(kind, "name", str(kind).rsplit(".", 1)[-1])


def _argument_name(argument: Any) -> str | None:
    name = getattr(argument, "name", None)
    return name if isinstance(name, str) else None


def _is_tensor_meta(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _iter_tensor_meta(value: Any):
    if _is_tensor_meta(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensor_meta(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensor_meta(item)


def _iter_tensor_names(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensor_names(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensor_names(item)


def _resolve_lifted_tensor(
    program: torch.export.ExportedProgram, target: Any, placeholder_name: str
) -> torch.Tensor:
    candidates = [target, placeholder_name]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        if candidate in program.state_dict:
            value = program.state_dict[candidate]
            if isinstance(value, torch.Tensor):
                return value
        if candidate in program.constants:
            value = program.constants[candidate]
            if isinstance(value, torch.Tensor):
                return value
    raise TorchExportError(
        f"could not resolve lifted tensor for placeholder {placeholder_name!r} "
        f"(target={target!r})"
    )


def _get_attr(module: torch.nn.Module, target: str) -> Any:
    value: Any = module
    for part in target.split("."):
        value = getattr(value, part)
    return value


def _unwrap_optional(type_name: str) -> tuple[str, bool]:
    prefix = "Optional["
    if type_name.startswith(prefix) and type_name.endswith("]"):
        return type_name[len(prefix) : -1], True
    return type_name, False


def _node_meta_value(value: Any) -> Any:
    return value.meta.get("val") if isinstance(value, torch.fx.Node) else value


def _schema_value_compatible(value: Any, schema_type: Any) -> bool:
    value = _node_meta_value(value)
    type_name, optional = _unwrap_optional(str(schema_type))
    if value is None:
        return optional

    if type_name.startswith("List[") and type_name.endswith("]"):
        if not isinstance(value, (tuple, list)):
            return False
        element_type = type_name[5:-1]
        return all(_schema_value_compatible(item, element_type) for item in value)

    if type_name == "Tensor":
        return _is_tensor_meta(value)
    # ScalarType, Layout, and MemoryFormat are all represented as int in an
    # ATen FunctionSchema even though their Python values retain distinct types.
    if isinstance(value, (torch.dtype, torch.layout, torch.memory_format)):
        return type_name == "int"
    if type_name in ("number", "Scalar"):
        return isinstance(value, (bool, int, float, complex))
    if type_name in ("int", "SymInt"):
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name in ("float", "SymFloat"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "Device":
        return isinstance(value, torch.device)
    return False


def _bind_schema(target: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> list[Any] | None:
    schema = target._schema
    schema_args = list(schema.arguments)
    if len(args) > len(schema_args):
        return None

    values: list[Any] = []
    consumed_kwargs: set[str] = set()
    for index, schema_arg in enumerate(schema_args):
        if index < len(args):
            if schema_arg.name in kwargs:
                return None
            value = args[index]
        elif schema_arg.name in kwargs:
            value = kwargs[schema_arg.name]
            consumed_kwargs.add(schema_arg.name)
        elif schema_arg.has_default_value():
            value = schema_arg.default_value
        else:
            return None
        if not _schema_value_compatible(value, schema_arg.type):
            return None
        values.append(value)

    if set(kwargs) != consumed_kwargs:
        return None
    return values


def _compatibility_score(values: Sequence[Any], target: Any) -> int:
    score = 0
    for value, schema_arg in zip(values, target._schema.arguments):
        actual = _node_meta_value(value)
        expected, optional = _unwrap_optional(str(schema_arg.type))
        if optional:
            score -= 1
        if _is_tensor_meta(actual) and expected == "Tensor":
            score += 20
        elif isinstance(actual, bool) and expected == "bool":
            score += 10
        elif (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and expected in ("int", "SymInt")
        ):
            score += 10
        elif isinstance(actual, float) and expected in ("float", "SymFloat"):
            score += 10
        elif isinstance(actual, (int, float, bool)) and expected in ("number", "Scalar"):
            score += 8
        elif isinstance(actual, (tuple, list)) and expected.startswith("List["):
            score += 5
    return score


def _resolve_overload(
    target: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Any, list[Any]]:
    bound = _bind_schema(target, args, kwargs)
    if bound is not None:
        return target, bound

    packet = getattr(target, "overloadpacket", None)
    if packet is None:
        raise TorchExportError(f"arguments do not match exported ATen schema: {target._schema}")

    matches: list[tuple[int, str, Any, list[Any]]] = []
    for overload_name in packet.overloads():
        candidate = getattr(packet, overload_name)
        candidate_bound = _bind_schema(candidate, args, kwargs)
        if candidate_bound is None:
            continue
        matches.append(
            (
                _compatibility_score(candidate_bound, candidate),
                candidate._schema.overload_name,
                candidate,
                candidate_bound,
            )
        )
    if not matches:
        rendered = f"args={args!r}, kwargs={dict(kwargs)!r}"
        raise TorchExportError(
            f"no ATen overload accepts exported arguments for {target._schema.name}: {rendered}"
        )
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        names = ", ".join(item[1] or "default" for item in matches if item[0] == matches[0][0])
        raise TorchExportError(
            f"ambiguous ATen overload for {target._schema.name}; matching overloads: {names}"
        )
    _, _, resolved, resolved_bound = matches[0]
    return resolved, resolved_bound


def _schema_element_type(schema_type: Any) -> str | None:
    type_name, _ = _unwrap_optional(str(schema_type))
    if type_name.startswith("List[") and type_name.endswith("]"):
        return type_name[5:-1]
    return None


def _encode_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return repr(value)


def _memory_format_name(value: torch.memory_format) -> str:
    names = {
        torch.contiguous_format: "contiguous",
        torch.preserve_format: "preserve",
        torch.channels_last: "channels_last",
        torch.channels_last_3d: "channels_last_3d",
    }
    try:
        return names[value]
    except KeyError as error:
        raise TorchExportError(f"unsupported torch memory format: {value}") from error


def _device_matches(lhs: torch.device, rhs: torch.device) -> bool:
    if lhs.type != rhs.type:
        return False
    if lhs.type != "cuda":
        return True
    lhs_index = 0 if lhs.index is None else lhs.index
    rhs_index = 0 if rhs.index is None else rhs.index
    return lhs_index == rhs_index


def _local_anchor_candidates(value: Any, context: _ConversionContext):
    if isinstance(value, torch.fx.Node):
        resolved = context.env.get(value)
        meta_tensors = list(_iter_tensor_meta(value.meta.get("val")))
        names = list(_iter_tensor_names(resolved))
        for name, meta in zip(names, meta_tensors):
            yield name, meta.device
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _local_anchor_candidates(item, context)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _local_anchor_candidates(item, context)


def _device_anchor(device: torch.device, source_values: Any, context: _ConversionContext) -> str:
    candidates = list(_local_anchor_candidates(source_values, context))
    candidates.extend(context.tensor_anchors)
    for name, candidate_device in candidates:
        if _device_matches(device, candidate_device):
            return name
    raise TorchExportError(
        f"cannot encode device {device}: no earlier tensor on that exported device is available "
        "as a runtime anchor"
    )


def _literal_tensor_name(value: torch.Tensor, context: _ConversionContext) -> str:
    name = f"{context.graph_stem}.__torch_export_literal_{context.literal_index}"
    context.literal_index += 1
    return context.add_weight(name, value)


def _resolved_tensor_names(value: Any, context: _ConversionContext) -> list[str | None]:
    if isinstance(value, torch.fx.Node):
        if value not in context.env:
            raise TorchExportError(f"FX value is used before it is defined: {value.name}")
        resolved = context.env[value]
        if isinstance(resolved, str):
            return [resolved]
        if isinstance(resolved, (tuple, list)):
            result: list[str | None] = []
            for item in resolved:
                if item is None:
                    result.append(None)
                elif isinstance(item, str):
                    result.append(item)
                else:
                    raise TorchExportError(f"unsupported nested FX tensor result: {value.name}")
            return result
        raise TorchExportError(f"FX value is not tensor-backed: {value.name}")
    if isinstance(value, torch.Tensor):
        return [_literal_tensor_name(value, context)]
    if value is None:
        return [None]
    raise TorchExportError(f"expected a tensor value, got {type(value).__name__}")


def _encode_argument(
    value: Any,
    schema_type: Any,
    inputs: list[str],
    all_source_values: Any,
    context: _ConversionContext,
) -> str:
    if value is None:
        return "n"

    type_name, _ = _unwrap_optional(str(schema_type))
    element_type = _schema_element_type(schema_type)

    if isinstance(value, torch.fx.Node) or isinstance(value, torch.Tensor):
        names = _resolved_tensor_names(value, context)
        if element_type is not None:
            optional_tensors = element_type == "Optional[Tensor]"
            if element_type not in ("Tensor", "Optional[Tensor]"):
                raise TorchExportError(
                    f"cannot encode tensor-backed value for schema type {schema_type}"
                )
            encoded: list[str] = []
            for name in names:
                if name is None:
                    if not optional_tensors:
                        raise TorchExportError("None appears in a non-optional tensor list")
                    encoded.append("n")
                else:
                    encoded.append(str(context.add_tensor_input(inputs, name)))
            return ("otl:" if optional_tensors else "tl:") + ",".join(encoded)
        if len(names) != 1 or names[0] is None:
            raise TorchExportError(f"schema argument {schema_type} requires one tensor")
        return f"t:{context.add_tensor_input(inputs, names[0])}"

    if isinstance(value, (tuple, list)):
        if element_type is None:
            raise TorchExportError(f"list value does not match schema type {schema_type}")
        if element_type in ("Tensor", "Optional[Tensor]"):
            optional_tensors = element_type == "Optional[Tensor]"
            encoded = []
            for item in value:
                names = _resolved_tensor_names(item, context)
                if len(names) != 1:
                    raise TorchExportError("nested tensor lists are not supported")
                name = names[0]
                if name is None:
                    if not optional_tensors:
                        raise TorchExportError("None appears in a non-optional tensor list")
                    encoded.append("n")
                else:
                    encoded.append(str(context.add_tensor_input(inputs, name)))
            return ("otl:" if optional_tensors else "tl:") + ",".join(encoded)
        if element_type in ("int", "SymInt"):
            return "il:" + ",".join(str(int(item)) for item in value)
        if element_type in ("float", "SymFloat"):
            return "fl:" + ",".join(_encode_float(float(item)) for item in value)
        if element_type == "bool":
            return "bl:" + ",".join("true" if item else "false" for item in value)
        raise TorchExportError(f"unsupported list schema argument: {schema_type}")

    if isinstance(value, torch.dtype):
        return f"dtype:{str(value).removeprefix('torch.')}"
    if isinstance(value, torch.layout):
        layout = str(value).removeprefix("torch.")
        if layout != "strided":
            raise TorchExportError(f"unsupported torch layout: {value}")
        return "layout:strided"
    if isinstance(value, torch.memory_format):
        return f"memory_format:{_memory_format_name(value)}"
    if isinstance(value, torch.device):
        anchor = _device_anchor(value, all_source_values, context)
        return f"device:{context.add_tensor_input(inputs, anchor)}"
    if isinstance(value, bool):
        return "b:true" if value else "b:false"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return f"f:{_encode_float(value)}"
    if isinstance(value, str):
        return f"s:{value}"
    if isinstance(value, complex):
        raise TorchExportError("complex scalar ATen arguments are not supported")
    raise TorchExportError(
        f"unsupported ATen argument value {value!r} ({type(value).__name__}) for {type_name}"
    )


def _result_template(meta: Any, base_name: str, path: tuple[int, ...] = ()) -> Any:
    if _is_tensor_meta(meta):
        _require_supported_tensor_dtype(meta, f"ATen result {base_name!r}")
        suffix = "" if not path else "_" + "_".join(str(index) for index in path)
        return base_name + suffix
    if meta is None:
        return None
    if isinstance(meta, tuple):
        return tuple(
            _result_template(item, base_name, path + (index,)) for index, item in enumerate(meta)
        )
    if isinstance(meta, list):
        return [
            _result_template(item, base_name, path + (index,)) for index, item in enumerate(meta)
        ]
    raise TorchExportError(
        f"ATen node {base_name!r} returns unsupported non-tensor value {meta!r}; "
        "dynamic SymInt/SymFloat execution is not yet supported"
    )


def _flatten_result(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_result(item))
        return result
    raise TorchExportError(f"unsupported result structure: {value!r}")


def _flatten_fx_output(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten_fx_output(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_fx_output(item))
        return result
    return [value]


def _unique_weight_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    index = 1
    while f"{base}_{index}" in used:
        index += 1
    result = f"{base}_{index}"
    used.add(result)
    return result


def exported_program_to_dli(
    program: torch.export.ExportedProgram,
    output_dir: str | Path,
    *,
    model_type: str = "pytorch_export",
    stem: str = "model",
    output_names: Sequence[str] | None = None,
    state_inputs: Mapping[str, str] | None = None,
    state_outputs: Mapping[str, str] | None = None,
    state_initializers: Mapping[str, StateInitializer] | None = None,
    weight_writer: WeightWriter | None = None,
) -> tuple[Path, Path]:
    """Convert a ``torch.export.ExportedProgram`` into DLI graph and weight files.

    The converter keeps computation model-agnostic: every call is serialized as
    one generic ``aten`` node. It currently targets fixed-shape inference graphs;
    tensor-valued parameters, buffers, and lifted constants are written to the
    normal DLI weights file.

    ``state_inputs`` maps graph input names to request-state keys,
    ``state_outputs`` maps produced tensor names to request-state keys, and
    ``state_initializers`` maps state keys to either an existing weight name or
    a tensor that should be added to the generated weights file. Pass one
    ``weight_writer`` to multiple conversions to share a weights manifest
    between independently named prefill and decode graphs.
    """

    if not isinstance(program, torch.export.ExportedProgram):
        raise TypeError("program must be a torch.export.ExportedProgram")

    # This pass gives us functional ATen IR, removes higher-order wrappers, and
    # makes user-input/buffer mutation explicit in GraphSignature outputs.
    exported = program.run_decompositions({})
    exported.validate()
    graph_module = exported.graph_module
    signature = exported.graph_signature

    destination = Path(output_dir)
    weights = weight_writer or WeightWriter(destination, stem)
    if weights.output_dir.resolve() != destination.resolve():
        raise ValueError("weight_writer.output_dir must match output_dir")
    context = _ConversionContext(exported, weights, stem)

    input_specs: dict[str, Any] = {}
    for spec in signature.input_specs:
        name = _argument_name(spec.arg)
        if name is not None:
            input_specs[name] = spec

    # Lifted tensor targets use source-module names (for example ``weight``),
    # while FX edges use placeholder/node names. Reserve every FX result name
    # before assigning weight names so a parameter cannot silently alias a user
    # input or an intermediate tensor in DLI's single tensor namespace.
    used_weight_names = {fx_node.name for fx_node in graph_module.graph.nodes}
    for fx_node in graph_module.graph.nodes:
        if fx_node.op == "call_function":
            used_weight_names.update(
                _flatten_result(_result_template(fx_node.meta.get("val"), fx_node.name))
            )
    output_fx_node: torch.fx.Node | None = None

    for fx_node in graph_module.graph.nodes:
        if fx_node.op == "placeholder":
            spec = input_specs.get(fx_node.name)
            if spec is None:
                raise TorchExportError(f"missing GraphSignature input spec for {fx_node.name}")
            kind = _kind_name(spec.kind)
            meta = fx_node.meta.get("val")
            if kind == "USER_INPUT":
                if not _is_tensor_meta(meta):
                    # torch.export retains placeholders for specialized Python
                    # arguments such as attention_mask=None or
                    # logits_to_keep=1. They are constants in this program, not
                    # runtime tensor inputs, and are normally unused in ATen IR.
                    context.env[fx_node] = meta
                    continue
                _require_supported_tensor_dtype(meta, f"user input {fx_node.name!r}")
                name = fx_node.name
                context.env[fx_node] = name
                context.graph_inputs.append(name)
                context.tensor_anchors.append((name, meta.device))
                continue
            if kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                tensor = _resolve_lifted_tensor(exported, spec.target, fx_node.name)
                base_name = str(spec.target) if spec.target is not None else fx_node.name
                if kind == "CONSTANT_TENSOR":
                    # Captured tensor literals have exporter-local names (often
                    # lifted_tensor_0), so namespace them across paired graphs.
                    base_name = f"{stem}.{base_name}"
                weight_name = _unique_weight_name(base_name, used_weight_names)
                context.env[fx_node] = context.add_weight(weight_name, tensor)
                if _is_tensor_meta(meta):
                    context.tensor_anchors.append((weight_name, meta.device))
                continue
            raise TorchExportError(
                f"unsupported ExportedProgram input kind {kind} for {fx_node.name!r}"
            )

        if fx_node.op == "get_attr":
            value = _get_attr(graph_module, str(fx_node.target))
            if not isinstance(value, torch.Tensor):
                raise TorchExportError(f"get_attr {fx_node.target!r} is not a tensor")
            weight_name = _unique_weight_name(str(fx_node.target), used_weight_names)
            context.env[fx_node] = context.add_weight(weight_name, value)
            meta = fx_node.meta.get("val")
            if _is_tensor_meta(meta):
                context.tensor_anchors.append((weight_name, meta.device))
            continue

        if fx_node.op == "call_function" and fx_node.target is operator.getitem:
            source = context.env.get(fx_node.args[0])
            index = fx_node.args[1]
            if not isinstance(source, (tuple, list)) or not isinstance(index, int):
                raise TorchExportError(
                    f"built-in getitem {fx_node.name!r} is not selecting an ATen tuple/list result"
                )
            try:
                context.env[fx_node] = source[index]
            except IndexError as error:
                raise TorchExportError(
                    f"getitem index is out of range at {fx_node.name!r}"
                ) from error
            continue

        if fx_node.op == "call_function":
            target = fx_node.target
            if not isinstance(target, torch._ops.OpOverload):
                raise TorchExportError(
                    f"non-ATen call remains after decomposition: {fx_node.name} ({target!r})"
                )
            if not target._schema.name.startswith("aten::"):
                raise TorchExportError(
                    f"non-ATen dispatcher operator is not supported: {target._schema.name}"
                )
            if target._schema.name == "aten::_assert_tensor_metadata":
                # Export-only assertion. DLI validates actual dispatcher calls,
                # and this op has no tensor result for the tensor-only engine.
                context.env[fx_node] = None
                continue

            resolved_target, ordered_values = _resolve_overload(
                target, tuple(fx_node.args), fx_node.kwargs
            )
            inputs: list[str] = []
            arguments = [
                _encode_argument(
                    value,
                    schema_arg.type,
                    inputs,
                    (fx_node.args, fx_node.kwargs),
                    context,
                )
                for value, schema_arg in zip(ordered_values, resolved_target._schema.arguments)
            ]
            result = _result_template(fx_node.meta.get("val"), fx_node.name)
            outputs = _flatten_result(result)
            if not outputs:
                raise TorchExportError(
                    f"ATen operator {resolved_target._schema} has no tensor output"
                )
            context.env[fx_node] = result
            for name, meta in zip(outputs, _iter_tensor_meta(fx_node.meta.get("val"))):
                context.tensor_anchors.append((name, meta.device))
            context.nodes.append(
                {
                    "name": fx_node.name,
                    "op": "aten",
                    "inputs": inputs,
                    "outputs": outputs,
                    "attrs": {
                        "name": resolved_target._schema.name,
                        "overload": resolved_target._schema.overload_name,
                        "arguments": arguments,
                    },
                }
            )
            continue

        if fx_node.op == "output":
            output_fx_node = fx_node
            continue

        raise TorchExportError(f"unsupported FX node: {fx_node.op} ({fx_node.name})")

    if output_fx_node is None:
        raise TorchExportError("ExportedProgram graph has no output node")
    flat_outputs = _flatten_fx_output(output_fx_node.args[0])
    output_specs = list(signature.output_specs)
    if len(flat_outputs) != len(output_specs):
        raise TorchExportError(
            "ExportedProgram graph output structure does not match GraphSignature"
        )

    external_outputs: list[str] = []
    for fx_value, spec in zip(flat_outputs, output_specs):
        if _kind_name(spec.kind) != "USER_OUTPUT":
            continue
        if not isinstance(fx_value, torch.fx.Node):
            raise TorchExportError("non-tensor constant user outputs are not supported")
        resolved = context.env.get(fx_value)
        if not isinstance(resolved, str):
            raise TorchExportError(f"user output {fx_value.name!r} does not resolve to one tensor")
        external_outputs.append(resolved)

    if output_names is None:
        desired_outputs = ["logits"] if len(external_outputs) == 1 else list(external_outputs)
    else:
        desired_outputs = list(output_names)
        if len(desired_outputs) != len(external_outputs):
            raise TorchExportError(
                f"output_names has {len(desired_outputs)} entries for "
                f"{len(external_outputs)} user outputs"
            )
    if len(set(desired_outputs)) != len(desired_outputs):
        raise TorchExportError("output_names must be unique")

    # A global textual rename is safe only when one output owns its source edge
    # and the requested name is new. Otherwise it can overwrite a live
    # intermediate (or collapse two returns of the same tensor). Materialize
    # those cases as terminal aten::alias nodes, after every consumer has run.
    existing_edges = set(context.graph_inputs) | set(weights.tensors)
    produced_edges: set[str] = set()
    for node in context.nodes:
        existing_edges.update(node["inputs"])
        existing_edges.update(node["outputs"])
        produced_edges.update(node["outputs"])

    source_counts = Counter(external_outputs)
    external_source_edges = set(external_outputs)
    rename: dict[str, str] = {}
    terminal_aliases: list[tuple[str, str]] = []
    for source, destination_name in zip(external_outputs, desired_outputs):
        if source == destination_name:
            continue
        if (
            source in produced_edges
            and source_counts[source] == 1
            and destination_name not in existing_edges
        ):
            rename[source] = destination_name
            existing_edges.add(destination_name)
            continue
        if destination_name in external_source_edges:
            raise TorchExportError(
                f"output name {destination_name!r} aliases another returned tensor edge; "
                "choose a fresh output name"
            )
        terminal_aliases.append((source, destination_name))

    for node in context.nodes:
        node["inputs"] = [rename.get(name, name) for name in node["inputs"]]
        node["outputs"] = [rename.get(name, name) for name in node["outputs"]]
    used_node_names = {node["name"] for node in context.nodes}
    for index, (source, destination_name) in enumerate(terminal_aliases):
        alias_name = f"__dli_output_alias_{index}"
        while alias_name in used_node_names:
            alias_name += "_"
        used_node_names.add(alias_name)
        context.nodes.append(
            {
                "name": alias_name,
                "op": "aten",
                "inputs": [rename.get(source, source)],
                "outputs": [destination_name],
                "attrs": {
                    "name": "aten::alias",
                    "overload": "",
                    "arguments": ["t:0"],
                },
            }
        )

    # State declarations are already expressed in final DLI edge names. This
    # lets callers alias all USER_OUTPUT leaves (including cache tensors) with
    # output_names and then mark the chosen aliases as request state.
    normalized_state_inputs = dict(state_inputs or {})
    normalized_state_outputs = dict(state_outputs or {})
    unknown_state_inputs = set(normalized_state_inputs) - set(context.graph_inputs)
    if unknown_state_inputs:
        raise TorchExportError(
            "state_inputs names are not tensor USER_INPUT edges: "
            + ", ".join(sorted(unknown_state_inputs))
        )
    produced_edges = {output_name for node in context.nodes for output_name in node["outputs"]}
    unknown_state_outputs = set(normalized_state_outputs) - produced_edges
    if unknown_state_outputs:
        raise TorchExportError(
            "state_outputs names are not produced DLI edges: "
            + ", ".join(sorted(unknown_state_outputs))
        )

    normalized_initializers: dict[str, str] = {}
    for state_key, initializer in (state_initializers or {}).items():
        if isinstance(initializer, str):
            normalized_initializers[state_key] = initializer
        elif isinstance(initializer, torch.Tensor):
            base_name = f"{stem}.__state_initializer__.{state_key}"
            weight_name = _unique_weight_name(base_name, used_weight_names)
            normalized_initializers[state_key] = context.add_weight(weight_name, initializer)
        else:
            raise TypeError(f"state initializer for {state_key!r} must be a tensor or weight name")

    graph_inputs = [name for name in context.graph_inputs if name not in normalized_state_inputs]
    weights_path = weights.write()
    public_outputs = [name for name in desired_outputs if name not in normalized_state_outputs]
    graph = {
        "format": "dli.graph.v1",
        "model_type": model_type,
        "weights": weights_path.name,
        "inputs": graph_inputs,
        "outputs": public_outputs,
        "nodes": context.nodes,
    }
    if normalized_state_inputs:
        graph["state_inputs"] = normalized_state_inputs
    if normalized_state_outputs:
        graph["state_outputs"] = normalized_state_outputs
    if normalized_initializers:
        graph["state_initializers"] = normalized_initializers

    destination.mkdir(parents=True, exist_ok=True)
    graph_path = destination / f"{stem}.dli.json"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return graph_path, weights_path


def export_program(
    program: torch.export.ExportedProgram,
    output_dir: str | Path,
    **kwargs: Any,
) -> tuple[Path, Path]:
    """Backward-compatible short alias for :func:`exported_program_to_dli`."""

    return exported_program_to_dli(program, output_dir, **kwargs)


__all__ = ["TorchExportError", "export_program", "exported_program_to_dli"]
