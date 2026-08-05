from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from .weights import WeightWriter


class ExportError(RuntimeError):
    pass


def _node(
    nodes: list[dict[str, Any]],
    name: str,
    op: str,
    inputs: list[str],
    outputs: list[str],
    attrs: dict[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {
        "name": name,
        "op": op,
        "inputs": inputs,
        "outputs": outputs,
    }
    if attrs:
        item["attrs"] = attrs
    nodes.append(item)


def _write_graph(output_dir: Path, stem: str, graph: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.dli.json"
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return path


def _shape_from_meta(node: torch.fx.Node) -> list[int]:
    meta = node.meta.get("tensor_meta")
    if meta is None:
        raise ExportError(f"missing tensor metadata for node {node.name}")
    return list(meta.shape)


def _export_fx(model: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...],
               output_dir: Path, model_type: str, stem: str) -> tuple[Path, Path]:
    if not example_inputs:
        raise ExportError("FX export requires at least one example input")
    traced = torch.fx.symbolic_trace(model.eval())
    ShapeProp(traced).propagate(*example_inputs)
    weights = WeightWriter(output_dir, stem)
    nodes: list[dict[str, Any]] = []
    env: dict[torch.fx.Node, str] = {}
    graph_inputs: list[str] = []
    graph_outputs: list[str] = []

    modules = dict(traced.named_modules())

    def add_param(name: str, tensor: torch.Tensor) -> str:
        return weights.add(name, tensor)

    for fx_node in traced.graph.nodes:
        if fx_node.op == "placeholder":
            env[fx_node] = fx_node.name
            graph_inputs.append(fx_node.name)
            continue
        if fx_node.op == "get_attr":
            attr = traced
            for part in str(fx_node.target).split("."):
                attr = getattr(attr, part)
            env[fx_node] = add_param(str(fx_node.target), attr)
            continue
        if fx_node.op == "call_module":
            module = modules[str(fx_node.target)]
            x = env[fx_node.args[0]]
            out = fx_node.name
            if isinstance(module, torch.nn.Linear):
                w = add_param(f"{fx_node.target}.weight", module.weight)
                inputs = [x, w]
                if module.bias is not None:
                    inputs.append(add_param(f"{fx_node.target}.bias", module.bias))
                _node(nodes, fx_node.name, "linear", inputs, [out])
            elif isinstance(module, torch.nn.BatchNorm2d):
                if module.training or not module.track_running_stats:
                    raise ExportError("BatchNorm2d export requires eval mode and tracked running statistics")
                if module.running_mean is None or module.running_var is None:
                    raise ExportError("BatchNorm2d export requires running_mean and running_var")
                weight = module.weight if module.weight is not None else torch.ones_like(module.running_mean)
                bias = module.bias if module.bias is not None else torch.zeros_like(module.running_mean)
                inputs = [
                    x,
                    add_param(f"{fx_node.target}.weight", weight),
                    add_param(f"{fx_node.target}.bias", bias),
                    add_param(f"{fx_node.target}.running_mean", module.running_mean),
                    add_param(f"{fx_node.target}.running_var", module.running_var),
                ]
                _node(nodes, fx_node.name, "batch_norm2d", inputs, [out], {"eps": float(module.eps)})
            elif isinstance(module, torch.nn.Conv2d):
                w = add_param(f"{fx_node.target}.weight", module.weight)
                inputs = [x, w]
                if module.bias is not None:
                    inputs.append(add_param(f"{fx_node.target}.bias", module.bias))
                _node(nodes, fx_node.name, "conv2d", inputs, [out],
                      {"stride": list(module.stride), "padding": list(module.padding)})
            elif isinstance(module, torch.nn.ReLU):
                _node(nodes, fx_node.name, "relu", [x], [out])
            elif isinstance(module, torch.nn.MaxPool2d):
                kernel = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
                padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                _node(nodes, fx_node.name, "max_pool2d", [x], [out],
                      {"kernel_size": list(kernel), "stride": list(stride), "padding": list(padding)})
            elif isinstance(module, torch.nn.Flatten):
                _node(nodes, fx_node.name, "reshape", [x], [out], {"shape": _shape_from_meta(fx_node)})
            else:
                raise ExportError(f"unsupported module: {fx_node.target} ({type(module).__name__})")
            env[fx_node] = out
            continue
        if fx_node.op in ("call_method", "call_function"):
            target = fx_node.target
            out = fx_node.name
            if target in ("reshape", "view") or target in (torch.reshape,):
                x = env[fx_node.args[0]]
                _node(nodes, fx_node.name, "reshape", [x], [out], {"shape": _shape_from_meta(fx_node)})
            elif target in (torch.flatten,):
                x = env[fx_node.args[0]]
                _node(nodes, fx_node.name, "reshape", [x], [out], {"shape": _shape_from_meta(fx_node)})
            else:
                raise ExportError(f"unsupported FX call: {target}")
            env[fx_node] = out
            continue
        if fx_node.op == "output":
            result = fx_node.args[0]
            if isinstance(result, tuple):
                graph_outputs = [env[item] for item in result]
            else:
                graph_outputs = [env[result]]
            continue
        raise ExportError(f"unsupported FX node: {fx_node.op}")

    weights_path = weights.write()
    graph = {
        "format": "dli.graph.v1",
        "model_type": model_type,
        "weights": weights_path.name,
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "nodes": nodes,
    }
    return _write_graph(output_dir, stem, graph), weights_path


def _is_qwen2_causal_lm(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    return getattr(config, "model_type", None) == "qwen2" and hasattr(model, "model") and hasattr(model.model, "layers")


def _linear_inputs(weights: WeightWriter, prefix: str, module: torch.nn.Module, source: str) -> list[str]:
    inputs = [source, weights.add(f"{prefix}.weight", module.weight)]
    bias = getattr(module, "bias", None)
    if bias is not None:
        inputs.append(weights.add(f"{prefix}.bias", bias))
    return inputs


def _qwen2_rotary_tables(config: Any) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    head_dim = hidden // heads
    if hidden % heads != 0:
        raise ExportError("Qwen2 hidden_size must be divisible by num_attention_heads")
    if head_dim % 2 != 0:
        raise ExportError("Qwen2 rotary export requires an even attention head dimension")
    rope_scaling = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type", "default") == "default":
        rope_scaling = None
    if rope_scaling not in (None, {}):
        raise ExportError("Qwen2 export currently supports default RoPE only")
    max_pos = int(getattr(config, "max_position_embeddings", 2048))
    theta = float(getattr(config, "rope_theta", 10000.0))
    pairs = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs).reshape(max_pos, pairs), torch.sin(freqs).reshape(max_pos, pairs)


def _export_qwen2_causal_lm(model: torch.nn.Module, output_dir: Path,
                            model_type: str, stem: str) -> tuple[Path, Path]:
    config = model.config
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    if kv_heads != heads:
        raise ExportError("Qwen2 export currently requires num_key_value_heads == num_attention_heads")
    if hidden % heads != 0:
        raise ExportError("Qwen2 hidden_size must be divisible by num_attention_heads")
    head_dim = hidden // heads

    weights = WeightWriter(output_dir, stem)
    nodes: list[dict[str, Any]] = []
    weights.add("model.embed_tokens.weight", model.model.embed_tokens.weight)
    cos, sin = _qwen2_rotary_tables(config)
    weights.add("rotary_cos", cos)
    weights.add("rotary_sin", sin)

    hidden_name = "hidden_states"
    _node(nodes, "model.embed_tokens", "embedding", ["input_ids", "model.embed_tokens.weight"], [hidden_name])

    for layer_index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_index}"
        residual = hidden_name
        norm = f"layers_{layer_index}_input_norm"
        weights.add(f"{prefix}.input_layernorm.weight", layer.input_layernorm.weight)
        _node(nodes, f"{prefix}.input_layernorm", "rms_norm",
              [hidden_name, f"{prefix}.input_layernorm.weight"], [norm],
              {"eps": float(layer.input_layernorm.variance_epsilon)})

        q = f"layers_{layer_index}_q"
        k = f"layers_{layer_index}_k"
        v = f"layers_{layer_index}_v"
        _node(nodes, f"{prefix}.self_attn.q_proj", "linear",
              _linear_inputs(weights, f"{prefix}.self_attn.q_proj", layer.self_attn.q_proj, norm), [q])
        _node(nodes, f"{prefix}.self_attn.k_proj", "linear",
              _linear_inputs(weights, f"{prefix}.self_attn.k_proj", layer.self_attn.k_proj, norm), [k])
        _node(nodes, f"{prefix}.self_attn.v_proj", "linear",
              _linear_inputs(weights, f"{prefix}.self_attn.v_proj", layer.self_attn.v_proj, norm), [v])

        q_heads = f"layers_{layer_index}_q_heads"
        k_heads = f"layers_{layer_index}_k_heads"
        v_heads = f"layers_{layer_index}_v_heads"
        _node(nodes, f"{prefix}.self_attn.q_reshape", "reshape", [q], [q_heads], {"shape": [1, heads, 1, head_dim]})
        _node(nodes, f"{prefix}.self_attn.k_reshape", "reshape", [k], [k_heads], {"shape": [1, heads, 1, head_dim]})
        _node(nodes, f"{prefix}.self_attn.v_reshape", "reshape", [v], [v_heads], {"shape": [1, heads, 1, head_dim]})

        q_rotary = f"layers_{layer_index}_q_rotary"
        k_rotary = f"layers_{layer_index}_k_rotary"
        _node(nodes, f"{prefix}.self_attn.rotary_embedding", "rotary_embedding",
              [q_heads, k_heads, "rotary_cos", "rotary_sin"], [q_rotary, k_rotary])

        attn_out = f"layers_{layer_index}_attention"
        _node(nodes, f"{prefix}.self_attn.attention", "attention",
              [q_rotary, k_rotary, v_heads], [attn_out],
              {"causal": True, "kv_cache": f"layer_{layer_index}"})
        merged = f"layers_{layer_index}_attention_merged"
        _node(nodes, f"{prefix}.self_attn.merge", "reshape", [attn_out], [merged], {"shape": [1, hidden]})
        proj = f"layers_{layer_index}_attention_proj"
        _node(nodes, f"{prefix}.self_attn.o_proj", "linear",
              _linear_inputs(weights, f"{prefix}.self_attn.o_proj", layer.self_attn.o_proj, merged), [proj])
        after_attn = f"layers_{layer_index}_after_attention"
        _node(nodes, f"{prefix}.attention_residual", "add", [residual, proj], [after_attn])

        post_norm = f"layers_{layer_index}_post_attention_norm"
        weights.add(f"{prefix}.post_attention_layernorm.weight", layer.post_attention_layernorm.weight)
        _node(nodes, f"{prefix}.post_attention_layernorm", "rms_norm",
              [after_attn, f"{prefix}.post_attention_layernorm.weight"], [post_norm],
              {"eps": float(layer.post_attention_layernorm.variance_epsilon)})
        gate = f"layers_{layer_index}_gate"
        up = f"layers_{layer_index}_up"
        silu = f"layers_{layer_index}_silu"
        gated = f"layers_{layer_index}_gated"
        down = f"layers_{layer_index}_mlp_down"
        _node(nodes, f"{prefix}.mlp.gate_proj", "linear",
              _linear_inputs(weights, f"{prefix}.mlp.gate_proj", layer.mlp.gate_proj, post_norm), [gate])
        _node(nodes, f"{prefix}.mlp.up_proj", "linear",
              _linear_inputs(weights, f"{prefix}.mlp.up_proj", layer.mlp.up_proj, post_norm), [up])
        _node(nodes, f"{prefix}.mlp.act_fn", "silu", [gate], [silu])
        _node(nodes, f"{prefix}.mlp.mul", "mul", [silu, up], [gated])
        _node(nodes, f"{prefix}.mlp.down_proj", "linear",
              _linear_inputs(weights, f"{prefix}.mlp.down_proj", layer.mlp.down_proj, gated), [down])
        hidden_name = f"layers_{layer_index}_output"
        _node(nodes, f"{prefix}.mlp_residual", "add", [after_attn, down], [hidden_name])

    weights.add("model.norm.weight", model.model.norm.weight)
    norm_out = "norm_hidden"
    _node(nodes, "model.norm", "rms_norm", [hidden_name, "model.norm.weight"], [norm_out],
          {"eps": float(model.model.norm.variance_epsilon)})
    _node(nodes, "lm_head", "linear",
          _linear_inputs(weights, "lm_head", model.lm_head, norm_out), ["logits"])

    weights_path = weights.write()
    graph = {
        "format": "dli.graph.v1",
        "model_type": model_type,
        "weights": weights_path.name,
        "inputs": ["input_ids"],
        "outputs": ["logits"],
        "nodes": nodes,
    }
    return _write_graph(output_dir, stem, graph), weights_path


def export_module(model: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...],
                  output_dir: str | Path, model_type: str = "pytorch_fx",
                  stem: str = "model") -> tuple[Path, Path]:
    output = Path(output_dir)
    if _is_qwen2_causal_lm(model):
        return _export_qwen2_causal_lm(model, output, "qwen2" if model_type == "pytorch_fx" else model_type, stem)
    return _export_fx(model, example_inputs, output, model_type, stem)


def _load_factory(spec: str) -> Callable[..., torch.nn.Module]:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ExportError("--model must be module:function")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    if not callable(factory):
        raise ExportError(f"model factory is not callable: {spec}")
    return factory


def _parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item)


def _parse_model_kwargs(items: list[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ExportError(f"--model-kwarg must be key=value: {item}")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    return kwargs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Python model factory as module:function")
    parser.add_argument("--example-shape", action="append", default=[])
    parser.add_argument("--model-kwarg", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", default="pytorch_fx")
    parser.add_argument("--stem", default="model")
    args = parser.parse_args()

    factory = _load_factory(args.model)
    model = factory(**_parse_model_kwargs(args.model_kwarg))
    example_inputs = tuple(torch.zeros(_parse_shape(shape), dtype=torch.float32) for shape in args.example_shape)
    graph, weights = export_module(model, example_inputs, args.output_dir, args.model_type, args.stem)
    print(graph)
    print(weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
