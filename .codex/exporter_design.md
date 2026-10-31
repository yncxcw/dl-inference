# Exporter Design

The runtime should stay graph-format agnostic and Python-free. Export is a
build-time/offline step that converts a source model plus weights into:

- `*.dli.json`: graph DAG
- `*.dli.weights.json`: tensor metadata
- `*.dli.weights.bin`: raw packed tensor payload

## Frontends

The preferred generic frontend is `torch.fx` for ordinary PyTorch modules. FX
gives a stable Python-level graph, and `ShapeProp` provides the output shapes
needed for reshape/flatten nodes.

For transformer families where FX captures too much Python control flow or
library internals, the exporter may add architecture lowerers. Qwen2 is handled
this way: the generic exporter detects `config.model_type == "qwen2"` and walks
the real Hugging Face module structure.

Autoregressive language models use `dli_export.causal_lm`, which needs no
per-model code. It builds a Transformers `StaticCache` for the model's config,
runs warm-up steps so every cache layer allocates its tensors, and makes each
tensor a layer holds (attention K/V, linear-attention conv and recurrent state,
write positions) a named request state input and output of a one-token
`torch.export` step. Functions wrapped by Transformers' Hub-kernel loader are
swapped for their PyTorch references while tracing. Export fails if a cache
layer keeps per-step state outside tensors, since tracing would freeze it.
Verified models: Qwen3.5-0.8B (hybrid), Qwen3-0.6B (dense), and small Llama,
Qwen3 and Qwen3.5 models in `tests/python/test_dli_export_causal_lm.py`.
Sliding-window cache layers are untested.

Model files in `examples/` should provide factories only. They should not carry
custom `export_model()` functions. Export policy stays centralized in
`python/dli_export/export.py`.

## Graph Wiring

Each graph node names an operator by string:

```json
{
  "name": "layers.0.self_attn.q_proj",
  "op": "linear",
  "inputs": ["hidden", "layers.0.self_attn.q_proj.weight"],
  "outputs": ["q"]
}
```

Weights are just tensors in the same namespace as activations. The engine does
not distinguish parameter tensors from input tensors once the `TensorMap` is
loaded.

## AOT Operator Registration

Each operator has:

```text
python/dli_ops/aot/<operator>/kernel.py
python/dli_ops/aot/<operator>/template.cc
```

The generator compiles Triton kernels into cubins, emits those cubins as C++
byte arrays, splices in every `template.cc`, and exports one native registration
function:

```cpp
extern "C" bool dli_register_operators(dli::OperatorRegistry* registry);
```

The runtime loads the plugin with `dlopen`, invokes that function, and can then
instantiate operators by the `op` string in the graph.
