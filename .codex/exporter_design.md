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
