# dl-inference

`dl-inference` is a C++ CUDA inference runtime. Operators are authored as
Python/Triton source at build time, ahead-of-time compiled to cubins, embedded
into a native `.so`, and loaded by the runtime with `dlopen`.

Runtime inference does not start Python, import Python modules, link
`libpython`, or use CPU fallback kernels.

## What Is Included

- C++ tensor, graph, engine, plugin registry, CUDA allocation, CUDA Driver API
  launcher, weight loading, and KV cache support.
- A generic Triton AOT operator plugin for `dli.graph.v1` graphs.
- One folder per AOT operator under `python/dli_ops/aot/<operator>/`, each with
  `kernel.py` and `template.cc`.
- Runnable examples: `dli_operator_linear`, `dli_alexnet`, and `dli_qwen2`.
- Optional gRPC client/server build hooks.

## Build

```bash
cmake -S . -B build \
  -DDLI_BUILD_TESTS=ON \
  -DDLI_BUILD_EXAMPLES=ON \
  -DDLI_ENABLE_TRITON_AOT=ON \
  -DDLI_ENABLE_GRPC=OFF

cmake --build build -j
```

The default AOT target is `sm80`. Override it with:

```bash
DLI_AOT_ARCH=90 cmake --build build -j
```

## Run

```bash
./build/dli_operator_linear
./build/dli_alexnet
./build/dli_qwen2
```

Each executable loads:

```text
build/operators/libdli_triton_aot_ops.so
```

and needs a visible NVIDIA GPU compatible with the AOT architecture.

## Model Format

The runtime executes `dli.graph.v1`, a compact JSON graph IR plus optional
weight files:

```text
model.dli.json
model.dli.weights.json
model.dli.weights.bin
```

The engine is model-agnostic. Exporters produce graph and weight files; the
engine resolves tensor names, creates operators by string type, and launches GPU
kernels through the loaded plugin.

The generic PyTorch exporter lives in `python/dli_export/`. It supports a small
FX path for common CNN/MLP layers and an architecture lowering path for Qwen2.

Example:

```bash
PYTHONPATH=python python3 -m dli_export.export \
  --model my_model:create_model \
  --example-shape 1,3,32,32 \
  --output-dir exported/model \
  --stem model
```

Qwen2 tiny offline export:

```bash
PYTHONPATH=python:examples/qwen2 python3 -m dli_export.export \
  --model qwen2:create_model \
  --output-dir /tmp/dli-qwen2 \
  --model-type qwen2 \
  --stem qwen2
```

Local Qwen2 checkpoint export:

```bash
PYTHONPATH=python:examples/qwen2 python3 -m dli_export.export \
  --model qwen2:create_model \
  --model-kwarg model_id=/path/to/qwen2 \
  --model-kwarg local_files_only=true \
  --output-dir /tmp/dli-qwen2 \
  --model-type qwen2 \
  --stem qwen2
```

Current Qwen2 lowering supports default RoPE and requires
`num_key_value_heads == num_attention_heads`.

## AOT Technical Notes

Each AOT operator owns its folder:

```text
python/dli_ops/aot/linear/kernel.py
python/dli_ops/aot/linear/template.cc
```

`kernel.py` contains the Triton `@triton.jit` kernel. `template.cc` contains the
C++ runtime operator class and a `register_<operator>()` function. The generator
is:

```text
tools/build_triton_aot_ops.py
```

`default_specs()` defines the AOT specialization table. Each `AotSpec` says
which Python module/function to compile and which constexpr signature to use.
At build time the generator imports the Python module, calls Triton's compiler
API with a CUDA target such as `sm80`, and receives a cubin blob.

That cubin is emitted into generated C++ as a static byte array:

```cpp
static const unsigned char linear_k4_b1_ab12cd34_cubin[] = {
  0x7f, 0x45, 0x4c, 0x46, /* ... cubin bytes ... */
};

static dli::CudaAotKernel& linear_k4_b1_ab12cd34_kernel() {
  static dli::CudaAotKernel kernel(
      "linear_kernel",
      linear_k4_b1_ab12cd34_cubin,
      sizeof(linear_k4_b1_ab12cd34_cubin),
      0);
  return kernel;
}
```

`CudaAotKernel` lazily calls `cuModuleLoadData` on that byte array and
`cuModuleGetFunction("linear_kernel")` before launch.

The generated plugin exports:

```cpp
extern "C" bool dli_register_operators(dli::OperatorRegistry* registry);
```

Each operator template registers a factory:

```cpp
void register_linear(dli::OperatorRegistry* registry) {
  registry->registerFactory("linear", [] {
    return std::make_unique<LinearOp>();
  });
}
```

`OperatorRegistry::loadLibrary(path)` calls `dlopen`, finds
`dli_register_operators` with `dlsym`, and invokes it. After that, graph nodes
wire to operators by string:

```json
{"name": "fc", "op": "linear", "inputs": ["x", "w", "b"], "outputs": ["y"]}
```

The C++ operator validates tensors, allocates output CUDA tensors, selects the
matching AOT specialization, prepares raw pointer arguments, and launches the
embedded cubin.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

The numeric operator test builds one-node graphs for each AOT operator, runs
them through the C++ engine and plugin, then compares copied-back outputs
against PyTorch. It skips cleanly when `torch.cuda.is_available()` is false.
