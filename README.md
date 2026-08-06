# dl-inference

`dl-inference` is a C++ CUDA inference runtime. Operators are authored as
Python/Triton source at build time, ahead-of-time compiled to cubins, embedded
into a native `.so`, and loaded by the runtime with `dlopen`.

The C++ inference engine and AOT plugin do not start Python, import Python
modules, link `libpython`, or use CPU fallback kernels. Optional Python
bindings are available for launching the same C++ engine from Python.

## What Is Included

- C++ tensor, graph, engine, plugin registry, CUDA allocation, CUDA Driver API
  launcher, weight loading, and KV cache support.
- `dli::Tensor` is backed by PyTorch/ATen tensors while preserving
  the runtime-facing `deviceData()` API used by AOT kernels.
- A generic Triton AOT operator plugin for `dli.graph.v1` graphs.
- One folder per AOT operator under `python/dli_ops/aot/<operator>/`, each with
  `kernel.py` and `template.cc`.
- Runnable Python examples for a linear layer, AlexNet-style CNN, and Qwen2.
- Optional gRPC client/server build hooks.

## Build

The default full build path is:

```bash
./build.sh
```

`build.sh` configures and builds the full local development configuration:

- `DLI_BUILD_TESTS=ON`
- `DLI_BUILD_EXAMPLES=ON`
- `DLI_ENABLE_TRITON_AOT=ON`
- `DLI_ENABLE_GRPC=OFF`
- `DLI_BUILD_PYTHON_BINDINGS=ON`

The equivalent manual commands are:

```bash
cmake -S . -B build \
  -DDLI_BUILD_TESTS=ON \
  -DDLI_BUILD_EXAMPLES=ON \
  -DDLI_ENABLE_TRITON_AOT=ON \
  -DDLI_ENABLE_GRPC=OFF

cmake --build build -j
```

The AOT target auto-detects the first visible CUDA device through PyTorch and
falls back to `sm80` when no GPU is visible at build time. Override it with:

```bash
DLI_AOT_ARCH=90 cmake --build build -j
```

The script accepts the same knobs through environment variables:

```bash
DLI_BUILD_DIR=build-sm90 DLI_AOT_ARCH=90 DLI_BUILD_JOBS=8 ./build.sh
DLI_ENABLE_GRPC=ON ./build.sh
```

`DLI_ENABLE_GRPC=ON` requires local Protobuf and gRPC C++ development packages.

The tensor backend is PyTorch/ATen. CMake locates the installed PyTorch C++
package through:

```bash
python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
```

## Continuous Integration

GitHub Actions runs the CPU-safe build for every pull request and every push to
`main`. The workflow uses Python 3.11, PyTorch 2.8, and Triton 3.4:

```text
.github/workflows/ci.yml
```

Standard `ubuntu-24.04` GitHub-hosted runners do not provide an NVIDIA GPU. CI
therefore builds with `DLI_ENABLE_TRITON_AOT=OFF` and does not execute CTest.
Test execution, AOT plugin compilation, operator numerics, and model end-to-end
tests will be enabled after a self-hosted or GitHub larger GPU runner is
available.

## Run Examples

```bash
python3 examples/operators/linear/main.py
python3 examples/alexnet/main.py
python3 examples/qwen2/main.py
```

Each example exports its model to `build/examples/<name>` at runtime, loads:

```text
build/operators/libdli_triton_aot_ops.so
```

through the Python engine binding, and needs a visible NVIDIA GPU compatible
with the AOT architecture. Use `--export-only` to generate the graph and weights
without launching GPU inference.

## Python Binding

The build also emits a native Python module under `build/python/dli`. Use it by
putting that directory on `PYTHONPATH`:

```bash
PYTHONPATH=build/python python3 - <<'PY'
import json
import torch
import dli

graph = dli.Graph.from_json(json.dumps({
    "format": "dli.graph.v1",
    "inputs": ["x"],
    "outputs": ["y"],
    "nodes": [{
        "name": "relu",
        "op": "aten",
        "inputs": ["x"],
        "outputs": ["y"],
        "attrs": {"name": "aten::relu"},
    }],
}))

engine = dli.Engine()
outputs = engine.run(graph, {"x": torch.tensor([-1.0, 2.0])})
print(outputs["y"])
PY
```

For Triton AOT graphs, load the generated operator plugin before `run`:

```python
engine = dli.Engine()
engine.load_library("build/operators/libdli_triton_aot_ops.so")
outputs = engine.run(graph, inputs)
```

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

Triton's compiled CUDA entry point includes a trailing global scratch pointer
parameter in the PTX signature. The framework rejects kernels that require
non-zero global scratch bytes today, but the launch ABI still needs the pointer
slot. Each AOT template appends a null argument:

```cpp
void* triton_scratch = nullptr;
void* args[] = {
    &x, &weight, &bias, &out, &m, &n, &triton_scratch,
};
```

Without this final `&triton_scratch` entry, `cuLaunchKernel` receives fewer
argument pointers than the cubin metadata declares and can segfault inside the
CUDA driver.

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

Run the full build plus test E2E path:

```bash
./test.sh
```

`test.sh` runs `./build.sh` first, then executes the full CTest suite from the
configured build directory. To test an already-built tree:

```bash
DLI_SKIP_BUILD=1 ./test.sh
```

List all registered tests:

```bash
ctest --test-dir build -N
```

Run all tests directly:

```bash
ctest --test-dir build --output-on-failure
```

Run a subset by CTest name:

```bash
./test.sh -R dli_test_engine
./test.sh -R 'dli_aot_.*_numerics'
./test.sh -R 'dli_e2e_(alexnet|qwen2)'
```

The default suite covers:

- C++ unit tests, one target per runtime class/module: `dli_test_*`.
- Python module tests: `dli_py_*`.
- Triton AOT module/spec tests: `dli_py_aot_*` and
  `dli_triton_aot_operator_specs`.
- Per-operator numeric tests: `dli_aot_<operator>_numerics`. Each one builds a
  one-node graph, runs it through the C++ engine and AOT plugin, then compares
  copied-back output against PyTorch.
- Runtime linkage guard: `dli_no_python_runtime_link`.
- Model E2E tests: `dli_e2e_alexnet` and `dli_e2e_qwen2`.

GPU-backed tests skip cleanly when `torch.cuda.is_available()` is false. The
gRPC tests are registered only when configured with `DLI_ENABLE_GRPC=ON`.
