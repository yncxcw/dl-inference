# Qwen2 Example

The example model file only provides a Hugging Face `Qwen2ForCausalLM` factory.
The generic exporter detects Qwen2 module structure and lowers it to
`dli.graph.v1`.

Tiny offline export:

```bash
PYTHONPATH=python:examples/qwen2 python3 -m dli_export.export \
  --model qwen2:create_model \
  --output-dir /tmp/dli-qwen2 \
  --model-type qwen2 \
  --stem qwen2
```

Local checkpoint export:

```bash
PYTHONPATH=python:examples/qwen2 python3 -m dli_export.export \
  --model qwen2:create_model \
  --model-kwarg model_id=/path/to/qwen2 \
  --model-kwarg local_files_only=true \
  --output-dir /tmp/dli-qwen2 \
  --model-type qwen2 \
  --stem qwen2
```

Build and run:

```bash
cmake -S . -B build -DDLI_ENABLE_TRITON_AOT=ON
cmake --build build -j
./build/dli_qwen2
```
