# Qwen2 Example

The example model file provides a Hugging Face `Qwen2ForCausalLM` factory. The
runtime entry point exports the model, then launches the graph through the
Python binding for the C++ engine.

```bash
cmake -S . -B build -DDLI_ENABLE_TRITON_AOT=ON
cmake --build build -j
python3 examples/qwen2/main.py
```

Generate only the graph and weights:

```bash
python3 examples/qwen2/main.py --export-only
```

Export and run a local checkpoint:

```bash
python3 examples/qwen2/main.py --model-id /path/to/qwen2 --local-files-only
```
