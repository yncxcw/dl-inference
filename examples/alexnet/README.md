# AlexNet Example

Exports a small AlexNet-style PyTorch model through the generic FX exporter at
runtime, then runs the graph through the Python binding for the C++ engine.

```bash
cmake -S . -B build -DDLI_ENABLE_TRITON_AOT=ON
cmake --build build -j
python3 examples/alexnet/main.py
```

Generate only the graph and weights:

```bash
python3 examples/alexnet/main.py --export-only
```
