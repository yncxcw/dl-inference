# Linear Operator Example

This example exports a one-layer PyTorch `nn.Linear` model at runtime, then
launches the exported graph through the Python binding for the C++ engine and
the AOT Triton operator plugin.

```bash
cmake -S . -B build -DDLI_ENABLE_TRITON_AOT=ON
cmake --build build -j
python3 examples/operators/linear/main.py
```

Generate only the graph and weights:

```bash
python3 examples/operators/linear/main.py --export-only
```
