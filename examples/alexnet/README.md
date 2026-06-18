# AlexNet Example

Exports a small AlexNet-style PyTorch model through the generic FX exporter and
runs it through the C++ engine.

```bash
cmake -S . -B build -DDLI_ENABLE_TRITON_AOT=ON
cmake --build build -j
./build/dli_alexnet
```
