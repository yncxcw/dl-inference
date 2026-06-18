# Linear Operator Example

This example runs a single `linear` node through the C++ engine and AOT Triton
operator plugin.

```bash
cmake -S . -B build -DDLI_ENABLE_TRITON_AOT=ON
cmake --build build -j
./build/dli_operator_linear
```
