#include "dli/cuda_runtime.h"
#include "test_support.h"

int main() {
  return dli_test::run("cuda_runtime", [] {
    const bool runtime_available = dli::cudaRuntimeAvailable();
    if (!runtime_available)
      dli_test::expect(!dli::cudaRuntimeError().empty(), "CUDA runtime error text");
    dli::cudaFreePointer(nullptr);
  });
}
