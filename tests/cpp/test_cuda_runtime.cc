/**
 * @file test_cuda_runtime.cc
 * @brief Tests CUDA runtime availability reporting without requiring a GPU.
 */

#include "dli/cuda_runtime.h"
#include "test_support.h"

// Verifies: an unavailable runtime reports an error string and freeing nullptr is safe.
int main() {
  return dli_test::run("cuda_runtime", [] {
    const bool runtime_available = dli::cudaRuntimeAvailable();
    if (!runtime_available)
      dli_test::expect(!dli::cudaRuntimeError().empty(), "CUDA runtime error text");
    dli::cudaFreePointer(nullptr);
  });
}
