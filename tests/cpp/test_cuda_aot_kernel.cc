/**
 * @file test_cuda_aot_kernel.cc
 * @brief Tests that CudaAotKernel handles empty launches and unavailable drivers gracefully.
 */

#include "dli/cuda_driver.h"
#include "test_support.h"

// Verifies: a zero-grid launch is a no-op and an unavailable driver reports an error string.
int main() {
  return dli_test::run("CudaAotKernel", [] {
    dli::CudaAotKernel kernel("empty", nullptr, 0, 0);
    kernel.launch(nullptr, 0, 1, 1, 1);
    const bool driver_available = dli::cudaDriverAvailable();
    if (!driver_available)
      dli_test::expect(!dli::cudaDriverError().empty(), "CUDA driver error text");
  });
}
