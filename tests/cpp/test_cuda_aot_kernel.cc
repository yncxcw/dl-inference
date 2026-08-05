#include "test_support.h"

#include "dli/cuda_driver.h"

int main() {
  return dli_test::run("CudaAotKernel", [] {
    dli::CudaAotKernel kernel("empty", nullptr, 0, 0);
    kernel.launch(nullptr, 0, 1, 1, 1);
    const bool driver_available = dli::cudaDriverAvailable();
    if (!driver_available) dli_test::expect(!dli::cudaDriverError().empty(), "CUDA driver error text");
  });
}

