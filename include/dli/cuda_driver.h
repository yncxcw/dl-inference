#pragma once

#include <cstddef>
#include <string>

namespace dli {

bool cudaDriverAvailable();
std::string cudaDriverError();

class CudaAotKernel {
 public:
  CudaAotKernel(const char* kernel_name, const unsigned char* cubin, std::size_t cubin_size,
                unsigned shared_memory_bytes);
  ~CudaAotKernel();

  CudaAotKernel(const CudaAotKernel&) = delete;
  CudaAotKernel& operator=(const CudaAotKernel&) = delete;

  void launch(void** args, unsigned grid_x, unsigned grid_y, unsigned grid_z,
              unsigned threads_per_block, void* stream = nullptr);

 private:
  void load();

  const char* kernel_name_;
  const unsigned char* cubin_;
  std::size_t cubin_size_;
  unsigned shared_memory_bytes_;
  void* module_ = nullptr;
  void* function_ = nullptr;
};

}  // namespace dli
