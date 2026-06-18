#pragma once

#include <cstddef>
#include <string>

namespace dli {

enum class CudaMemcpyKind {
  HostToDevice = 1,
  DeviceToHost = 2,
  DeviceToDevice = 3,
};

bool cudaRuntimeAvailable();
std::string cudaRuntimeError();
void* cudaMallocBytes(std::size_t bytes);
void cudaFreePointer(void* pointer) noexcept;
void cudaMemcpyBytes(void* dst, const void* src, std::size_t bytes, CudaMemcpyKind kind);

}  // namespace dli
