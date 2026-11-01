/**
 * @file cuda_runtime.h
 * @brief Minimal dlopen-based wrapper over the CUDA runtime (malloc, free, memcpy).
 */

#ifndef DLI_CUDA_RUNTIME_H_
#define DLI_CUDA_RUNTIME_H_

#include <cstddef>
#include <string>

namespace dli {

/// Direction of a cudaMemcpy; values match the CUDA runtime's cudaMemcpyKind.
enum class CudaMemcpyKind {
  HostToDevice = 1,    ///< Copy from host memory to device memory.
  DeviceToHost = 2,    ///< Copy from device memory to host memory.
  DeviceToDevice = 3,  ///< Copy between device allocations.
};

/**
 * @brief Reports whether libcudart was found and exposes the required symbols.
 * @return True if the CUDA runtime is usable.
 */
bool cudaRuntimeAvailable();

/// @return Human-readable reason the runtime is unavailable; empty if it loaded successfully.
std::string cudaRuntimeError();

/**
 * @brief Allocates device memory.
 * @param bytes Allocation size in bytes.
 * @return Device pointer; release it with cudaFreePointer().
 * @throws std::runtime_error If the runtime is unavailable or cudaMalloc fails.
 */
void* cudaMallocBytes(std::size_t bytes);

/**
 * @brief Frees device memory obtained from cudaMallocBytes().
 *
 * Null pointers and an unavailable runtime are ignored; errors from cudaFree are swallowed.
 *
 * @param pointer Device pointer to free, or nullptr.
 */
void cudaFreePointer(void* pointer) noexcept;

/**
 * @brief Copies bytes between host and/or device memory.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param bytes Number of bytes to copy.
 * @param kind Copy direction.
 * @throws std::runtime_error If the runtime is unavailable or cudaMemcpy fails.
 */
void cudaMemcpyBytes(void* dst, const void* src, std::size_t bytes, CudaMemcpyKind kind);

}  // namespace dli

#endif  // DLI_CUDA_RUNTIME_H_
