/**
 * @file cuda_driver.h
 * @brief Lazy CUDA driver API loader and ahead-of-time (cubin) kernel launcher.
 */

#ifndef DLI_CUDA_DRIVER_H_
#define DLI_CUDA_DRIVER_H_

#include <cstddef>
#include <string>

namespace dli {

/**
 * @brief Reports whether libcuda was found, initialized and exposes the required symbols.
 *
 * The driver is loaded with dlopen on first use, so this never requires linking against CUDA.
 *
 * @return True if the CUDA driver API is usable.
 */
bool cudaDriverAvailable();

/// @return Human-readable reason the driver is unavailable; empty if it loaded successfully.
std::string cudaDriverError();

/**
 * @brief A single precompiled GPU kernel embedded as a cubin image.
 *
 * The module is loaded lazily on the first non-empty launch. The kernel name and cubin bytes are
 * borrowed and must outlive the object (they are normally static data). Instances are not
 * copyable.
 */
class CudaAotKernel {
 public:
  /**
   * @brief Describes a kernel without touching the GPU.
   * @param kernel_name Entry-point symbol inside the cubin; must outlive this object.
   * @param cubin Pointer to the cubin image; must outlive this object.
   * @param cubin_size Size of the cubin image in bytes.
   * @param shared_memory_bytes Dynamic shared memory per block, in bytes.
   */
  CudaAotKernel(const char* kernel_name, const unsigned char* cubin, std::size_t cubin_size,
                unsigned shared_memory_bytes);
  ~CudaAotKernel();

  CudaAotKernel(const CudaAotKernel&) = delete;
  CudaAotKernel& operator=(const CudaAotKernel&) = delete;

  /**
   * @brief Launches the kernel, loading the module on first use.
   *
   * Returns immediately without loading anything if any grid dimension is zero.
   *
   * @param args Array of pointers to the kernel arguments, as for cuLaunchKernel.
   * @param grid_x Grid size in X, in blocks.
   * @param grid_y Grid size in Y, in blocks.
   * @param grid_z Grid size in Z, in blocks.
   * @param threads_per_block Block size in X; Y and Z are 1.
   * @param stream CUDA stream handle, or nullptr for the default stream.
   * @throws std::runtime_error If the driver is unavailable, the cubin is empty, or any driver
   *     call fails.
   */
  void launch(void** args, unsigned grid_x, unsigned grid_y, unsigned grid_z,
              unsigned threads_per_block, void* stream = nullptr);

 private:
  /// Loads the module and resolves the function once; no-op if already loaded.
  void load();

  /// Entry-point symbol name (borrowed).
  const char* kernel_name_;
  /// Cubin image (borrowed).
  const unsigned char* cubin_;
  /// Cubin image size in bytes.
  std::size_t cubin_size_;
  /// Dynamic shared memory per block, in bytes.
  unsigned shared_memory_bytes_;
  /// CUmodule handle; null until load() succeeds.
  void* module_ = nullptr;
  /// CUfunction handle; null until load() succeeds.
  void* function_ = nullptr;
};

}  // namespace dli

#endif  // DLI_CUDA_DRIVER_H_
