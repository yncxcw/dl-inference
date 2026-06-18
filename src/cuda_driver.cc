#include "dli/cuda_driver.h"

#include <dlfcn.h>

#include <stdexcept>

namespace dli {
namespace {

using CUresult = int;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
constexpr CUresult kCudaSuccess = 0;

using CuInitFn = CUresult (*)(unsigned int);
using CuModuleLoadDataFn = CUresult (*)(CUmodule*, const void*);
using CuModuleGetFunctionFn = CUresult (*)(CUfunction*, CUmodule, const char*);
using CuModuleUnloadFn = CUresult (*)(CUmodule);
using CuLaunchKernelFn = CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                                      unsigned, unsigned, unsigned, CUstream, void**, void**);
using CuGetErrorStringFn = CUresult (*)(CUresult, const char**);

struct Driver {
  void* handle = nullptr;
  CuInitFn init = nullptr;
  CuModuleLoadDataFn module_load_data = nullptr;
  CuModuleGetFunctionFn module_get_function = nullptr;
  CuModuleUnloadFn module_unload = nullptr;
  CuLaunchKernelFn launch_kernel = nullptr;
  CuGetErrorStringFn get_error_string = nullptr;
  std::string load_error;

  Driver() {
    const char* names[] = {"libcuda.so.1", "libcuda.so"};
    for (const char* name : names) {
      handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr) break;
      if (const char* error = dlerror()) load_error = error;
    }
    if (handle == nullptr) return;
    init = reinterpret_cast<CuInitFn>(dlsym(handle, "cuInit"));
    module_load_data = reinterpret_cast<CuModuleLoadDataFn>(dlsym(handle, "cuModuleLoadData"));
    module_get_function = reinterpret_cast<CuModuleGetFunctionFn>(dlsym(handle, "cuModuleGetFunction"));
    module_unload = reinterpret_cast<CuModuleUnloadFn>(dlsym(handle, "cuModuleUnload"));
    launch_kernel = reinterpret_cast<CuLaunchKernelFn>(dlsym(handle, "cuLaunchKernel"));
    get_error_string = reinterpret_cast<CuGetErrorStringFn>(dlsym(handle, "cuGetErrorString"));
    if (init == nullptr || module_load_data == nullptr || module_get_function == nullptr ||
        module_unload == nullptr || launch_kernel == nullptr || get_error_string == nullptr) {
      load_error = "libcuda is missing required driver API symbols";
      dlclose(handle);
      handle = nullptr;
      return;
    }
    const auto init_result = init(0);
    if (init_result != kCudaSuccess) {
      const char* text = nullptr;
      get_error_string(init_result, &text);
      load_error = std::string("cuInit failed: ") + (text == nullptr ? "unknown error" : text);
      dlclose(handle);
      handle = nullptr;
    }
  }

  ~Driver() {
    if (handle != nullptr) dlclose(handle);
  }
};

Driver& driver() {
  static Driver instance;
  return instance;
}

std::string errorString(CUresult result) {
  const char* text = nullptr;
  if (driver().get_error_string != nullptr) driver().get_error_string(result, &text);
  return text == nullptr ? "CUDA driver error " + std::to_string(result) : text;
}

void requireDriver() {
  if (!cudaDriverAvailable()) throw std::runtime_error("CUDA driver is unavailable: " + cudaDriverError());
}

void requireSuccess(CUresult result, const std::string& operation) {
  if (result != kCudaSuccess) throw std::runtime_error(operation + " failed: " + errorString(result));
}

}  // namespace

bool cudaDriverAvailable() {
  return driver().handle != nullptr;
}

std::string cudaDriverError() {
  return driver().load_error;
}

CudaAotKernel::CudaAotKernel(const char* kernel_name, const unsigned char* cubin,
                             std::size_t cubin_size, unsigned shared_memory_bytes)
    : kernel_name_(kernel_name),
      cubin_(cubin),
      cubin_size_(cubin_size),
      shared_memory_bytes_(shared_memory_bytes) {}

CudaAotKernel::~CudaAotKernel() {
  if (module_ != nullptr && cudaDriverAvailable()) {
    driver().module_unload(static_cast<CUmodule>(module_));
  }
}

void CudaAotKernel::load() {
  if (function_ != nullptr) return;
  requireDriver();
  if (cubin_ == nullptr || cubin_size_ == 0) {
    throw std::runtime_error(std::string("AOT kernel has no cubin: ") + kernel_name_);
  }
  auto& drv = driver();
  requireSuccess(drv.module_load_data(reinterpret_cast<CUmodule*>(&module_), cubin_),
                 std::string("cuModuleLoadData(") + kernel_name_ + ")");
  requireSuccess(drv.module_get_function(reinterpret_cast<CUfunction*>(&function_),
                                         static_cast<CUmodule>(module_), kernel_name_),
                 std::string("cuModuleGetFunction(") + kernel_name_ + ")");
}

void CudaAotKernel::launch(void** args, unsigned grid_x, unsigned grid_y, unsigned grid_z,
                           unsigned threads_per_block, void* stream) {
  if (grid_x == 0 || grid_y == 0 || grid_z == 0) return;
  load();
  requireSuccess(driver().launch_kernel(static_cast<CUfunction>(function_), grid_x, grid_y,
                                        grid_z, threads_per_block, 1, 1,
                                        shared_memory_bytes_, static_cast<CUstream>(stream),
                                        args, nullptr),
                 std::string("cuLaunchKernel(") + kernel_name_ + ")");
}

}  // namespace dli
