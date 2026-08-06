#include "dli/cuda_runtime.h"

#include <dlfcn.h>

#include <sstream>
#include <stdexcept>

namespace dli {
namespace {

using CudaError = int;
using CudaMallocFn = CudaError (*)(void**, std::size_t);
using CudaFreeFn = CudaError (*)(void*);
using CudaMemcpyFn = CudaError (*)(void*, const void*, std::size_t, int);
using CudaGetErrorStringFn = const char* (*)(CudaError);

struct Runtime {
  void* handle = nullptr;
  CudaMallocFn malloc_fn = nullptr;
  CudaFreeFn free_fn = nullptr;
  CudaMemcpyFn memcpy_fn = nullptr;
  CudaGetErrorStringFn get_error_string_fn = nullptr;
  std::string load_error;

  Runtime() {
    const char* names[] = {"libcudart.so", "libcudart.so.12", "libcudart.so.11.0"};
    for (const char* name : names) {
      handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr) break;
      if (const char* error = dlerror()) load_error = error;
    }
    if (handle == nullptr) return;
    malloc_fn = reinterpret_cast<CudaMallocFn>(dlsym(handle, "cudaMalloc"));
    free_fn = reinterpret_cast<CudaFreeFn>(dlsym(handle, "cudaFree"));
    memcpy_fn = reinterpret_cast<CudaMemcpyFn>(dlsym(handle, "cudaMemcpy"));
    get_error_string_fn =
        reinterpret_cast<CudaGetErrorStringFn>(dlsym(handle, "cudaGetErrorString"));
    if (malloc_fn == nullptr || free_fn == nullptr || memcpy_fn == nullptr ||
        get_error_string_fn == nullptr) {
      load_error = "libcudart is missing required symbols";
      dlclose(handle);
      handle = nullptr;
    }
  }

  ~Runtime() {
    if (handle != nullptr) dlclose(handle);
  }

  std::string errorString(CudaError error) const {
    return get_error_string_fn == nullptr ? "CUDA error " + std::to_string(error)
                                          : get_error_string_fn(error);
  }
};

Runtime& runtime() {
  static auto* instance = new Runtime;
  return *instance;
}

void requireCuda(CudaError error, const char* operation) {
  if (error == 0) return;
  std::ostringstream message;
  message << operation << " failed: " << runtime().errorString(error);
  throw std::runtime_error(message.str());
}

}  // namespace

bool cudaRuntimeAvailable() { return runtime().handle != nullptr; }

std::string cudaRuntimeError() { return runtime().load_error; }

void* cudaMallocBytes(std::size_t bytes) {
  if (!cudaRuntimeAvailable())
    throw std::runtime_error("CUDA runtime is unavailable: " + cudaRuntimeError());
  void* pointer = nullptr;
  requireCuda(runtime().malloc_fn(&pointer, bytes), "cudaMalloc");
  return pointer;
}

void cudaFreePointer(void* pointer) noexcept {
  if (pointer == nullptr || !cudaRuntimeAvailable()) return;
  runtime().free_fn(pointer);
}

void cudaMemcpyBytes(void* dst, const void* src, std::size_t bytes, CudaMemcpyKind kind) {
  if (!cudaRuntimeAvailable())
    throw std::runtime_error("CUDA runtime is unavailable: " + cudaRuntimeError());
  requireCuda(runtime().memcpy_fn(dst, src, bytes, static_cast<int>(kind)), "cudaMemcpy");
}

}  // namespace dli
