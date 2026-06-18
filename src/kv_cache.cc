#include "dli/kv_cache.h"

#include "dli/cuda_runtime.h"

#include <stdexcept>

namespace dli {

void KVCache::append(const std::string& name, const Tensor& key, const Tensor& value) {
  if (!key.isCuda() || !value.isCuda()) throw std::invalid_argument("KV cache expects CUDA tensors");
  if (key.dtype() != DType::Float32 || value.dtype() != DType::Float32) {
    throw std::invalid_argument("KV cache expects float32 tensors");
  }
  if (key.rank() != 4 || value.rank() != 4) throw std::invalid_argument("KV cache expects rank-4 tensors");
  if (key.shape() != value.shape()) throw std::invalid_argument("KV key/value shape mismatch");

  const auto it = entries_.find(name);
  if (it == entries_.end()) {
    entries_.emplace(name, KVCacheEntry{key, value});
    return;
  }

  const auto& old_key = it->second.key;
  const auto& old_value = it->second.value;
  if (old_key.dim(0) != key.dim(0) || old_key.dim(1) != key.dim(1) ||
      old_key.dim(3) != key.dim(3)) {
    throw std::invalid_argument("KV cache append shape mismatch");
  }
  auto shape = old_key.shape();
  shape[2] += key.dim(2);
  Tensor new_key = Tensor::cuda(DType::Float32, shape, key.deviceId());
  Tensor new_value = Tensor::cuda(DType::Float32, shape, value.deviceId());
  cudaMemcpyBytes(new_key.deviceData(), old_key.deviceData(), old_key.nbytes(), CudaMemcpyKind::DeviceToDevice);
  cudaMemcpyBytes(static_cast<char*>(new_key.deviceData()) + old_key.nbytes(), key.deviceData(), key.nbytes(),
                  CudaMemcpyKind::DeviceToDevice);
  cudaMemcpyBytes(new_value.deviceData(), old_value.deviceData(), old_value.nbytes(), CudaMemcpyKind::DeviceToDevice);
  cudaMemcpyBytes(static_cast<char*>(new_value.deviceData()) + old_value.nbytes(), value.deviceData(), value.nbytes(),
                  CudaMemcpyKind::DeviceToDevice);
  it->second = KVCacheEntry{std::move(new_key), std::move(new_value)};
}

const KVCacheEntry* KVCache::get(const std::string& name) const {
  const auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : &it->second;
}

}  // namespace dli
