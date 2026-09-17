#include "dli/kv_cache.h"

#include <stdexcept>

#include "dli/cuda_runtime.h"

namespace dli {

void KVCache::append(const std::string& name, const Tensor& key, const Tensor& value) {
  if (!key.isCuda() || !value.isCuda())
    throw std::invalid_argument("KV cache expects CUDA tensors");
  if (key.dtype() != DType::Float32 || value.dtype() != DType::Float32) {
    throw std::invalid_argument("KV cache expects float32 tensors");
  }
  if (key.rank() != 4 || value.rank() != 4)
    throw std::invalid_argument("KV cache expects rank-4 tensors");
  if (key.shape() != value.shape()) throw std::invalid_argument("KV key/value shape mismatch");
  if (key.deviceId() != value.deviceId())
    throw std::invalid_argument("KV key/value device mismatch");

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
  if (old_key.deviceId() != key.deviceId())
    throw std::invalid_argument("KV cache append device mismatch");
  auto shape = old_key.shape();
  shape[2] += key.dim(2);
  Tensor new_key = Tensor::cuda(DType::Float32, shape, key.deviceId());
  Tensor new_value = Tensor::cuda(DType::Float32, shape, value.deviceId());

  // [batch, heads, sequence, head_dim] is contiguous with sequence nested
  // inside each batch/head plane. Copy each plane independently so appending
  // sequence tokens remains correct when batch * heads is greater than one.
  const auto planes = static_cast<std::size_t>(old_key.dim(0) * old_key.dim(1));
  const auto element_bytes = byteSize(DType::Float32);
  const auto old_plane_bytes =
      static_cast<std::size_t>(old_key.dim(2) * old_key.dim(3)) * element_bytes;
  const auto appended_plane_bytes =
      static_cast<std::size_t>(key.dim(2) * key.dim(3)) * element_bytes;
  const auto combined_plane_bytes = old_plane_bytes + appended_plane_bytes;

  const auto append_planes = [&](Tensor& destination, const Tensor& existing,
                                 const Tensor& appended) {
    auto* destination_bytes = static_cast<char*>(destination.deviceData());
    const auto* existing_bytes = static_cast<const char*>(existing.deviceData());
    const auto* appended_bytes = static_cast<const char*>(appended.deviceData());
    for (std::size_t plane = 0; plane < planes; ++plane) {
      cudaMemcpyBytes(destination_bytes + plane * combined_plane_bytes,
                      existing_bytes + plane * old_plane_bytes, old_plane_bytes,
                      CudaMemcpyKind::DeviceToDevice);
      cudaMemcpyBytes(destination_bytes + plane * combined_plane_bytes + old_plane_bytes,
                      appended_bytes + plane * appended_plane_bytes, appended_plane_bytes,
                      CudaMemcpyKind::DeviceToDevice);
    }
  };
  append_planes(new_key, old_key, key);
  append_planes(new_value, old_value, value);
  it->second = KVCacheEntry{std::move(new_key), std::move(new_value)};
}

const KVCacheEntry* KVCache::get(const std::string& name) const {
  const auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : &it->second;
}

std::size_t KVCache::sequenceLength(const std::string& name) const {
  const auto* entry = get(name);
  return entry == nullptr ? 0 : static_cast<std::size_t>(entry->key.dim(2));
}

}  // namespace dli
