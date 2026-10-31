#include "dli/kv_cache.h"

#include <stdexcept>

namespace dli {
namespace {

// Non-owning view over a contiguous float32 CUDA tensor, including external
// buffers that have no ATen storage. Valid only while `tensor` is alive.
at::Tensor cudaView(const Tensor& tensor) {
  return at::from_blob(const_cast<void*>(tensor.deviceData()), tensor.shape(),
                       at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, tensor.deviceId()));
}

}  // namespace

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
  // Attention kernels read [batch, heads, sequence, head_dim] contiguously, so
  // the cache is rebuilt rather than written into spare capacity. One
  // concatenation per tensor replaces 2 * batch * heads plane copies. The old
  // buffers are never modified, which lets ExecutionState snapshots share them.
  Tensor new_key(at::cat({cudaView(old_key), cudaView(key)}, 2));
  Tensor new_value(at::cat({cudaView(old_value), cudaView(value)}, 2));
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

KVCache KVCache::deepClone() const {
  KVCache result;
  for (const auto& [name, entry] : entries_) {
    result.entries_.emplace(name, KVCacheEntry{entry.key.clone(), entry.value.clone()});
  }
  return result;
}

}  // namespace dli
