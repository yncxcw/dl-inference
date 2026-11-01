/**
 * @file kv_cache.h
 * @brief Per-layer key/value cache for incremental attention on CUDA.
 */

#ifndef DLI_KV_CACHE_H_
#define DLI_KV_CACHE_H_

#include <cstddef>
#include <map>
#include <string>

#include "dli/tensor.h"

namespace dli {

/// Accumulated keys and values of one attention layer.
struct KVCacheEntry {
  /// Keys with shape [batch, heads, sequence, head_dim].
  Tensor key;
  /// Values with the same shape as @ref key.
  Tensor value;
};

/// Named cache of KVCacheEntry objects that grows along the sequence dimension.
class KVCache {
 public:
  /**
   * @brief Appends new keys and values to the named entry, creating it if needed.
   *
   * Appending allocates a new, longer tensor and copies old and new data into it.
   *
   * @param name Cache entry name (typically one per layer).
   * @param key New keys, rank-4 float32 CUDA tensor.
   * @param value New values, same shape and type as @p key.
   * @throws std::invalid_argument If tensors are not CUDA float32 rank-4 tensors, key and value
   *     shapes differ, or batch, head or head_dim differ from the existing entry.
   * @throws std::runtime_error If a CUDA copy fails.
   */
  void append(const std::string& name, const Tensor& key, const Tensor& value);

  /**
   * @brief Looks up an entry.
   * @param name Cache entry name.
   * @return Pointer to the entry, or nullptr if absent; invalidated by append() or clear().
   */
  const KVCacheEntry* get(const std::string& name) const;

  /// @return Number of named entries.
  std::size_t size() const { return entries_.size(); }
  /// Removes all entries.
  void clear() { entries_.clear(); }

 private:
  /// Entries keyed by name.
  std::map<std::string, KVCacheEntry> entries_;
};

}  // namespace dli

#endif  // DLI_KV_CACHE_H_
