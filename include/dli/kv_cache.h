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

/**
 * @brief Named cache of KVCacheEntry objects that grows along the sequence dimension.
 *
 * Published entries are never written in place: append() replaces an entry with newly allocated
 * tensors, so copies of a KVCache (and of an ExecutionState) may share entries. Operators must
 * treat tensors returned by get() as read-only.
 */
class KVCache {
 public:
  /**
   * @brief Appends new keys and values to the named entry, creating it if needed.
   *
   * Appending concatenates the old and new data along the sequence dimension into new tensors
   * and replaces the entry; the old tensors are never modified.
   *
   * @param name Cache entry name (typically one per layer).
   * @param key New keys, rank-4 float32 CUDA tensor.
   * @param value New values, same shape, type and device as @p key.
   * @throws std::invalid_argument If tensors are not CUDA float32 rank-4 tensors, key and value
   *     shapes or devices differ, or batch, head, head_dim or device differ from the existing
   *     entry.
   * @throws c10::Error If the concatenation fails on the device.
   */
  void append(const std::string& name, const Tensor& key, const Tensor& value);

  /**
   * @brief Looks up an entry.
   * @param name Cache entry name.
   * @return Pointer to the entry, or nullptr if absent; invalidated by append() or clear().
   */
  const KVCacheEntry* get(const std::string& name) const;

  /**
   * @brief Returns the cached sequence length of an entry.
   * @param name Cache entry name.
   * @return Size of the entry's sequence dimension, or 0 if the entry is absent.
   */
  std::size_t sequenceLength(const std::string& name) const;

  /// @return Number of named entries.
  std::size_t size() const { return entries_.size(); }

  /**
   * @brief Copies the cache together with every entry's key and value tensors.
   * @return A cache that shares no tensor storage with this one.
   */
  KVCache deepClone() const;

  /// Removes all entries.
  void clear() { entries_.clear(); }

 private:
  /// Entries keyed by name.
  std::map<std::string, KVCacheEntry> entries_;
};

}  // namespace dli

#endif  // DLI_KV_CACHE_H_
