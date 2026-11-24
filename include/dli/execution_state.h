/**
 * @file execution_state.h
 * @brief Per-request mutable state for autoregressive graph execution.
 */

#ifndef DLI_EXECUTION_STATE_H_
#define DLI_EXECUTION_STATE_H_

#include <cstddef>
#include <map>
#include <optional>
#include <string>

#include "dli/kv_cache.h"
#include "dli/tensor.h"

namespace dli {

/**
 * @brief Mutable state of one autoregressive request: its KV cache and named state tensors.
 *
 * State belongs to a request, not to a graph or its weights, so one engine can serve several
 * requests by giving each its own ExecutionState. KV entries cover ordinary attention; named
 * tensors let hybrid models keep convolution and recurrent state as well. Copies are shallow
 * and share tensor storage; deepClone() makes an isolated snapshot.
 */
class ExecutionState {
 public:
  /// Creates an empty state with no KV entries and no named tensors.
  ExecutionState() = default;
  /// Makes a shallow copy that shares KV entries and named-tensor storage with the source.
  ExecutionState(const ExecutionState&) = default;
  /// Replaces this state with a shallow copy of the source.
  ExecutionState& operator=(const ExecutionState&) = default;
  /// Takes over the source's entries; the source is left valid but unspecified.
  ExecutionState(ExecutionState&&) noexcept = default;
  /// Replaces this state with the source's entries; the source is left valid but unspecified.
  ExecutionState& operator=(ExecutionState&&) noexcept = default;

  /// Returns the request's KV cache for attention operators to extend.
  KVCache& kvCache() { return kv_cache_; }
  /// Returns the request's KV cache.
  const KVCache& kvCache() const { return kv_cache_; }

  /**
   * @brief Looks up a named state tensor.
   *
   * Like KVCache::get(), this returns a pointer: a missing name gives null instead of an
   * exception, and the lookup copies nothing. To keep the tensor, copy it; copies share storage.
   * Stored tensors are read-only through this interface. Stateful operators publish
   * replacements with setTensor(), which keeps per-run copies isolated.
   *
   * @param name State tensor name, as used by the graph's state inputs and outputs.
   * @return A copy of the stored tensor handle, or std::nullopt if @p name is not set. The copy
   *         shares storage with the stored tensor, so the lookup is cheap, and it stays valid
   *         after setTensor() or reset() replaces or drops the stored one.
   */
  std::optional<Tensor> findTensor(const std::string& name) const;

  /**
   * @brief Stores or replaces a named state tensor.
   * @param name State tensor name.
   * @param tensor New value. The state keeps this handle, so it shares storage with the
   *        caller's copy.
   */
  void setTensor(std::string name, Tensor tensor);

  /**
   * @brief Returns a snapshot that later runs cannot change through this state.
   *
   * Named tensors are copied in full with Tensor::clone(), because a custom operator may update
   * them in place. KV entries are shared: KVCache::append() never writes into an entry's tensors
   * but replaces them with new, longer ones, so the snapshot keeps the old ones.
   *
   * @return The isolated copy.
   * @throws std::logic_error If a named tensor has no storage.
   * @throws std::runtime_error If copying external CUDA memory fails.
   */
  ExecutionState deepClone() const;

  /// Returns the number of named state tensors; KV entries are not counted.
  std::size_t tensorCount() const { return tensors_.size(); }

  /// Clears all KV entries and named tensors.
  void reset();

 private:
  /// Attention keys and values, extended by attention operators on each run.
  KVCache kv_cache_;
  /// Named non-attention state, such as convolution or recurrent state, keyed by graph name.
  std::map<std::string, Tensor> tensors_;
};

}  // namespace dli

#endif  // DLI_EXECUTION_STATE_H_
