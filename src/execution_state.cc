/**
 * @file execution_state.cc
 * @brief Implements dli::ExecutionState.
 */

#include "dli/execution_state.h"

#include <utility>

namespace dli {

const Tensor* ExecutionState::findTensor(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

void ExecutionState::setTensor(std::string name, Tensor tensor) {
  tensors_[std::move(name)] = std::move(tensor);
}

ExecutionState ExecutionState::deepClone() const {
  ExecutionState result;
  // Share the KV entries. KVCache::append() never writes into an entry's key or value tensor: it
  // allocates longer ones, copies the old and new positions into them and swaps them in. If this
  // snapshot shares layer 0's keys [1, 2, 10, 64] and a later run appends one position, the live
  // cache moves to a new [1, 2, 11, 64] tensor and the snapshot keeps the old one. Sharing avoids
  // copying the whole sequence on every step.
  result.kv_cache_ = kv_cache_;
  // Copy the named tensors in full, because a custom operator may update them in place. Unlike KV
  // entries they do not grow with the sequence, and Engine::run() deep-clones the state only for
  // graphs that have custom operators.
  for (const auto& [name, tensor] : tensors_) {
    result.tensors_.emplace(name, tensor.clone());
  }
  return result;
}

void ExecutionState::reset() {
  kv_cache_.clear();
  tensors_.clear();
}

}  // namespace dli
