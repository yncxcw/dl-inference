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
  // KV entries are immutable: append() publishes freshly allocated tensors, so
  // sharing them keeps the snapshot isolated without copying the whole
  // sequence on every decode step. Named tensors may be mutated by custom
  // operators and are still copied.
  result.kv_cache_ = kv_cache_;
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
