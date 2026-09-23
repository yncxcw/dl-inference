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
  result.kv_cache_ = kv_cache_.deepClone();
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
