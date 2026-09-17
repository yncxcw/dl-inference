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

void ExecutionState::reset() {
  kv_cache_.clear();
  tensors_.clear();
}

}  // namespace dli
