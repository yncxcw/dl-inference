#pragma once

#include <map>
#include <string>

#include "dli/kv_cache.h"
#include "dli/tensor.h"

namespace dli {

// Mutable state belongs to one autoregressive request, not to a graph or its
// weights. KV entries cover ordinary attention; named tensors allow hybrid
// models to retain convolution and recurrent state as well.
class ExecutionState {
 public:
  ExecutionState() = default;
  ExecutionState(const ExecutionState&) = default;
  ExecutionState& operator=(const ExecutionState&) = default;
  ExecutionState(ExecutionState&&) noexcept = default;
  ExecutionState& operator=(ExecutionState&&) noexcept = default;

  KVCache& kvCache() { return kv_cache_; }
  const KVCache& kvCache() const { return kv_cache_; }

  // Stored tensors are immutable through this interface. Stateful operators
  // publish replacements with setTensor(), which keeps per-run copies isolated.
  const Tensor* findTensor(const std::string& name) const;
  void setTensor(std::string name, Tensor tensor);
  ExecutionState deepClone() const;

  std::size_t tensorCount() const { return tensors_.size(); }
  void reset();

 private:
  KVCache kv_cache_;
  std::map<std::string, Tensor> tensors_;
};

}  // namespace dli
