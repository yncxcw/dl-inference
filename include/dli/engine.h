#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "dli/execution_state.h"
#include "dli/graph.h"
#include "dli/operator.h"
#include "dli/tensor.h"

namespace dli {

using TensorMap = std::map<std::string, Tensor>;

struct RunOptions {
  ExecutionState* state = nullptr;
  std::int64_t position_offset = 0;
};

class Engine {
 public:
  Engine();

  OperatorRegistry& registry() { return registry_; }
  const OperatorRegistry& registry() const { return registry_; }
  ExecutionState& defaultState() { return default_state_; }
  const ExecutionState& defaultState() const { return default_state_; }
  KVCache& kvCache() { return default_state_.kvCache(); }
  const KVCache& kvCache() const { return default_state_.kvCache(); }
  void reset() { default_state_.reset(); }

  TensorMap run(const Graph& graph, TensorMap inputs);
  TensorMap run(const Graph& graph, TensorMap inputs, const RunOptions& options);

 private:
  OperatorRegistry registry_;
  ExecutionState default_state_;
};

}  // namespace dli
