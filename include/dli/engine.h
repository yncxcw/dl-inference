#pragma once

#include <map>
#include <string>

#include "dli/graph.h"
#include "dli/operator.h"
#include "dli/tensor.h"

namespace dli {

using TensorMap = std::map<std::string, Tensor>;

class Engine {
 public:
  Engine();

  OperatorRegistry& registry() { return registry_; }
  const OperatorRegistry& registry() const { return registry_; }
  KVCache& kvCache() { return kv_cache_; }
  const KVCache& kvCache() const { return kv_cache_; }

  TensorMap run(const Graph& graph, TensorMap inputs);

 private:
  OperatorRegistry registry_;
  KVCache kv_cache_;
};

}  // namespace dli
