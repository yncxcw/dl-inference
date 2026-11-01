/**
 * @file engine.h
 * @brief Graph executor that owns the operator registry and KV cache.
 */

#ifndef DLI_ENGINE_H_
#define DLI_ENGINE_H_

#include <map>
#include <string>

#include "dli/graph.h"
#include "dli/operator.h"
#include "dli/tensor.h"

namespace dli {

/// Named tensors: graph inputs, intermediate values, weights or outputs.
using TensorMap = std::map<std::string, Tensor>;

/**
 * @brief Executes a Graph node by node using operators from its registry.
 *
 * The engine keeps its KV cache across run() calls, which enables incremental decoding.
 */
class Engine {
 public:
  /// Creates an engine whose registry contains only the built-in "aten" operator.
  Engine();

  /// @return Mutable operator registry, used to register factories or load plugins.
  OperatorRegistry& registry() { return registry_; }
  /// @return Read-only operator registry.
  const OperatorRegistry& registry() const { return registry_; }
  /// @return Mutable KV cache shared by all runs of this engine.
  KVCache& kvCache() { return kv_cache_; }
  /// @return Read-only KV cache.
  const KVCache& kvCache() const { return kv_cache_; }

  /**
   * @brief Runs every node of a graph in order.
   * @param graph Graph to execute; nodes must already be topologically sorted.
   * @param inputs Initial tensors (inputs and weights) keyed by name.
   * @return The tensors named in Graph::outputs, or every tensor if that list is empty.
   * @throws std::invalid_argument If a node input is missing, an operator is not registered, or a
   *     declared graph output was not produced.
   */
  TensorMap run(const Graph& graph, TensorMap inputs);

 private:
  /// Operators available to this engine.
  OperatorRegistry registry_;
  /// Key/value cache handed to operators through ExecutionContext.
  KVCache kv_cache_;
};

}  // namespace dli

#endif  // DLI_ENGINE_H_
