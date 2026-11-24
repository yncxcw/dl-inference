/**
 * @file engine.h
 * @brief Graph executor that owns the operator registry and a default request state.
 */

#ifndef DLI_ENGINE_H_
#define DLI_ENGINE_H_

#include <cstdint>
#include <map>
#include <string>

#include "dli/execution_state.h"
#include "dli/graph.h"
#include "dli/operator.h"
#include "dli/tensor.h"

namespace dli {

/// Named tensors: graph inputs, intermediate values, weights or outputs.
using TensorMap = std::map<std::string, Tensor>;

/// Per-call options for Engine::run().
struct RunOptions {
  /// Request state to read and update. Null selects the engine's default state, the one the
  /// two-argument run() uses, so a caller serving one request at a time need not create one.
  ExecutionState* state = nullptr;
  /// Absolute position of the run's first token in the request, passed to operators through
  /// ExecutionContext. Position-dependent operators need it: rotary embedding rotates each token
  /// by an angle that depends on its position, but a one-token decode graph sees only the new
  /// token. After a 10-token prompt runs at offset 0, the first decode step runs at offset 10,
  /// the next at 11, and so on. The caller tracks it, because the caller feeds the tokens.
  std::int64_t position_offset = 0;
};

/**
 * @brief Executes a Graph node by node using operators from its registry.
 *
 * Mutable state (the KV cache and named state tensors) lives in an ExecutionState. Callers pass
 * one per request through RunOptions. Runs without one use the engine's default state, which
 * persists across run() calls, so code that decodes one request through the two-argument run()
 * and kvCache() keeps working without creating a state.
 */
class Engine {
 public:
  /// Creates an engine whose registry contains only the built-in "aten" operator.
  Engine();

  /// @return Mutable operator registry, used to register factories or load plugins.
  OperatorRegistry& registry() { return registry_; }
  /// @return Read-only operator registry.
  const OperatorRegistry& registry() const { return registry_; }
  /// @return Mutable state used by runs that pass no ExecutionState.
  ExecutionState& defaultState() { return default_state_; }
  /// @return Read-only default state.
  const ExecutionState& defaultState() const { return default_state_; }
  /// @return Mutable KV cache of the default state.
  KVCache& kvCache() { return default_state_.kvCache(); }
  /// @return Read-only KV cache of the default state.
  const KVCache& kvCache() const { return default_state_.kvCache(); }
  /// Clears the default state's KV cache and named tensors.
  void reset() { default_state_.reset(); }

  /**
   * @brief Runs every node of a graph in order against the default state.
   * @param graph Graph to execute; nodes must already be topologically sorted.
   * @param inputs Initial tensors (inputs and weights) keyed by name.
   * @return The tensors named in Graph::outputs, or every tensor if that list is empty.
   * @throws std::invalid_argument If a node input is missing, an operator is not registered, or a
   *     declared graph output was not produced.
   */
  TensorMap run(const Graph& graph, TensorMap inputs);

  /**
   * @brief Runs a graph against a request state, publishing state changes only on success.
   *
   * The run works on a private copy of the state. Graph::state_inputs are fed from it, or from
   * Graph::state_initializers while the state has no value for a key, and Graph::state_outputs
   * are stored into it. The copy replaces the state only after every node and output check
   * succeeds, so a failed run leaves the state unchanged.
   *
   * @param graph Graph to execute; nodes must already be topologically sorted.
   * @param inputs Initial tensors (inputs and weights) keyed by name.
   * @param options Request state and position offset; a null state selects the default state.
   * @return The tensors named in Graph::outputs, or every tensor if that list is empty.
   * @throws std::invalid_argument If position_offset is negative, a state tensor and its
   *     initializer are both missing, a node input or declared output is missing, or an operator
   *     is not registered.
   */
  TensorMap run(const Graph& graph, TensorMap inputs, const RunOptions& options);

 private:
  /// Operators available to this engine.
  OperatorRegistry registry_;
  /// State used by runs that pass no ExecutionState.
  ExecutionState default_state_;
};

}  // namespace dli

#endif  // DLI_ENGINE_H_
