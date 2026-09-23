#include "dli/engine.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "dli/aten_operator.h"
#include "dli/logging.h"
#include "dli/utils.h"

namespace dli {

Engine::Engine() { registerAtenOperator(registry_); }

TensorMap Engine::run(const Graph& graph, TensorMap tensors) {
  return run(graph, std::move(tensors), {});
}

TensorMap Engine::run(const Graph& graph, TensorMap tensors, const RunOptions& options) {
  if (options.position_offset < 0)
    throw std::invalid_argument("position_offset must be non-negative");
  auto* state = options.state == nullptr ? &default_state_ : options.state;
  // Mutate a private view and publish it only after the whole graph (including
  // output validation) succeeds. Functional ATen schemas cannot mutate state
  // storage, so their common path shares immutable tensor handles. Custom
  // operators receive a deep snapshot because their implementations cannot be
  // mechanically checked for in-place mutations before they throw.
  const bool has_custom_operator =
      std::any_of(graph.nodes.begin(), graph.nodes.end(),
                  [](const Node& node) { return node.op_type != "aten"; });
  ExecutionState pending_state = has_custom_operator ? state->deepClone() : ExecutionState(*state);
  for (const auto& [input_name, state_key] : graph.state_inputs) {
    const Tensor* state_tensor = pending_state.findTensor(state_key);
    if (state_tensor == nullptr) {
      const auto initializer = graph.state_initializers.find(state_key);
      if (initializer == graph.state_initializers.end()) {
        throw std::invalid_argument("missing state tensor and initializer: " + state_key);
      }
      const auto tensor = tensors.find(initializer->second);
      if (tensor == tensors.end()) {
        throw std::invalid_argument("missing state initializer tensor: " + initializer->second);
      }
      pending_state.setTensor(state_key,
                              has_custom_operator ? tensor->second.clone() : tensor->second);
      state_tensor = pending_state.findTensor(state_key);
    }
    tensors[input_name] = *state_tensor;
  }
  ExecutionContext context{&pending_state.kvCache(), &pending_state, options.position_offset};
  for (const auto& node : graph.nodes) {
    std::vector<const Tensor*> inputs;
    inputs.reserve(node.inputs.size());
    for (const auto& name : node.inputs) {
      const auto it = tensors.find(name);
      if (it == tensors.end())
        throw std::invalid_argument("node '" + node.name + "' missing input tensor: " + name);
      inputs.push_back(&it->second);
      LOG_INFO << "node '" << node.name << "' input tensor: " << name
               << " shape: " << formatShape(it->second.shape());
    }

    std::vector<Tensor> output_storage(node.outputs.size());
    std::vector<Tensor*> outputs;
    outputs.reserve(output_storage.size());
    for (auto& output : output_storage) outputs.push_back(&output);

    auto op = registry_.create(node.op_type);
    LOG_INFO << "node '" << node.name << "' op: " << op->type();
    op->compute(inputs, outputs, node.attributes, context);

    for (std::size_t i = 0; i < node.outputs.size(); ++i) {
      tensors[node.outputs[i]] = std::move(output_storage[i]);
      LOG_INFO << "node '" << node.name << "' output tensor: " << node.outputs[i]
               << " shape: " << formatShape(tensors[node.outputs[i]].shape());
    }
  }

  for (const auto& [output_name, state_key] : graph.state_outputs) {
    const auto output = tensors.find(output_name);
    if (output == tensors.end()) {
      throw std::invalid_argument("state output tensor was not produced: " + output_name);
    }
    pending_state.setTensor(state_key, output->second);
  }

  if (graph.outputs.empty()) {
    *state = std::move(pending_state);
    return tensors;
  }
  TensorMap result;
  for (const auto& name : graph.outputs) {
    const auto it = tensors.find(name);
    if (it == tensors.end())
      throw std::invalid_argument("graph output tensor was not produced: " + name);
    result.emplace(name, it->second);
  }
  *state = std::move(pending_state);
  return result;
}

}  // namespace dli
