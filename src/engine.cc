#include "dli/engine.h"

#include <stdexcept>

#include "dli/aten_operator.h"
#include "dli/logging.h"
#include "dli/utils.h"

namespace dli {

Engine::Engine() {
  registerAtenOperator(registry_);
}

TensorMap Engine::run(const Graph& graph, TensorMap tensors) {
  ExecutionContext context{&kv_cache_};
  for (const auto& node : graph.nodes) {
    std::vector<const Tensor*> inputs;
    inputs.reserve(node.inputs.size());
    for (const auto& name : node.inputs) {
      const auto it = tensors.find(name);
      if (it == tensors.end()) throw std::invalid_argument("node '" + node.name + "' missing input tensor: " + name);
      inputs.push_back(&it->second);
      LOG_INFO << "node '" << node.name << "' input tensor: " << name << " shape: "
               << formatShape(it->second.shape());
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
      LOG_INFO << "node '" << node.name << "' output tensor: " << node.outputs[i] << " shape: "
               << formatShape(tensors[node.outputs[i]].shape());
    }
  }

  if (graph.outputs.empty()) return tensors;
  TensorMap result;
  for (const auto& name : graph.outputs) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) throw std::invalid_argument("graph output tensor was not produced: " + name);
    result.emplace(name, it->second);
  }
  return result;
}

}  // namespace dli
