#include <ATen/ATen.h>

#include <memory>
#include <string>
#include <vector>

#include "dli/engine.h"
#include "dli/operator.h"
#include "test_support.h"

namespace {

class IdentityOp final : public dli::Operator {
 public:
  std::string type() const override { return "identity_test"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs, const dli::Attributes&,
               dli::ExecutionContext&) const override {
    if (inputs.size() != 1 || outputs.size() != 1)
      throw std::invalid_argument("identity_test arity");
    *outputs[0] = *inputs[0];
  }
};

class StateCounterOp final : public dli::Operator {
 public:
  std::string type() const override { return "state_counter_test"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs, const dli::Attributes&,
               dli::ExecutionContext& context) const override {
    if (!inputs.empty() || outputs.size() != 1 || context.state == nullptr)
      throw std::invalid_argument("state_counter_test arity or state");
    std::int64_t count = 0;
    if (const auto* previous = context.state->findTensor("counter")) {
      count = previous->data<std::int64_t>()[0];
    }
    dli::Tensor next(dli::DType::Int64, {1});
    next.data<std::int64_t>()[0] = count + 1;
    context.state->setTensor("counter", next);
    dli::Tensor result(dli::DType::Int64, {2});
    result.data<std::int64_t>()[0] = count + 1;
    result.data<std::int64_t>()[1] = context.position_offset;
    *outputs[0] = std::move(result);
  }
};

}  // namespace

int main() {
  return dli_test::run("Engine", [] {
    dli::Engine engine;
    dli_test::expect(engine.registry().contains("aten"),
                     "aten dispatcher should be a core builtin");
    dli_test::expect(!engine.registry().contains("conv2d"), "conv2d must not be a CPU builtin");
    dli_test::expect(!engine.registry().contains("attention"),
                     "attention must not be a CPU builtin");
    dli_test::expect(!engine.registry().contains("linear"), "linear must not be a CPU builtin");
    dli_test::expect(!engine.registry().contains("rms_norm"), "rms_norm must not be a CPU builtin");

    dli::Graph graph;
    graph.inputs = {"x"};
    graph.outputs = {"missing_output"};
    graph.nodes.push_back({"identity", "identity_test", {"x"}, {"y"}, {}});
    auto input = dli::Tensor(at::tensor({1.0f}, at::TensorOptions().dtype(at::kFloat)));
    engine.registry().registerFactory("identity_test",
                                      [] { return std::make_unique<IdentityOp>(); });
    engine.registry().registerFactory("state_counter_test",
                                      [] { return std::make_unique<StateCounterOp>(); });
    dli_test::expectThrows([&] { engine.run(graph, {}); }, "engine missing input should throw");
    dli_test::expectThrows([&] { engine.run(graph, {{"x", input}}); },
                           "engine missing graph output should throw");

    dli::Graph state_graph;
    state_graph.outputs = {"result"};
    state_graph.nodes.push_back({"state_counter", "state_counter_test", {}, {"result"}, {}});
    dli::ExecutionState first_state;
    dli::ExecutionState second_state;
    auto first = engine.run(state_graph, {}, {.state = &first_state, .position_offset = 7});
    auto second = engine.run(state_graph, {}, {.state = &first_state, .position_offset = 8});
    auto isolated = engine.run(state_graph, {}, {.state = &second_state, .position_offset = 3});
    dli_test::expect(first.at("result").data<std::int64_t>()[0] == 1, "first state starts at one");
    dli_test::expect(first.at("result").data<std::int64_t>()[1] == 7,
                     "engine forwards the position offset");
    dli_test::expect(second.at("result").data<std::int64_t>()[0] == 2,
                     "state persists between runs");
    dli_test::expect(isolated.at("result").data<std::int64_t>()[0] == 1,
                     "execution states are isolated");

    dli::Graph failing_state_graph;
    failing_state_graph.outputs = {"result"};
    failing_state_graph.nodes.push_back(
        {"state_counter", "state_counter_test", {}, {"result"}, {}});
    failing_state_graph.nodes.push_back({"fail_after_state", "missing_operator", {}, {}, {}});
    dli_test::expectThrows([&] { engine.run(failing_state_graph, {}, {.state = &first_state}); },
                           "failed runs do not publish partial state");
    auto after_failure = engine.run(state_graph, {}, {.state = &first_state});
    dli_test::expect(after_failure.at("result").data<std::int64_t>()[0] == 3,
                     "state rolls back when a later node fails");

    first_state.reset();
    auto reset = engine.run(state_graph, {}, {.state = &first_state});
    dli_test::expect(reset.at("result").data<std::int64_t>()[0] == 1, "reset clears named state");
    dli_test::expectThrows(
        [&] { engine.run(state_graph, {}, {.state = &first_state, .position_offset = -1}); },
        "engine rejects negative positions");
  });
}
