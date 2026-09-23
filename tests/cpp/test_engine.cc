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

class MutateStateThenFailOp final : public dli::Operator {
 public:
  std::string type() const override { return "mutate_state_then_fail_test"; }
  void compute(const std::vector<const dli::Tensor*>&, const std::vector<dli::Tensor*>&,
               const dli::Attributes&, dli::ExecutionContext& context) const override {
    if (context.state == nullptr) throw std::invalid_argument("missing state");
    const auto* stored = context.state->findTensor("mutable");
    if (stored == nullptr) throw std::invalid_argument("missing mutable tensor");
    auto alias = stored->torchTensor();
    alias.add_(1);
    throw std::runtime_error("failure after in-place state mutation");
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
    engine.registry().registerFactory("mutate_state_then_fail_test",
                                      [] { return std::make_unique<MutateStateThenFailOp>(); });
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

    dli::ExecutionState mutating_state;
    dli::Tensor mutable_tensor(dli::DType::Int64, {1});
    mutable_tensor.data<std::int64_t>()[0] = 5;
    mutating_state.setTensor("mutable", mutable_tensor);
    dli::Graph mutating_failure_graph;
    mutating_failure_graph.nodes.push_back(
        {"mutate_then_fail", "mutate_state_then_fail_test", {}, {}, {}});
    dli_test::expectThrows(
        [&] { engine.run(mutating_failure_graph, {}, {.state = &mutating_state}); },
        "custom operator mutation failure should throw");
    dli_test::expect(mutating_state.findTensor("mutable")->data<std::int64_t>()[0] == 5,
                     "failed custom operator cannot mutate the published state snapshot");

    dli::ExecutionState initializing_state;
    dli::Graph mutating_initializer_graph = mutating_failure_graph;
    mutating_initializer_graph.state_inputs = {{"state_input", "mutable"}};
    mutating_initializer_graph.state_initializers = {{"mutable", "initial"}};
    dli_test::expectThrows(
        [&] {
          engine.run(mutating_initializer_graph, {{"initial", mutable_tensor}},
                     {.state = &initializing_state});
        },
        "custom operator initializer mutation failure should throw");
    dli_test::expect(mutable_tensor.data<std::int64_t>()[0] == 5,
                     "failed custom operator cannot mutate a shared initializer");
    dli_test::expect(initializing_state.findTensor("mutable") == nullptr,
                     "failed initialization is not published");

    first_state.reset();
    auto reset = engine.run(state_graph, {}, {.state = &first_state});
    dli_test::expect(reset.at("result").data<std::int64_t>()[0] == 1, "reset clears named state");
    dli_test::expectThrows(
        [&] { engine.run(state_graph, {}, {.state = &first_state, .position_offset = -1}); },
        "engine rejects negative positions");

    dli::Graph edge_state_graph;
    edge_state_graph.outputs = {"cache_out"};
    edge_state_graph.state_inputs = {{"cache_in", "layer.0.cache"}};
    edge_state_graph.state_outputs = {{"cache_out", "layer.0.cache"}};
    edge_state_graph.state_initializers = {{"layer.0.cache", "cache_initial"}};
    dli::Attributes add_attrs;
    add_attrs.set("name", std::string("aten::add"));
    add_attrs.set("overload", std::string("Tensor"));
    add_attrs.set("arguments", std::vector<std::string>{"t:0", "t:1", "i:1"});
    edge_state_graph.nodes.push_back(
        {"advance_cache", "aten", {"cache_in", "delta"}, {"cache_out"}, add_attrs});

    auto cache_initial = dli::Tensor(at::zeros({1}, at::TensorOptions().dtype(at::kFloat)));
    auto delta_one = dli::Tensor(at::ones({1}, at::TensorOptions().dtype(at::kFloat)));
    auto delta_two = dli::Tensor(at::full({1}, 2.0, at::TensorOptions().dtype(at::kFloat)));
    dli::ExecutionState edge_first;
    dli::ExecutionState edge_second;
    auto edge_step_one =
        engine.run(edge_state_graph, {{"cache_initial", cache_initial}, {"delta", delta_one}},
                   {.state = &edge_first});
    auto edge_step_two =
        engine.run(edge_state_graph, {{"cache_initial", cache_initial}, {"delta", delta_two}},
                   {.state = &edge_first});
    auto edge_isolated =
        engine.run(edge_state_graph, {{"cache_initial", cache_initial}, {"delta", delta_two}},
                   {.state = &edge_second});
    dli_test::expect(edge_step_one.at("cache_out").data<float>()[0] == 1.0f,
                     "state edge seeds from its initializer");
    dli_test::expect(edge_step_two.at("cache_out").data<float>()[0] == 3.0f,
                     "state edge feeds a prior output into the next run");
    dli_test::expect(edge_isolated.at("cache_out").data<float>()[0] == 2.0f,
                     "state edge remains request-local");

    auto failing_edge_graph = edge_state_graph;
    failing_edge_graph.nodes.push_back({"fail_after_cache", "missing_operator", {}, {}, {}});
    dli_test::expectThrows(
        [&] {
          engine.run(failing_edge_graph, {{"cache_initial", cache_initial}, {"delta", delta_one}},
                     {.state = &edge_first});
        },
        "state edge update rolls back after a later failure");
    auto edge_after_failure =
        engine.run(edge_state_graph, {{"cache_initial", cache_initial}, {"delta", delta_one}},
                   {.state = &edge_first});
    dli_test::expect(edge_after_failure.at("cache_out").data<float>()[0] == 4.0f,
                     "failed state-edge run does not publish its output");
  });
}
