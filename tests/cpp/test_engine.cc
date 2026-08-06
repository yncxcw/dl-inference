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
    dli_test::expectThrows([&] { engine.run(graph, {}); }, "engine missing input should throw");
    dli_test::expectThrows([&] { engine.run(graph, {{"x", input}}); },
                           "engine missing graph output should throw");
  });
}
