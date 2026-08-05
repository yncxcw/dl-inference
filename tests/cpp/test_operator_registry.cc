#include "test_support.h"

#include <memory>
#include <string>
#include <vector>

#include "dli/operator.h"

namespace {

class IdentityOp final : public dli::Operator {
 public:
  std::string type() const override { return "identity_test"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes&,
               dli::ExecutionContext&) const override {
    if (inputs.size() != 1 || outputs.size() != 1) throw std::invalid_argument("identity_test arity");
    *outputs[0] = *inputs[0];
  }
};

std::string parsePluginPath(int argc, char** argv) {
  std::string plugin_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--plugin" && i + 1 < argc) {
      plugin_path = argv[++i];
    } else {
      throw std::invalid_argument("unknown or incomplete test argument: " + arg);
    }
  }
  return plugin_path;
}

}  // namespace

int main(int argc, char** argv) {
  return dli_test::run("OperatorRegistry", [&] {
    const auto plugin_path = parsePluginPath(argc, argv);
    dli::OperatorRegistry registry;
    dli_test::expect(!registry.contains("identity_test"), "registry initially empty");
    dli_test::expectThrows([&] { registry.create("identity_test"); },
                           "unknown operator should throw");
    registry.registerFactory("identity_test", [] { return std::make_unique<IdentityOp>(); });
    dli_test::expect(registry.contains("identity_test"), "registry contains registered op");
    dli_test::expect(registry.create("identity_test")->type() == "identity_test",
                     "registry creates registered op");
    dli_test::expectThrows([&] { registry.loadLibrary("/tmp/dli_missing_operator_plugin.so"); },
                           "missing plugin load should throw");

    if (!plugin_path.empty()) {
      const std::vector<std::string> operator_types = {
          "embedding", "rms_norm", "linear", "matmul", "add", "mul", "relu", "silu",
          "max_pool2d", "softmax", "reshape", "transpose", "conv2d", "attention",
          "rotary_embedding"};
      for (const auto& type : operator_types) {
        dli_test::expect(!registry.contains(type), "operator should require plugin: " + type);
      }
      registry.loadLibrary(plugin_path);
      for (const auto& type : operator_types) {
        dli_test::expect(registry.contains(type), "Triton plugin did not register operator: " + type);
      }
    }
  });
}

