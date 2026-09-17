#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
          "embedding", "rms_norm",  "linear", "matmul",     "add",
          "mul",       "relu",      "silu",   "max_pool2d", "softmax",
          "reshape",   "transpose", "conv2d", "attention",  "rotary_embedding"};
      for (const auto& type : operator_types) {
        dli_test::expect(!registry.contains(type), "operator should require plugin: " + type);
      }
      registry.loadLibrary(plugin_path);
      for (const auto& type : operator_types) {
        dli_test::expect(registry.contains(type),
                         "Triton plugin did not register operator: " + type);
      }

      auto attention = registry.create("attention");
      dli::KVCache cache;
      dli::ExecutionContext context;
      context.kv_cache = &cache;
      dli::Attributes attrs;
      attrs.set("kv_cache", std::string("shape_validation"));
      auto* fake_device_pointer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(1));
      const auto expectRejectedWithoutCacheMutation =
          [&](const std::vector<std::int64_t>& q_shape, const std::vector<std::int64_t>& k_shape,
              const std::vector<std::int64_t>& v_shape, const std::string& message) {
            auto q = dli::Tensor::externalCuda(dli::DType::Float32, q_shape, fake_device_pointer);
            auto k = dli::Tensor::externalCuda(dli::DType::Float32, k_shape, fake_device_pointer);
            auto v = dli::Tensor::externalCuda(dli::DType::Float32, v_shape, fake_device_pointer);
            dli::Tensor output;
            dli_test::expectThrows(
                [&] { attention->compute({&q, &k, &v}, {&output}, attrs, context); }, message);
            dli_test::expect(cache.size() == 0, message + " must not mutate the KV cache");
          };
      expectRejectedWithoutCacheMutation({2, 8, 1, 2}, {1, 2, 1, 2}, {1, 2, 1, 2},
                                         "attention batch mismatch");
      expectRejectedWithoutCacheMutation({1, 7, 1, 2}, {1, 2, 1, 2}, {1, 2, 1, 2},
                                         "attention non-integral GQA ratio");
      expectRejectedWithoutCacheMutation({1, 8, 1, 4}, {1, 2, 1, 2}, {1, 2, 1, 2},
                                         "attention head-dimension mismatch");
      expectRejectedWithoutCacheMutation({1, 8, 1, 2}, {1, 2, 1, 2}, {1, 2, 2, 2},
                                         "attention key/value mismatch");
      expectRejectedWithoutCacheMutation({1, 8, 3, 2}, {1, 2, 3, 2}, {1, 2, 3, 2},
                                         "unsupported attention specialization");
    }
  });
}
