#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <ATen/ATen.h>

#include "dli/cuda_runtime.h"
#include "dli/engine.h"
#include "dli/graph.h"
#include "dli/logging.h"
#include "dli/weights.h"

namespace {

struct Args {
  std::string plugin_path;
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--plugin" && i + 1 < argc) args.plugin_path = argv[++i];
    else throw std::invalid_argument("unknown or incomplete test argument: " + arg);
  }
  return args;
}

void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void testHostTensorMetadata() {
  auto torch_tensor = at::tensor({1.0f, 2.0f, 3.0f, 4.0f},
                                 at::TensorOptions().dtype(at::kFloat)).view({2, 2});
  auto tensor = dli::Tensor(torch_tensor);
  expect(tensor.isCpu(), "host tensor device");
  expect(tensor.numel() == 4, "host tensor numel");
  expect(tensor.nbytes() == 16, "host tensor nbytes");
  expect(tensor.data<float>()[2] == 3.0f, "host tensor value");
}

void testExternalCudaTensorMetadata() {
  auto* fake_ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1000));
  auto tensor = dli::Tensor::externalCuda(dli::DType::Float32, {2, 3}, fake_ptr, 0);
  expect(tensor.isCuda(), "external tensor is CUDA");
  expect(tensor.deviceData() == fake_ptr, "external CUDA pointer");
  expect(tensor.nbytes() == 24, "external CUDA nbytes");
  auto view = tensor.withShape({3, 2});
  expect(view.isCuda(), "CUDA view device");
  expect(view.deviceData() == fake_ptr, "CUDA view pointer");
  expect(view.shape() == std::vector<std::int64_t>({3, 2}), "CUDA view shape");
}

void testEngineHasNoCpuDeepLearningBuiltins() {
  dli::Engine engine;
  expect(engine.registry().contains("aten"), "aten dispatcher should be a core builtin");
  expect(!engine.registry().contains("conv2d"), "conv2d must not be a CPU builtin");
  expect(!engine.registry().contains("attention"), "attention must not be a CPU builtin");
  expect(!engine.registry().contains("linear"), "linear must not be a CPU builtin");
  expect(!engine.registry().contains("rms_norm"), "rms_norm must not be a CPU builtin");
}

void expectAllClose(const at::Tensor& actual, const at::Tensor& expected, double atol = 1.0e-6) {
  expect(at::allclose(actual, expected, atol, 0.0), "tensor values differ");
}

void testAtenReluOperator() {
  dli::Graph graph;
  graph.model_type = "aten_relu_test";
  graph.inputs = {"x"};
  graph.outputs = {"y"};
  dli::Attributes attrs;
  attrs.set("name", std::string("aten::relu"));
  graph.nodes.push_back({"relu", "aten", {"x"}, {"y"}, std::move(attrs)});

  auto input = at::tensor({-1.0f, 0.5f, 2.0f, -3.0f}, at::TensorOptions().dtype(at::kFloat));
  dli::Engine engine;
  auto outputs = engine.run(graph, {{"x", dli::Tensor(input)}});
  expectAllClose(outputs.at("y").torchTensor(), at::relu(input));
}

void testAtenAddOperator() {
  dli::Graph graph;
  graph.model_type = "aten_add_test";
  graph.inputs = {"lhs", "rhs"};
  graph.outputs = {"sum"};
  dli::Attributes attrs;
  attrs.set("name", std::string("aten::add"));
  attrs.set("overload", std::string("Tensor"));
  attrs.set("attr_order", std::vector<std::string>{"alpha"});
  attrs.set("alpha", 2.0);
  graph.nodes.push_back({"add", "aten", {"lhs", "rhs"}, {"sum"}, std::move(attrs)});

  auto lhs = at::tensor({1.0f, 2.0f, 3.0f}, at::TensorOptions().dtype(at::kFloat));
  auto rhs = at::tensor({10.0f, 20.0f, 30.0f}, at::TensorOptions().dtype(at::kFloat));
  dli::Engine engine;
  auto outputs = engine.run(graph, {{"lhs", dli::Tensor(lhs)}, {"rhs", dli::Tensor(rhs)}});
  expectAllClose(outputs.at("sum").torchTensor(), at::add(lhs, rhs, 2.0));
}

void testDefaultLoggingSeverityLevel() {
  expect(dli::logLevel() == dli::LogSeverity::Warn, "default logging level is WARN");
}

void testLoggingMacros() {
  struct Record {
    dli::LogSeverity severity;
    std::string message;
  };
  std::vector<Record> records;
  dli::setLogSink([&](dli::LogSeverity severity, std::string_view, int, std::string_view message) {
    records.push_back({severity, std::string(message)});
  });
  const auto previous_level = dli::logLevel();
  dli::setLogLevel(dli::LogSeverity::Debug);
  LOG_DEBUG << "hello " << "debug";
  LOG_INFO << "hello " << "info";
  LOG_WARN << "hello " << "warn";
  LOG_ERROR << "hello " << "error";
  dli::setLogLevel(previous_level);
  dli::resetLogSink();

  expect(records.size() == 4, "logging record count");
  expect(records[0].severity == dli::LogSeverity::Debug, "debug severity");
  expect(records[0].message == "hello debug", "debug message");
  expect(records[1].severity == dli::LogSeverity::Info, "info severity");
  expect(records[1].message == "hello info", "info message");
  expect(records[2].severity == dli::LogSeverity::Warn, "warn severity");
  expect(records[2].message == "hello warn", "warn message");
  expect(records[3].severity == dli::LogSeverity::Error, "error severity");
  expect(records[3].message == "hello error", "error message");
}

void testLoggingSeverityLevel() {
  struct Record {
    dli::LogSeverity severity;
    std::string message;
  };
  std::vector<Record> records;
  dli::setLogSink([&](dli::LogSeverity severity, std::string_view, int, std::string_view message) {
    records.push_back({severity, std::string(message)});
  });
  const auto previous_level = dli::logLevel();
  dli::setLogLevel(dli::LogSeverity::Warn);
  LOG_DEBUG << "debug";
  LOG_INFO << "info";
  LOG_WARN << "warn";
  LOG_ERROR << "error";
  dli::setLogLevel(previous_level);
  dli::resetLogSink();

  expect(records.size() == 2, "logging severity level record count");
  expect(records[0].severity == dli::LogSeverity::Warn, "warn threshold keeps warn");
  expect(records[0].message == "warn", "warn threshold message");
  expect(records[1].severity == dli::LogSeverity::Error, "warn threshold keeps error");
  expect(records[1].message == "error", "error threshold message");
}

void testWeightManifestLoad() {
  const std::string bin_path = "dli_test_weights.bin";
  const std::string manifest_path = "dli_test_weights.json";
  {
    const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::int64_t ids[] = {7, 8};
    std::ofstream bin(bin_path, std::ios::binary);
    bin.write(reinterpret_cast<const char*>(weight), sizeof(weight));
    bin.write(reinterpret_cast<const char*>(ids), sizeof(ids));
  }
  {
    std::ofstream manifest(manifest_path);
    manifest << R"({
  "format": "dli.weights.v1",
  "data": "dli_test_weights.bin",
  "tensors": {
    "linear.weight": {"dtype": "float32", "shape": [2, 2], "offset": 0, "nbytes": 16},
    "ids": {"dtype": "int64", "shape": [2], "offset": 16, "nbytes": 16}
  }
})";
  }

  const auto weights = dli::loadWeights(manifest_path);
  expect(weights.at("linear.weight").isCpu(), "weights load to CPU by default");
  expect(weights.at("linear.weight").shape() == std::vector<std::int64_t>({2, 2}), "weight shape");
  expect(weights.at("linear.weight").data<float>()[3] == 4.0f, "weight value");
  expect(weights.at("ids").data<std::int64_t>()[1] == 8, "int64 weight value");
  std::remove(bin_path.c_str());
  std::remove(manifest_path.c_str());
}

void testPluginRegistersGenericOperators(const std::string& plugin_path) {
  expect(!plugin_path.empty(), "Triton plugin path was not supplied");
  const std::vector<std::string> operator_types = {
      "embedding", "rms_norm", "linear", "matmul", "add", "mul", "relu", "silu",
      "max_pool2d", "softmax", "reshape", "transpose", "conv2d", "attention",
      "rotary_embedding"};

  dli::Engine engine;
  for (const auto& type : operator_types) {
    expect(!engine.registry().contains(type), "operator should require plugin: " + type);
  }
  engine.registry().loadLibrary(plugin_path);
  for (const auto& type : operator_types) {
    expect(engine.registry().contains(type), "Triton plugin did not register operator: " + type);
  }

  dli::Graph graph;
  graph.model_type = "plugin_registration_test";
  graph.inputs = {"input", "weight"};
  graph.outputs = {"output"};
  graph.nodes.push_back({"linear", "linear", {"input", "weight"}, {"output"}, {}});
  const auto roundtrip = dli::Graph::fromJson(graph.toJson());
  expect(roundtrip.nodes.size() == graph.nodes.size(), "graph JSON roundtrip node count");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parseArgs(argc, argv);
    testHostTensorMetadata();
    testExternalCudaTensorMetadata();
    testEngineHasNoCpuDeepLearningBuiltins();
    testAtenReluOperator();
    testAtenAddOperator();
    testDefaultLoggingSeverityLevel();
    testLoggingMacros();
    testLoggingSeverityLevel();
    testWeightManifestLoad();
    testPluginRegistersGenericOperators(args.plugin_path);
    std::cout << "all dli CPU-side contract tests passed";
    if (!dli::cudaRuntimeAvailable()) {
      std::cout << " (CUDA runtime unavailable here: " << dli::cudaRuntimeError() << ")";
    }
    std::cout << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << "\n";
    return 1;
  }
}
