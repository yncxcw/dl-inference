#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "dli/cuda_runtime.h"
#include "dli/engine.h"
#include "dli/graph.h"
#include "dli/weights.h"

namespace {

struct Args {
  std::string graph = "build/examples/qwen2/qwen2.dli.json";
  std::string weights = "build/examples/qwen2/qwen2.dli.weights.json";
  std::string plugin = "build/operators/libdli_triton_aot_ops.so";
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--graph" && i + 1 < argc) args.graph = argv[++i];
    else if (arg == "--weights" && i + 1 < argc) args.weights = argv[++i];
    else if (arg == "--plugin" && i + 1 < argc) args.plugin = argv[++i];
    else throw std::invalid_argument("usage: dli_qwen2 [--graph path] [--weights path] [--plugin path]");
  }
  return args;
}

dli::Tensor torchCudaInt64(std::vector<std::int64_t> shape, const std::vector<std::int64_t>& values) {
  auto tensor = at::tensor(values, at::TensorOptions().dtype(at::kLong)).view(shape);
  return dli::Tensor(tensor.to(at::Device(at::kCUDA, 0)));
}

std::vector<float> downloadFloat(const dli::Tensor& tensor) {
  std::vector<float> values(tensor.numel());
  dli::cudaMemcpyBytes(values.data(), tensor.deviceData(), tensor.nbytes(), dli::CudaMemcpyKind::DeviceToHost);
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parseArgs(argc, argv);
    dli::Engine engine;
    engine.registry().loadLibrary(args.plugin);
    auto tensors = dli::loadWeights(args.weights, dli::DeviceType::Cuda);
    tensors["input_ids"] = torchCudaInt64({1}, {2});
    auto outputs = engine.run(dli::Graph::fromJsonFile(args.graph), std::move(tensors));
    const auto values = downloadFloat(outputs.at("logits"));
    std::cout << "logits:";
    for (std::size_t i = 0; i < std::min<std::size_t>(values.size(), 16); ++i) std::cout << " " << values[i];
    std::cout << "\nkv_cache_entries: " << engine.kvCache().size() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen2 example failed: " << error.what() << "\n";
    return 1;
  }
}
