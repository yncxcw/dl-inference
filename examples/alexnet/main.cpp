#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dli/cuda_runtime.h"
#include "dli/engine.h"
#include "dli/graph.h"
#include "dli/weights.h"

namespace {

struct Args {
  std::string graph = "build/examples/alexnet/alexnet.dli.json";
  std::string weights = "build/examples/alexnet/alexnet.dli.weights.json";
  std::string plugin = "build/operators/libdli_triton_aot_ops.so";
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--graph" && i + 1 < argc) args.graph = argv[++i];
    else if (arg == "--weights" && i + 1 < argc) args.weights = argv[++i];
    else if (arg == "--plugin" && i + 1 < argc) args.plugin = argv[++i];
    else throw std::invalid_argument("usage: dli_alexnet [--graph path] [--weights path] [--plugin path]");
  }
  return args;
}

dli::Tensor uploadInput() {
  std::vector<float> values(1 * 3 * 32 * 32);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>((i % 23) - 11) / 23.0f;
  dli::Tensor tensor = dli::Tensor::cuda(dli::DType::Float32, {1, 3, 32, 32});
  dli::cudaMemcpyBytes(tensor.deviceData(), values.data(), tensor.nbytes(), dli::CudaMemcpyKind::HostToDevice);
  return tensor;
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
    tensors["x"] = uploadInput();
    auto outputs = engine.run(dli::Graph::fromJsonFile(args.graph), std::move(tensors));
    const auto values = downloadFloat(outputs.begin()->second);
    std::cout << "output:";
    for (std::size_t i = 0; i < std::min<std::size_t>(values.size(), 8); ++i) std::cout << " " << values[i];
    std::cout << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "alexnet example failed: " << error.what() << "\n";
    return 1;
  }
}
