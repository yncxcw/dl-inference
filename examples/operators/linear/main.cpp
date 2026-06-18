#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dli/cuda_runtime.h"
#include "dli/engine.h"
#include "dli/graph.h"

namespace {

struct Args {
  std::string graph = "examples/operators/linear/linear_graph.dli.json";
  std::string plugin = "build/operators/libdli_triton_aot_ops.so";
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--graph" && i + 1 < argc) args.graph = argv[++i];
    else if (arg == "--plugin" && i + 1 < argc) args.plugin = argv[++i];
    else throw std::invalid_argument("usage: dli_operator_linear [--graph path] [--plugin path]");
  }
  return args;
}

dli::Tensor uploadFloat(std::vector<std::int64_t> shape, const std::vector<float>& values) {
  dli::Tensor tensor = dli::Tensor::cuda(dli::DType::Float32, std::move(shape));
  if (tensor.numel() != values.size()) throw std::invalid_argument("float upload size mismatch");
  dli::cudaMemcpyBytes(tensor.deviceData(), values.data(), tensor.nbytes(), dli::CudaMemcpyKind::HostToDevice);
  return tensor;
}

std::vector<float> downloadFloat(const dli::Tensor& tensor) {
  std::vector<float> values(tensor.numel());
  dli::cudaMemcpyBytes(values.data(), tensor.deviceData(), tensor.nbytes(), dli::CudaMemcpyKind::DeviceToHost);
  return values;
}

dli::TensorMap linearInputs() {
  dli::TensorMap inputs;
  inputs["input"] = uploadFloat({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  inputs["weight"] = uploadFloat({3, 4}, {1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1});
  inputs["bias"] = uploadFloat({3}, {0.5f, -1.0f, 2.0f});
  return inputs;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parseArgs(argc, argv);
    dli::Engine engine;
    engine.registry().loadLibrary(args.plugin);
    auto outputs = engine.run(dli::Graph::fromJsonFile(args.graph), linearInputs());
    const auto values = downloadFloat(outputs.at("output"));
    std::cout << "output:";
    for (const auto value : values) std::cout << " " << value;
    std::cout << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "linear example failed: " << error.what() << "\n";
    return 1;
  }
}
