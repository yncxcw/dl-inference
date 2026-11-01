/**
 * @file operator_engine_runner.cc
 * @brief Command-line tool that runs a graph with a plugin library on CUDA and dumps the outputs to
 * files.
 */

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dli/cuda_runtime.h"
#include "dli/engine.h"
#include "dli/graph.h"
#include "dli/tensor.h"
#include "dli/weights.h"

namespace {

/// @brief Parsed command-line arguments of the runner.
struct Args {
  std::string graph;
  std::string inputs;
  std::string plugin;
  std::string output_dir;
};

/**
 * @brief Parses and validates the command line.
 * @param argc Argument count from main().
 * @param argv Argument vector from main().
 * @return The four required paths: --graph, --inputs, --plugin and --output-dir.
 * @throws std::invalid_argument On an unknown or missing argument.
 */
Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--graph" && i + 1 < argc)
      args.graph = argv[++i];
    else if (arg == "--inputs" && i + 1 < argc)
      args.inputs = argv[++i];
    else if (arg == "--plugin" && i + 1 < argc)
      args.plugin = argv[++i];
    else if (arg == "--output-dir" && i + 1 < argc)
      args.output_dir = argv[++i];
    else
      throw std::invalid_argument(
          "usage: dli_operator_engine_runner --graph path --inputs path --plugin path --output-dir "
          "path");
  }
  if (args.graph.empty() || args.inputs.empty() || args.plugin.empty() || args.output_dir.empty()) {
    throw std::invalid_argument("missing required runner argument");
  }
  return args;
}

/// @brief Escapes quote and backslash for embedding @p value in a JSON string literal.
std::string escapeJson(const std::string& value) {
  std::string out;
  for (const char c : value) {
    if (c == '"')
      out += "\\\"";
    else if (c == '\\')
      out += "\\\\";
    else
      out.push_back(c);
  }
  return out;
}

/**
 * @brief Maps an output tensor name to its file name (`<name>.bin`).
 * @throws std::invalid_argument If the name has characters outside [A-Za-z0-9_-].
 */
std::string outputFileName(const std::string& name) {
  for (const char c : name) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                    c == '_' || c == '-';
    if (!ok) throw std::invalid_argument("unsupported output tensor name: " + name);
  }
  return name + ".bin";
}

/// @brief Copies a tensor's raw bytes to host memory (from the GPU when necessary).
std::vector<std::byte> tensorBytes(const dli::Tensor& tensor) {
  std::vector<std::byte> bytes(tensor.nbytes());
  if (tensor.isCuda()) {
    dli::cudaMemcpyBytes(bytes.data(), tensor.deviceData(), bytes.size(),
                         dli::CudaMemcpyKind::DeviceToHost);
  } else {
    const void* source = tensor.dtype() == dli::DType::Float32
                             ? static_cast<const void*>(tensor.data<float>())
                             : static_cast<const void*>(tensor.data<std::int64_t>());
    std::memcpy(bytes.data(), source, bytes.size());
  }
  return bytes;
}

/// @brief Writes @p bytes to a binary file; throws std::runtime_error if it cannot be opened.
void writeBytes(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("failed to open output file: " + path.string());
  if (!bytes.empty())
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

/// @brief Writes the "dli.outputs.v1" manifest listing each output's dtype, shape and file.
void writeManifest(const std::filesystem::path& path,
                   const std::vector<std::pair<std::string, dli::Tensor>>& outputs) {
  std::ofstream file(path);
  if (!file) throw std::runtime_error("failed to open output manifest: " + path.string());
  file << "{\n  \"format\": \"dli.outputs.v1\",\n  \"tensors\": {\n";
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& [name, tensor] = outputs[i];
    file << "    \"" << escapeJson(name) << "\": {\"dtype\": \"" << dli::toString(tensor.dtype())
         << "\", \"shape\": [";
    for (std::size_t j = 0; j < tensor.shape().size(); ++j) {
      if (j) file << ", ";
      file << tensor.shape()[j];
    }
    file << "], \"file\": \"" << escapeJson(outputFileName(name)) << "\"}";
    if (i + 1 != outputs.size()) file << ",";
    file << "\n";
  }
  file << "  }\n}\n";
}

}  // namespace

/// @brief Runs the graph and writes one .bin per output plus outputs.json; returns 0 on success.
int main(int argc, char** argv) {
  try {
    const auto args = parseArgs(argc, argv);
    std::filesystem::create_directories(args.output_dir);
    auto graph = dli::Graph::fromJsonFile(args.graph);
    auto inputs = dli::loadWeights(args.inputs, dli::DeviceType::Cuda);
    dli::Engine engine;
    engine.registry().loadLibrary(args.plugin);
    auto result = engine.run(graph, std::move(inputs));
    std::vector<std::pair<std::string, dli::Tensor>> ordered;
    for (const auto& name : graph.outputs) {
      auto it = result.find(name);
      if (it == result.end()) throw std::runtime_error("missing graph output: " + name);
      writeBytes(std::filesystem::path(args.output_dir) / outputFileName(name),
                 tensorBytes(it->second));
      ordered.emplace_back(name, it->second);
    }
    writeManifest(std::filesystem::path(args.output_dir) / "outputs.json", ordered);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "operator engine runner failed: " << error.what() << "\n";
    return 1;
  }
}
