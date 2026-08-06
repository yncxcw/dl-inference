#include "dli/utils.h"

#include <sstream>
#include <stdexcept>

namespace dli {

std::int64_t product(const std::vector<std::int64_t>& shape, std::size_t begin, std::size_t end) {
  if (end == static_cast<std::size_t>(-1)) end = shape.size();
  std::int64_t out = 1;
  for (std::size_t i = begin; i < end; ++i) out *= shape[i];
  return out;
}

unsigned ceilDiv(std::int64_t a, std::int64_t b) { return static_cast<unsigned>((a + b - 1) / b); }

std::string formatShape(const std::vector<std::int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ',';
    out << shape[i];
  }
  out << ']';
  return out.str();
}

void requireCudaInputs(const std::vector<const Tensor*>& inputs, const std::string& op) {
  for (const auto* tensor : inputs) {
    if (!tensor->isCuda()) throw std::invalid_argument(op + " expects CUDA tensors");
  }
}

void requireFloat(const Tensor& tensor, const std::string& op) {
  if (tensor.dtype() != DType::Float32) throw std::invalid_argument(op + " expects float32");
}

double attrDouble(const Attributes& attrs, const std::string& name, double fallback) {
  if (!attrs.contains(name)) return fallback;
  if (attrs.values().at(name).index() == 0) {
    return static_cast<double>(attrs.require<std::int64_t>(name));
  }
  return attrs.require<double>(name);
}

std::vector<std::int64_t> attrInts(const Attributes& attrs, const std::string& name,
                                   std::vector<std::int64_t> fallback) {
  if (!attrs.contains(name)) return fallback;
  return attrs.require<std::vector<std::int64_t>>(name);
}

void* ptr(const Tensor& tensor) { return const_cast<void*>(tensor.deviceData()); }

std::vector<std::int64_t> reshapeShape(const std::vector<std::int64_t>& input,
                                       std::vector<std::int64_t> target) {
  std::int64_t known = 1;
  int infer = -1;
  for (std::size_t i = 0; i < target.size(); ++i) {
    if (target[i] == -1) {
      if (infer != -1) throw std::invalid_argument("reshape has multiple inferred dimensions");
      infer = static_cast<int>(i);
    } else if (target[i] <= 0) {
      throw std::invalid_argument("reshape dimension must be positive or -1");
    } else {
      known *= target[i];
    }
  }
  const auto numel = product(input);
  if (infer != -1) {
    if (numel % known != 0)
      throw std::invalid_argument("reshape inferred dimension is not integral");
    target[static_cast<std::size_t>(infer)] = numel / known;
  }
  if (product(target) != numel) throw std::invalid_argument("reshape changes element count");
  return target;
}

}  // namespace dli
