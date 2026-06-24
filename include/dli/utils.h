#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dli/attributes.h"
#include "dli/tensor.h"

namespace dli {

std::int64_t product(const std::vector<std::int64_t>& shape, std::size_t begin = 0,
                     std::size_t end = static_cast<std::size_t>(-1));
unsigned ceilDiv(std::int64_t a, std::int64_t b);
std::string formatShape(const std::vector<std::int64_t>& shape);
void requireCudaInputs(const std::vector<const Tensor*>& inputs, const std::string& op);
void requireFloat(const Tensor& tensor, const std::string& op);
double attrDouble(const Attributes& attrs, const std::string& name, double fallback);
std::vector<std::int64_t> attrInts(const Attributes& attrs, const std::string& name,
                                   std::vector<std::int64_t> fallback);
void* ptr(const Tensor& tensor);
std::vector<std::int64_t> reshapeShape(const std::vector<std::int64_t>& input,
                                       std::vector<std::int64_t> target);

}  // namespace dli
