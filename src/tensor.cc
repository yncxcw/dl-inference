#include "dli/tensor.h"

#include "dli/cuda_runtime.h"

#include <cstring>
#include <limits>

namespace dli {

struct Tensor::DeviceAllocation {
  explicit DeviceAllocation(void* ptr) : pointer(ptr) {}
  ~DeviceAllocation() { cudaFreePointer(pointer); }
  void* pointer = nullptr;
};

std::string toString(DeviceType device) {
  switch (device) {
    case DeviceType::Cpu: return "cpu";
    case DeviceType::Cuda: return "cuda";
  }
  throw std::invalid_argument("unknown device type");
}

DeviceType deviceFromString(const std::string& device) {
  if (device == "cpu") return DeviceType::Cpu;
  if (device == "cuda" || device == "gpu") return DeviceType::Cuda;
  throw std::invalid_argument("unsupported device type: " + device);
}

std::string toString(DType dtype) {
  switch (dtype) {
    case DType::Float32: return "float32";
    case DType::Int64: return "int64";
  }
  throw std::invalid_argument("unknown dtype");
}

DType dtypeFromString(const std::string& dtype) {
  if (dtype == "float32" || dtype == "f32") return DType::Float32;
  if (dtype == "int64" || dtype == "i64") return DType::Int64;
  throw std::invalid_argument("unsupported dtype: " + dtype);
}

std::size_t byteSize(DType dtype) {
  switch (dtype) {
    case DType::Float32: return sizeof(float);
    case DType::Int64: return sizeof(std::int64_t);
  }
  throw std::invalid_argument("unknown dtype");
}

Tensor::Tensor(DType dtype, std::vector<std::int64_t> shape)
    : dtype_(dtype), shape_(std::move(shape)) {
  if (shape_.empty()) shape_.push_back(1);
  for (const auto dim : shape_) {
    if (dim < 0) throw std::invalid_argument("negative tensor dimension");
  }
  storage_.resize(numel() * byteSize(dtype_));
}

Tensor Tensor::zeros(DType dtype, std::vector<std::int64_t> shape) {
  return Tensor(dtype, std::move(shape));
}

Tensor Tensor::cuda(DType dtype, std::vector<std::int64_t> shape, int device_id) {
  Tensor tensor;
  tensor.dtype_ = dtype;
  tensor.device_ = DeviceType::Cuda;
  tensor.device_id_ = device_id;
  tensor.shape_ = std::move(shape);
  if (tensor.shape_.empty()) tensor.shape_.push_back(1);
  for (const auto dim : tensor.shape_) {
    if (dim < 0) throw std::invalid_argument("negative tensor dimension");
  }
  tensor.device_allocation_ = std::make_shared<DeviceAllocation>(cudaMallocBytes(tensor.nbytes()));
  return tensor;
}

Tensor Tensor::externalCuda(DType dtype, std::vector<std::int64_t> shape, void* data,
                            int device_id) {
  if (data == nullptr) throw std::invalid_argument("external CUDA tensor pointer is null");
  Tensor tensor;
  tensor.dtype_ = dtype;
  tensor.device_ = DeviceType::Cuda;
  tensor.device_id_ = device_id;
  tensor.shape_ = std::move(shape);
  if (tensor.shape_.empty()) tensor.shape_.push_back(1);
  for (const auto dim : tensor.shape_) {
    if (dim < 0) throw std::invalid_argument("negative tensor dimension");
  }
  tensor.external_device_ptr_ = data;
  return tensor;
}

Tensor Tensor::fromFloat32(std::vector<std::int64_t> shape, std::vector<float> values) {
  Tensor tensor(DType::Float32, std::move(shape));
  if (tensor.numel() != values.size()) {
    throw std::invalid_argument("float32 tensor data size does not match shape");
  }
  std::memcpy(tensor.rawStorage().data(), values.data(), values.size() * sizeof(float));
  return tensor;
}

Tensor Tensor::fromInt64(std::vector<std::int64_t> shape, std::vector<std::int64_t> values) {
  Tensor tensor(DType::Int64, std::move(shape));
  if (tensor.numel() != values.size()) {
    throw std::invalid_argument("int64 tensor data size does not match shape");
  }
  std::memcpy(tensor.rawStorage().data(), values.data(), values.size() * sizeof(std::int64_t));
  return tensor;
}

std::int64_t Tensor::dim(std::size_t index) const {
  if (index >= shape_.size()) throw std::out_of_range("tensor dimension out of range");
  return shape_[index];
}

std::size_t Tensor::numel() const {
  std::size_t result = 1;
  for (const auto dim : shape_) {
    if (dim == 0) return 0;
    if (static_cast<std::uint64_t>(dim) > std::numeric_limits<std::size_t>::max() / result) {
      throw std::overflow_error("tensor shape is too large");
    }
    result *= static_cast<std::size_t>(dim);
  }
  return result;
}

std::size_t Tensor::nbytes() const {
  return numel() * byteSize(dtype_);
}

void* Tensor::deviceData() {
  if (device_ != DeviceType::Cuda) throw std::logic_error("tensor is not on CUDA");
  return device_allocation_ ? device_allocation_->pointer : external_device_ptr_;
}

const void* Tensor::deviceData() const {
  if (device_ != DeviceType::Cuda) throw std::logic_error("tensor is not on CUDA");
  return device_allocation_ ? device_allocation_->pointer : external_device_ptr_;
}

Tensor Tensor::withShape(std::vector<std::int64_t> shape) const {
  Tensor view = *this;
  view.shape_ = std::move(shape);
  if (view.shape_.empty()) view.shape_.push_back(1);
  for (const auto dim : view.shape_) {
    if (dim < 0) throw std::invalid_argument("negative tensor dimension");
  }
  if (view.numel() != numel()) throw std::invalid_argument("tensor view changes element count");
  return view;
}

bool sameShape(const Tensor& lhs, const Tensor& rhs) {
  return lhs.shape() == rhs.shape();
}

std::size_t contiguousOffset(const std::vector<std::int64_t>& shape,
                             const std::vector<std::int64_t>& indices) {
  if (shape.size() != indices.size()) throw std::invalid_argument("rank mismatch in contiguousOffset");
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (std::size_t reverse = 0; reverse < shape.size(); ++reverse) {
    const auto i = shape.size() - 1 - reverse;
    if (indices[i] < 0 || indices[i] >= shape[i]) throw std::out_of_range("tensor index out of range");
    offset += static_cast<std::size_t>(indices[i]) * stride;
    stride *= static_cast<std::size_t>(shape[i]);
  }
  return offset;
}

}  // namespace dli
