#include "dli/tensor.h"

#include <ATen/ATen.h>

#include <cstring>
#include <limits>
#include <utility>

namespace dli {
namespace {

at::ScalarType torchDType(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return at::kFloat;
    case DType::Int64:
      return at::kLong;
  }
  throw std::invalid_argument("unknown dtype");
}

DType dtypeFromTorch(at::ScalarType dtype) {
  if (dtype == at::kFloat) return DType::Float32;
  if (dtype == at::kLong) return DType::Int64;
  throw std::invalid_argument("unsupported torch tensor dtype");
}

DeviceType deviceFromTorch(const at::Tensor& tensor) {
  if (tensor.device().is_cpu()) return DeviceType::Cpu;
  if (tensor.device().is_cuda()) return DeviceType::Cuda;
  throw std::invalid_argument("unsupported torch tensor device");
}

std::vector<std::int64_t> shapeFromTorch(const at::Tensor& tensor) {
  return {tensor.sizes().begin(), tensor.sizes().end()};
}

at::TensorOptions optionsFor(DType dtype, DeviceType device, int device_id = 0) {
  auto options = at::TensorOptions().dtype(torchDType(dtype));
  if (device == DeviceType::Cpu) return options.device(at::kCPU);
  return options.device(at::Device(at::kCUDA, device_id));
}

void checkHostData(const Tensor& tensor, DType dtype) {
  if (!tensor.isCpu()) throw std::logic_error("host data access requested for non-CPU tensor");
  if (tensor.dtype() != dtype)
    throw std::logic_error("tensor dtype does not match requested host type");
}

}  // namespace

struct Tensor::Impl {
  explicit Impl(at::Tensor value) : tensor(std::move(value)) {}
  at::Tensor tensor;
};

Tensor::Tensor(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
  const auto& tensor = this->impl().tensor;
  dtype_ = dtypeFromTorch(tensor.scalar_type());
  device_ = deviceFromTorch(tensor);
  device_id_ = tensor.device().is_cuda() ? static_cast<int>(tensor.device().index()) : 0;
  shape_ = shapeFromTorch(tensor);
  if (shape_.empty()) shape_.push_back(1);
}

Tensor::Tensor(at::Tensor tensor) : Tensor(std::make_shared<Impl>(tensor.contiguous())) {}

const Tensor::Impl& Tensor::impl() const {
  if (!impl_) throw std::logic_error("tensor is uninitialized");
  return *impl_;
}

Tensor::Impl& Tensor::impl() {
  if (!impl_) throw std::logic_error("tensor is uninitialized");
  return *impl_;
}

std::string toString(DeviceType device) {
  switch (device) {
    case DeviceType::Cpu:
      return "cpu";
    case DeviceType::Cuda:
      return "cuda";
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
    case DType::Float32:
      return "float32";
    case DType::Int64:
      return "int64";
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
    case DType::Float32:
      return sizeof(float);
    case DType::Int64:
      return sizeof(std::int64_t);
  }
  throw std::invalid_argument("unknown dtype");
}

Tensor::Tensor(DType dtype, std::vector<std::int64_t> shape) {
  if (shape.empty()) shape.push_back(1);
  for (const auto dim : shape) {
    if (dim < 0) throw std::invalid_argument("negative tensor dimension");
  }
  *this = Tensor(std::make_shared<Impl>(at::zeros(shape, optionsFor(dtype, DeviceType::Cpu))));
}

Tensor Tensor::zeros(DType dtype, std::vector<std::int64_t> shape) {
  return Tensor(dtype, std::move(shape));
}

Tensor Tensor::cuda(DType dtype, std::vector<std::int64_t> shape, int device_id) {
  if (shape.empty()) shape.push_back(1);
  for (const auto dim : shape) {
    if (dim < 0) throw std::invalid_argument("negative tensor dimension");
  }
  return Tensor(
      std::make_shared<Impl>(at::empty(shape, optionsFor(dtype, DeviceType::Cuda, device_id))));
}

Tensor Tensor::externalCuda(DType dtype, std::vector<std::int64_t> shape, void* data,
                            int device_id) {
  if (data == nullptr) throw std::invalid_argument("external CUDA tensor pointer is null");
  if (shape.empty()) shape.push_back(1);
  Tensor tensor;
  tensor.dtype_ = dtype;
  tensor.device_ = DeviceType::Cuda;
  tensor.device_id_ = device_id;
  tensor.shape_ = std::move(shape);
  tensor.external_device_ptr_ = data;
  return tensor;
}

std::int64_t Tensor::dim(std::size_t index) const {
  if (index >= shape_.size()) throw std::out_of_range("tensor dimension out of range");
  return shape_[index];
}

std::size_t Tensor::numel() const {
  if (!impl_) {
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
  const auto elements = impl().tensor.numel();
  if (elements < 0) throw std::overflow_error("negative tensor numel");
  return static_cast<std::size_t>(elements);
}

std::size_t Tensor::nbytes() const { return numel() * byteSize(dtype_); }

void* Tensor::deviceData() {
  if (device_ != DeviceType::Cuda) throw std::logic_error("tensor is not on CUDA");
  return impl_ ? impl().tensor.data_ptr() : external_device_ptr_;
}

const void* Tensor::deviceData() const {
  if (device_ != DeviceType::Cuda) throw std::logic_error("tensor is not on CUDA");
  return impl_ ? impl().tensor.data_ptr() : external_device_ptr_;
}

Tensor Tensor::withShape(std::vector<std::int64_t> shape) const {
  if (shape.empty()) shape.push_back(1);
  std::size_t requested = 1;
  for (const auto dim : shape) {
    if (dim < 0) throw std::invalid_argument("negative tensor dimension");
    requested *= static_cast<std::size_t>(dim);
  }
  if (requested != numel()) throw std::invalid_argument("tensor view changes element count");
  if (!impl_) {
    Tensor view = *this;
    view.shape_ = std::move(shape);
    return view;
  }
  return Tensor(std::make_shared<Impl>(impl().tensor.view(shape)));
}

at::Tensor& Tensor::torchTensor() { return impl().tensor; }

const at::Tensor& Tensor::torchTensor() const { return impl().tensor; }

template <>
float* Tensor::data<float>() {
  checkHostData(*this, DType::Float32);
  return impl().tensor.data_ptr<float>();
}

template <>
const float* Tensor::data<float>() const {
  checkHostData(*this, DType::Float32);
  return impl().tensor.data_ptr<float>();
}

template <>
std::int64_t* Tensor::data<std::int64_t>() {
  checkHostData(*this, DType::Int64);
  return impl().tensor.data_ptr<std::int64_t>();
}

template <>
const std::int64_t* Tensor::data<std::int64_t>() const {
  checkHostData(*this, DType::Int64);
  return impl().tensor.data_ptr<std::int64_t>();
}

bool sameShape(const Tensor& lhs, const Tensor& rhs) { return lhs.shape() == rhs.shape(); }

std::size_t contiguousOffset(const std::vector<std::int64_t>& shape,
                             const std::vector<std::int64_t>& indices) {
  if (shape.size() != indices.size())
    throw std::invalid_argument("rank mismatch in contiguousOffset");
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (std::size_t reverse = 0; reverse < shape.size(); ++reverse) {
    const auto i = shape.size() - 1 - reverse;
    if (indices[i] < 0 || indices[i] >= shape[i])
      throw std::out_of_range("tensor index out of range");
    offset += static_cast<std::size_t>(indices[i]) * stride;
    stride *= static_cast<std::size_t>(shape[i]);
  }
  return offset;
}

}  // namespace dli
