#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>

namespace dli {

enum class DeviceType { Cpu, Cuda };
enum class DType { Float32, Int64 };

std::string toString(DeviceType device);
DeviceType deviceFromString(const std::string& device);
std::string toString(DType dtype);
DType dtypeFromString(const std::string& dtype);
std::size_t byteSize(DType dtype);

class Tensor {
 public:
  Tensor() = default;
  Tensor(DType dtype, std::vector<std::int64_t> shape);
  explicit Tensor(at::Tensor tensor);

  static Tensor zeros(DType dtype, std::vector<std::int64_t> shape);
  static Tensor cuda(DType dtype, std::vector<std::int64_t> shape, int device_id = 0);
  static Tensor externalCuda(DType dtype, std::vector<std::int64_t> shape, void* data,
                             int device_id = 0);

  DType dtype() const { return dtype_; }
  DeviceType device() const { return device_; }
  int deviceId() const { return device_id_; }
  bool isCpu() const { return device_ == DeviceType::Cpu; }
  bool isCuda() const { return device_ == DeviceType::Cuda; }
  const std::vector<std::int64_t>& shape() const { return shape_; }
  std::size_t rank() const { return shape_.size(); }
  std::int64_t dim(std::size_t index) const;
  std::size_t numel() const;
  std::size_t nbytes() const;
  void* deviceData();
  const void* deviceData() const;
  Tensor withShape(std::vector<std::int64_t> shape) const;
  at::Tensor& torchTensor();
  const at::Tensor& torchTensor() const;

  template <typename T>
  T* data();

  template <typename T>
  const T* data() const;

 private:
  struct Impl;

  explicit Tensor(std::shared_ptr<Impl> impl);
  const Impl& impl() const;
  Impl& impl();

  DType dtype_ = DType::Float32;
  DeviceType device_ = DeviceType::Cpu;
  int device_id_ = 0;
  std::vector<std::int64_t> shape_;
  std::shared_ptr<Impl> impl_;
  void* external_device_ptr_ = nullptr;
};

bool sameShape(const Tensor& lhs, const Tensor& rhs);
std::size_t contiguousOffset(const std::vector<std::int64_t>& shape,
                             const std::vector<std::int64_t>& indices);

}  // namespace dli
