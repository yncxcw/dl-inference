#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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

  static Tensor zeros(DType dtype, std::vector<std::int64_t> shape);
  static Tensor cuda(DType dtype, std::vector<std::int64_t> shape, int device_id = 0);
  static Tensor externalCuda(DType dtype, std::vector<std::int64_t> shape, void* data,
                             int device_id = 0);
  static Tensor fromFloat32(std::vector<std::int64_t> shape, std::vector<float> values);
  static Tensor fromInt64(std::vector<std::int64_t> shape, std::vector<std::int64_t> values);

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

  template <typename T>
  T* data() {
    checkType<T>();
    return reinterpret_cast<T*>(storage_.data());
  }

  template <typename T>
  const T* data() const {
    checkType<T>();
    return reinterpret_cast<const T*>(storage_.data());
  }

  template <typename T>
  std::span<T> span() {
    return {data<T>(), numel()};
  }

  template <typename T>
  std::span<const T> span() const {
    return {data<T>(), numel()};
  }

  std::vector<std::byte>& rawStorage() { return storage_; }
  const std::vector<std::byte>& rawStorage() const { return storage_; }

 private:
  struct DeviceAllocation;

  template <typename T>
  void checkType() const {
    if (device_ != DeviceType::Cpu) throw std::logic_error("host data access requested for non-CPU tensor");
    if constexpr (std::is_same_v<T, float>) {
      if (dtype_ != DType::Float32) throw std::logic_error("tensor dtype is not float32");
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
      if (dtype_ != DType::Int64) throw std::logic_error("tensor dtype is not int64");
    } else {
      static_assert(!sizeof(T), "unsupported tensor element type");
    }
  }

  DType dtype_ = DType::Float32;
  DeviceType device_ = DeviceType::Cpu;
  int device_id_ = 0;
  std::vector<std::int64_t> shape_;
  std::vector<std::byte> storage_;
  std::shared_ptr<DeviceAllocation> device_allocation_;
  void* external_device_ptr_ = nullptr;
};

bool sameShape(const Tensor& lhs, const Tensor& rhs);
std::size_t contiguousOffset(const std::vector<std::int64_t>& shape,
                             const std::vector<std::int64_t>& indices);

}  // namespace dli
