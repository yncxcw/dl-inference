/**
 * @file tensor.h
 * @brief Tensor type backed by ATen, plus dtype/device enums and shape helpers.
 */

#ifndef DLI_TENSOR_H_
#define DLI_TENSOR_H_

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dli {

/// Where a tensor's storage lives.
enum class DeviceType { Cpu, Cuda };
/// Supported element types.
enum class DType { Float32, Int64 };

/**
 * @brief Returns the canonical name of a device type.
 * @param device Device type.
 * @return "cpu" or "cuda".
 * @throws std::invalid_argument If @p device is not a known value.
 */
std::string toString(DeviceType device);

/**
 * @brief Parses a device name.
 * @param device "cpu", "cuda" or "gpu".
 * @return The matching device type.
 * @throws std::invalid_argument If the name is not recognized.
 */
DeviceType deviceFromString(const std::string& device);

/**
 * @brief Returns the canonical name of an element type.
 * @param dtype Element type.
 * @return "float32" or "int64".
 * @throws std::invalid_argument If @p dtype is not a known value.
 */
std::string toString(DType dtype);

/**
 * @brief Parses an element type name.
 * @param dtype "float32"/"f32" or "int64"/"i64".
 * @return The matching element type.
 * @throws std::invalid_argument If the name is not recognized.
 */
DType dtypeFromString(const std::string& dtype);

/**
 * @brief Returns the size of one element.
 * @param dtype Element type.
 * @return Size in bytes.
 * @throws std::invalid_argument If @p dtype is not a known value.
 */
std::size_t byteSize(DType dtype);

/**
 * @brief Dense contiguous tensor with cheap shared-ownership copies.
 *
 * A tensor either wraps an at::Tensor (CPU or CUDA) or, via externalCuda(), a non-owning view of
 * caller-managed device memory. Copies share the same storage. Rank-0 shapes are normalized to
 * {1}.
 */
class Tensor {
 public:
  /// Creates an empty, uninitialized tensor; most accessors throw until it is assigned.
  Tensor() = default;

  /**
   * @brief Creates a zero-filled CPU tensor.
   * @param dtype Element type.
   * @param shape Dimensions; an empty shape becomes {1}.
   * @throws std::invalid_argument If any dimension is negative.
   */
  Tensor(DType dtype, std::vector<std::int64_t> shape);

  /**
   * @brief Wraps an existing ATen tensor, making it contiguous if necessary.
   * @param tensor Source tensor; must be float32 or int64 on CPU or CUDA.
   * @throws std::invalid_argument If the dtype or device is unsupported.
   */
  explicit Tensor(at::Tensor tensor);

  /**
   * @brief Creates a zero-filled CPU tensor; same as the (dtype, shape) constructor.
   * @param dtype Element type.
   * @param shape Dimensions.
   * @return The new tensor.
   * @throws std::invalid_argument If any dimension is negative.
   */
  static Tensor zeros(DType dtype, std::vector<std::int64_t> shape);

  /**
   * @brief Allocates an uninitialized CUDA tensor.
   * @param dtype Element type.
   * @param shape Dimensions; an empty shape becomes {1}.
   * @param device_id CUDA device ordinal.
   * @return The new tensor.
   * @throws std::invalid_argument If any dimension is negative.
   */
  static Tensor cuda(DType dtype, std::vector<std::int64_t> shape, int device_id = 0);

  /**
   * @brief Wraps caller-owned device memory without taking ownership.
   * @param dtype Element type.
   * @param shape Dimensions; an empty shape becomes {1}.
   * @param data Device pointer; must stay valid while the tensor or its views are used.
   * @param device_id CUDA device ordinal that owns @p data.
   * @return A tensor that has no torchTensor().
   * @throws std::invalid_argument If @p data is null.
   */
  static Tensor externalCuda(DType dtype, std::vector<std::int64_t> shape, void* data,
                             int device_id = 0);

  /// @return Element type.
  DType dtype() const { return dtype_; }
  /// @return Device type.
  DeviceType device() const { return device_; }
  /// @return CUDA device ordinal (0 for CPU tensors).
  int deviceId() const { return device_id_; }
  /// @return True if the tensor is on the CPU.
  bool isCpu() const { return device_ == DeviceType::Cpu; }
  /// @return True if the tensor is on a CUDA device.
  bool isCuda() const { return device_ == DeviceType::Cuda; }
  /// @return Dimensions of the tensor.
  const std::vector<std::int64_t>& shape() const { return shape_; }
  /// @return Number of dimensions.
  std::size_t rank() const { return shape_.size(); }

  /**
   * @brief Returns one dimension.
   * @param index Zero-based dimension index.
   * @return Size of that dimension.
   * @throws std::out_of_range If @p index >= rank().
   */
  std::int64_t dim(std::size_t index) const;

  /**
   * @brief Returns the element count.
   * @return Product of all dimensions.
   * @throws std::overflow_error If the count does not fit in size_t.
   */
  std::size_t numel() const;

  /// @return Storage size in bytes (numel() times the element size).
  std::size_t nbytes() const;

  /**
   * @brief Returns the raw device pointer.
   * @return Pointer to CUDA memory.
   * @throws std::logic_error If the tensor is not on CUDA.
   */
  void* deviceData();
  /// @copydoc deviceData()
  const void* deviceData() const;

  /**
   * @brief Returns a view with a different shape over the same storage.
   * @param shape New dimensions; an empty shape becomes {1}.
   * @return The reshaped view.
   * @throws std::invalid_argument If a dimension is negative or the element count changes.
   */
  Tensor withShape(std::vector<std::int64_t> shape) const;

  /**
   * @brief Returns the underlying ATen tensor.
   * @return Reference valid as long as this tensor's storage is alive.
   * @throws std::logic_error If the tensor is uninitialized or an external CUDA view.
   */
  at::Tensor& torchTensor();
  /// @copydoc torchTensor()
  const at::Tensor& torchTensor() const;

  /**
   * @brief Returns a typed host pointer to the elements.
   * @tparam T float or std::int64_t, matching dtype().
   * @return Pointer to contiguous CPU data.
   * @throws std::logic_error If the tensor is not on the CPU or @p T does not match dtype().
   */
  template <typename T>
  T* data();

  /// @copydoc data()
  template <typename T>
  const T* data() const;

 private:
  /// Opaque holder of the at::Tensor.
  struct Impl;

  /// Adopts an Impl and caches its dtype, device and shape.
  explicit Tensor(std::shared_ptr<Impl> impl);
  /// @return The Impl; throws std::logic_error if uninitialized.
  const Impl& impl() const;
  /// @copydoc impl() const
  Impl& impl();

  /// Element type.
  DType dtype_ = DType::Float32;
  /// Device type.
  DeviceType device_ = DeviceType::Cpu;
  /// CUDA device ordinal.
  int device_id_ = 0;
  /// Cached dimensions.
  std::vector<std::int64_t> shape_;
  /// Shared storage; null for default-constructed and external CUDA tensors.
  std::shared_ptr<Impl> impl_;
  /// Non-owning device pointer used when impl_ is null (externalCuda()).
  void* external_device_ptr_ = nullptr;
};

/**
 * @brief Compares the shapes of two tensors.
 * @param lhs First tensor.
 * @param rhs Second tensor.
 * @return True if both have identical dimensions.
 */
bool sameShape(const Tensor& lhs, const Tensor& rhs);

/**
 * @brief Computes the row-major linear offset of a multi-dimensional index.
 * @param shape Tensor dimensions.
 * @param indices Index along each dimension.
 * @return Element offset from the start of the tensor.
 * @throws std::invalid_argument If the ranks differ.
 * @throws std::out_of_range If any index is outside its dimension.
 */
std::size_t contiguousOffset(const std::vector<std::int64_t>& shape,
                             const std::vector<std::int64_t>& indices);

}  // namespace dli

#endif  // DLI_TENSOR_H_
