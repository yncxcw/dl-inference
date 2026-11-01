/**
 * @file utils.h
 * @brief Shape, attribute and validation helpers shared by operator implementations.
 */

#ifndef DLI_UTILS_H_
#define DLI_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dli/attributes.h"
#include "dli/tensor.h"

namespace dli {

/**
 * @brief Multiplies a range of dimensions.
 * @param shape Dimensions.
 * @param begin First dimension index to include.
 * @param end One past the last index; the default (-1 as size_t) means shape.size().
 * @return The product, or 1 for an empty range.
 */
std::int64_t product(const std::vector<std::int64_t>& shape, std::size_t begin = 0,
                     std::size_t end = static_cast<std::size_t>(-1));

/**
 * @brief Integer division rounding up.
 * @param a Dividend.
 * @param b Divisor; must be positive.
 * @return ceil(a / b), converted to unsigned (for kernel grid sizes).
 */
unsigned ceilDiv(std::int64_t a, std::int64_t b);

/**
 * @brief Formats a shape for messages.
 * @param shape Dimensions.
 * @return Text such as "[2,3,4]".
 */
std::string formatShape(const std::vector<std::int64_t>& shape);

/**
 * @brief Checks that every input tensor is on CUDA.
 * @param inputs Non-null tensor pointers.
 * @param op Operator name for the error message.
 * @throws std::invalid_argument If any tensor is not on CUDA.
 */
void requireCudaInputs(const std::vector<const Tensor*>& inputs, const std::string& op);

/**
 * @brief Checks that a tensor is float32.
 * @param tensor Tensor to check.
 * @param op Operator name for the error message.
 * @throws std::invalid_argument If the dtype is not float32.
 */
void requireFloat(const Tensor& tensor, const std::string& op);

/**
 * @brief Reads a numeric attribute as double.
 * @param attrs Attribute bag.
 * @param name Attribute name.
 * @param fallback Value returned when the attribute is absent.
 * @return The attribute value; integer attributes are converted.
 * @throws std::invalid_argument If the attribute is neither an integer nor a double.
 */
double attrDouble(const Attributes& attrs, const std::string& name, double fallback);

/**
 * @brief Reads an integer-list attribute.
 * @param attrs Attribute bag.
 * @param name Attribute name.
 * @param fallback Value returned when the attribute is absent.
 * @return The attribute value or @p fallback.
 * @throws std::invalid_argument If the attribute is not an integer list.
 */
std::vector<std::int64_t> attrInts(const Attributes& attrs, const std::string& name,
                                   std::vector<std::int64_t> fallback);

/**
 * @brief Returns a mutable device pointer for a const CUDA tensor (for kernel argument arrays).
 * @param tensor CUDA tensor.
 * @return Device pointer.
 * @throws std::logic_error If the tensor is not on CUDA.
 */
void* ptr(const Tensor& tensor);

/**
 * @brief Resolves a reshape target that may contain one inferred (-1) dimension.
 * @param input Current dimensions.
 * @param target Requested dimensions; entries are positive or a single -1.
 * @return Fully resolved dimensions with the same element count as @p input.
 * @throws std::invalid_argument If there are several -1 entries, a dimension is not positive or
 *     -1, the inferred size is not integral, or the element count would change.
 */
std::vector<std::int64_t> reshapeShape(const std::vector<std::int64_t>& input,
                                       std::vector<std::int64_t> target);

}  // namespace dli

#endif  // DLI_UTILS_H_
