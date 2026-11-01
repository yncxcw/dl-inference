/**
 * @file aten_operator.h
 * @brief Registration entry point for the generic ATen dispatcher operator.
 */

#ifndef DLI_ATEN_OPERATOR_H_
#define DLI_ATEN_OPERATOR_H_

#include "dli/operator.h"

namespace dli {

/**
 * @brief Registers the built-in "aten" operator that forwards to PyTorch's dispatcher.
 *
 * The registered operator reads the ATen schema name from the "name" attribute, an optional
 * overload from "overload", and the ordered scalar attributes listed in "attr_order".
 *
 * @param registry Registry that receives the "aten" factory; an existing "aten" entry is replaced.
 */
void registerAtenOperator(OperatorRegistry& registry);

}  // namespace dli

#endif  // DLI_ATEN_OPERATOR_H_
