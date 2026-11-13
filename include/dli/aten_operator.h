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
 * The registered operator reads the ATen schema name from the "name" attribute and an optional
 * overload from "overload". If the node has an "arguments" attribute, each entry encodes one
 * schema argument, in order: "t:<i>" is node input i, "n" is None, and the other kinds hold
 * scalars, lists, enums or a device, as aten_operator.cc documents. Bools must be true, false, 1
 * or 0. Without "arguments", the operator passes every node input, then the attributes named in
 * "attr_order". Schemas that write to an argument are rejected.
 *
 * @param registry Registry that receives the "aten" factory; an existing "aten" entry is replaced.
 */
void registerAtenOperator(OperatorRegistry& registry);

}  // namespace dli

#endif  // DLI_ATEN_OPERATOR_H_
