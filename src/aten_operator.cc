#include "dli/aten_operator.h"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace dli {
namespace {

c10::IValue attributeToIValue(const AttributeValue& value) {
  return std::visit(
      [](const auto& item) -> c10::IValue {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::int64_t>) {
          return c10::IValue(item);
        } else if constexpr (std::is_same_v<T, double>) {
          return c10::IValue(item);
        } else if constexpr (std::is_same_v<T, bool>) {
          return c10::IValue(item);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return c10::IValue(item);
        } else {
          return c10::IValue(item);
        }
      },
      value);
}

c10::IValue attributeToIValue(const Attributes& attrs, const std::string& name) {
  const auto it = attrs.values().find(name);
  if (it == attrs.values().end()) {
    throw std::invalid_argument("aten operator missing ordered attr: " + name);
  }
  return attributeToIValue(it->second);
}

std::vector<at::Tensor> tensorsFromIValue(const c10::IValue& value) {
  if (value.isTensor()) return {value.toTensor()};
  if (value.isTensorList()) return value.toTensorVector();
  if (value.isTuple()) {
    std::vector<at::Tensor> tensors;
    for (const auto& item : value.toTupleRef().elements()) {
      if (!item.isTensor()) throw std::invalid_argument("aten tuple output contains non-tensor value");
      tensors.push_back(item.toTensor());
    }
    return tensors;
  }
  throw std::invalid_argument("aten operator output is not a tensor, tensor list, or tensor tuple");
}

class AtenOperator final : public Operator {
 public:
  std::string type() const override { return "aten"; }

  void compute(const std::vector<const Tensor*>& inputs,
               const std::vector<Tensor*>& outputs,
               const Attributes& attrs,
               ExecutionContext&) const override {
    // The graph stores the ATen operator name and optional overload separately,
    // for example name="aten::add" and overload="Tensor".
    const auto name = attrs.require<std::string>("name");
    const auto overload = attrs.value_or<std::string>("overload", "");
    const auto attr_order = attrs.value_or<std::vector<std::string>>("attr_order", {});

    // Box tensor inputs and ordered scalar/list attributes into the c10 stack in
    // the same order expected by the selected ATen schema.
    c10::Stack stack;
    stack.reserve(inputs.size() + attr_order.size());
    for (const auto* input : inputs) {
      if (input == nullptr) throw std::invalid_argument("aten operator received null input tensor");
      stack.emplace_back(input->torchTensor());
    }
    for (const auto& attr_name : attr_order) {
      if (attr_name == "name" || attr_name == "overload" || attr_name == "attr_order") {
        throw std::invalid_argument("aten attr_order contains reserved attribute: " + attr_name);
      }
      stack.emplace_back(attributeToIValue(attrs, attr_name));
    }

    // Resolve the schema through PyTorch's dispatcher and invoke it with boxed
    // arguments. callBoxed replaces the input stack contents with boxed returns.
    const auto handle =
        c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str());
    handle.callBoxed(&stack);
    if (stack.empty()) throw std::runtime_error("aten operator produced no boxed return value");

    // Normalize ATen's possible tensor return shapes into a flat tensor vector
    // and move each returned tensor into the caller-provided DLI output slots.
    auto returned = tensorsFromIValue(stack.back());
    if (returned.size() != outputs.size()) {
      throw std::invalid_argument("aten operator output count mismatch");
    }
    for (std::size_t i = 0; i < returned.size(); ++i) {
      if (outputs[i] == nullptr) throw std::invalid_argument("aten operator received null output tensor");
      *outputs[i] = Tensor(std::move(returned[i]));
    }
  }
};

}  // namespace

void registerAtenOperator(OperatorRegistry& registry) {
  registry.registerFactory("aten", [] { return std::make_unique<AtenOperator>(); });
}

}  // namespace dli
