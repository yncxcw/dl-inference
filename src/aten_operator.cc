#include "dli/aten_operator.h"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/core/MemoryFormat.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

void appendTensorsFromIValue(const c10::IValue& value, std::vector<at::Tensor>& tensors) {
  if (value.isNone()) return;
  if (value.isTensor()) {
    tensors.push_back(value.toTensor());
    return;
  }
  if (value.isTensorList()) {
    const auto values = value.toTensorVector();
    tensors.insert(tensors.end(), values.begin(), values.end());
    return;
  }
  if (value.isTuple()) {
    for (const auto& item : value.toTupleRef().elements()) {
      appendTensorsFromIValue(item, tensors);
    }
    return;
  }
  if (value.isList()) {
    for (const auto& item : value.toListRef()) {
      appendTensorsFromIValue(item, tensors);
    }
    return;
  }
  throw std::invalid_argument("aten operator output contains an unsupported non-tensor value");
}

std::vector<at::Tensor> tensorsFromIValue(const c10::IValue& value) {
  std::vector<at::Tensor> tensors;
  appendTensorsFromIValue(value, tensors);
  return tensors;
}

std::vector<std::string> split(std::string_view value, char delimiter) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start <= value.size()) {
    const auto end = value.find(delimiter, start);
    result.emplace_back(
        value.substr(start, end == std::string_view::npos ? value.size() - start : end - start));
    if (end == std::string_view::npos) break;
    start = end + 1;
  }
  return result;
}

const at::Tensor& tensorArgument(const std::vector<const Tensor*>& inputs,
                                 const std::string& index_text) {
  const auto index = static_cast<std::size_t>(std::stoull(index_text));
  if (index >= inputs.size() || inputs[index] == nullptr)
    throw std::invalid_argument("aten argument references missing tensor input");
  return inputs[index]->torchTensor();
}

at::ScalarType scalarTypeFromName(const std::string& name) {
  if (name == "bool") return at::kBool;
  if (name == "uint8") return at::kByte;
  if (name == "int8") return at::kChar;
  if (name == "int16") return at::kShort;
  if (name == "int32") return at::kInt;
  if (name == "int64") return at::kLong;
  if (name == "float16") return at::kHalf;
  if (name == "bfloat16") return at::kBFloat16;
  if (name == "float32") return at::kFloat;
  if (name == "float64") return at::kDouble;
  throw std::invalid_argument("unsupported aten dtype argument: " + name);
}

c10::IValue decodeArgument(const std::string& spec, const std::vector<const Tensor*>& inputs) {
  if (spec == "n") return c10::IValue();
  const auto separator = spec.find(':');
  const auto kind = spec.substr(0, separator);
  const auto value = separator == std::string::npos ? std::string{} : spec.substr(separator + 1);
  if (kind == "t") return c10::IValue(tensorArgument(inputs, value));
  if (kind == "i") return c10::IValue(static_cast<std::int64_t>(std::stoll(value)));
  if (kind == "f") return c10::IValue(std::stod(value));
  if (kind == "b") return c10::IValue(value == "1" || value == "true");
  if (kind == "s") return c10::IValue(value);
  if (kind == "il") {
    std::vector<std::int64_t> items;
    if (!value.empty()) {
      for (const auto& item : split(value, ',')) items.push_back(std::stoll(item));
    }
    return c10::IValue(std::move(items));
  }
  if (kind == "fl") {
    std::vector<double> items;
    if (!value.empty()) {
      for (const auto& item : split(value, ',')) items.push_back(std::stod(item));
    }
    return c10::IValue(std::move(items));
  }
  if (kind == "bl") {
    c10::List<bool> items;
    if (!value.empty()) {
      for (const auto& item : split(value, ',')) items.push_back(item == "1" || item == "true");
    }
    return c10::IValue(std::move(items));
  }
  if (kind == "tl") {
    c10::List<at::Tensor> items;
    if (!value.empty()) {
      for (const auto& item : split(value, ',')) items.push_back(tensorArgument(inputs, item));
    }
    return c10::IValue(std::move(items));
  }
  if (kind == "otl") {
    c10::List<std::optional<at::Tensor>> items;
    if (!value.empty()) {
      for (const auto& item : split(value, ',')) {
        if (item == "n")
          items.push_back(std::nullopt);
        else
          items.push_back(tensorArgument(inputs, item));
      }
    }
    return c10::IValue(std::move(items));
  }
  if (kind == "dtype") {
    return c10::IValue(static_cast<std::int64_t>(scalarTypeFromName(value)));
  }
  if (kind == "layout") {
    if (value != "strided") throw std::invalid_argument("unsupported aten layout: " + value);
    return c10::IValue(static_cast<std::int64_t>(at::kStrided));
  }
  if (kind == "memory_format") {
    c10::MemoryFormat format;
    if (value == "contiguous")
      format = c10::MemoryFormat::Contiguous;
    else if (value == "preserve")
      format = c10::MemoryFormat::Preserve;
    else if (value == "channels_last")
      format = c10::MemoryFormat::ChannelsLast;
    else if (value == "channels_last_3d")
      format = c10::MemoryFormat::ChannelsLast3d;
    else
      throw std::invalid_argument("unsupported aten memory format: " + value);
    return c10::IValue(static_cast<std::int64_t>(format));
  }
  if (kind == "device") {
    const auto& anchor = tensorArgument(inputs, value);
    return c10::IValue(anchor.device());
  }
  throw std::invalid_argument("unsupported encoded aten argument: " + spec);
}

class AtenOperator final : public Operator {
 public:
  std::string type() const override { return "aten"; }

  void compute(const std::vector<const Tensor*>& inputs, const std::vector<Tensor*>& outputs,
               const Attributes& attrs, ExecutionContext&) const override {
    // The graph stores the ATen operator name and optional overload separately,
    // for example name="aten::add" and overload="Tensor".
    const auto name = attrs.require<std::string>("name");
    const auto overload = attrs.value_or<std::string>("overload", "");
    const auto arguments = attrs.value_or<std::vector<std::string>>("arguments", {});
    const auto attr_order = attrs.value_or<std::vector<std::string>>("attr_order", {});

    // Box tensor inputs and ordered scalar/list attributes into the c10 stack in
    // the same order expected by the selected ATen schema.
    c10::Stack stack;
    if (!arguments.empty()) {
      stack.reserve(arguments.size());
      for (const auto& argument : arguments) stack.emplace_back(decodeArgument(argument, inputs));
    } else {
      stack.reserve(inputs.size() + attr_order.size());
      for (const auto* input : inputs) {
        if (input == nullptr)
          throw std::invalid_argument("aten operator received null input tensor");
        stack.emplace_back(input->torchTensor());
      }
      for (const auto& attr_name : attr_order) {
        if (attr_name == "name" || attr_name == "overload" || attr_name == "attr_order" ||
            attr_name == "arguments") {
          throw std::invalid_argument("aten attr_order contains reserved attribute: " + attr_name);
        }
        stack.emplace_back(attributeToIValue(attrs, attr_name));
      }
    }

    // Resolve the schema through PyTorch's dispatcher and invoke it with boxed
    // arguments. callBoxed replaces the input stack contents with boxed returns.
    const auto handle =
        c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str());
    for (const auto& argument : handle.schema().arguments()) {
      const auto* alias_info = argument.alias_info();
      if (alias_info != nullptr && alias_info->isWrite()) {
        throw std::invalid_argument(
            "mutating ATen schemas are not supported; export a functional graph: " + name);
      }
    }
    handle.callBoxed(&stack);
    if (stack.empty()) throw std::runtime_error("aten operator produced no boxed return value");

    // Boxed calls place one IValue on the stack per schema return. A return may
    // itself be a tensor list or tuple, so flatten both levels before assigning
    // the DLI output edges.
    std::vector<at::Tensor> returned;
    for (const auto& value : stack) {
      auto tensors = tensorsFromIValue(value);
      returned.insert(returned.end(), std::make_move_iterator(tensors.begin()),
                      std::make_move_iterator(tensors.end()));
    }
    if (returned.size() != outputs.size()) {
      throw std::invalid_argument("aten operator output count mismatch");
    }
    for (std::size_t i = 0; i < returned.size(); ++i) {
      if (outputs[i] == nullptr)
        throw std::invalid_argument("aten operator received null output tensor");
      *outputs[i] = Tensor(std::move(returned[i]));
    }
  }
};

}  // namespace

void registerAtenOperator(OperatorRegistry& registry) {
  registry.registerFactory("aten", [] { return std::make_unique<AtenOperator>(); });
}

}  // namespace dli
