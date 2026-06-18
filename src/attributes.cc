#include "dli/attributes.h"

namespace dli {

void Attributes::set(std::string name, AttributeValue value) {
  values_[std::move(name)] = std::move(value);
}

bool Attributes::contains(const std::string& name) const {
  return values_.find(name) != values_.end();
}

}  // namespace dli
