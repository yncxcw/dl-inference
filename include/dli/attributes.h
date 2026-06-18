#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace dli {

using AttributeValue = std::variant<std::int64_t, double, bool, std::string,
                                    std::vector<std::int64_t>, std::vector<double>,
                                    std::vector<std::string>>;

class Attributes {
 public:
  void set(std::string name, AttributeValue value);
  bool contains(const std::string& name) const;

  template <typename T>
  const T& require(const std::string& name) const {
    const auto it = values_.find(name);
    if (it == values_.end()) throw std::invalid_argument("missing required attribute: " + name);
    if (!std::holds_alternative<T>(it->second)) {
      throw std::invalid_argument("attribute has unexpected type: " + name);
    }
    return std::get<T>(it->second);
  }

  template <typename T>
  T value_or(const std::string& name, T fallback) const {
    const auto it = values_.find(name);
    if (it == values_.end()) return fallback;
    if (!std::holds_alternative<T>(it->second)) {
      throw std::invalid_argument("attribute has unexpected type: " + name);
    }
    return std::get<T>(it->second);
  }

  const std::map<std::string, AttributeValue>& values() const { return values_; }

 private:
  std::map<std::string, AttributeValue> values_;
};

}  // namespace dli
