/**
 * @file attributes.h
 * @brief Typed, string-keyed attribute bag attached to graph nodes.
 */

#ifndef DLI_ATTRIBUTES_H_
#define DLI_ATTRIBUTES_H_

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace dli {

/// Value types an operator attribute may hold.
using AttributeValue =
    std::variant<std::int64_t, double, bool, std::string, std::vector<std::int64_t>,
                 std::vector<double>, std::vector<std::string>>;

/**
 * @brief Ordered map from attribute name to AttributeValue.
 *
 * Operators read their configuration (axes, epsilon, flags, ...) from this container.
 */
class Attributes {
 public:
  /**
   * @brief Stores an attribute, replacing any existing value with the same name.
   * @param name Attribute name.
   * @param value Attribute value.
   */
  void set(std::string name, AttributeValue value);

  /**
   * @brief Tests whether an attribute exists.
   * @param name Attribute name.
   * @return True if @p name is present.
   */
  bool contains(const std::string& name) const;

  /**
   * @brief Returns a required attribute of a specific type.
   * @tparam T Expected alternative of AttributeValue.
   * @param name Attribute name.
   * @return Reference to the stored value; valid until the attribute is replaced.
   * @throws std::invalid_argument If the attribute is missing or does not hold a @p T.
   */
  template <typename T>
  const T& require(const std::string& name) const {
    const auto it = values_.find(name);
    if (it == values_.end()) throw std::invalid_argument("missing required attribute: " + name);
    if (!std::holds_alternative<T>(it->second)) {
      throw std::invalid_argument("attribute has unexpected type: " + name);
    }
    return std::get<T>(it->second);
  }

  /**
   * @brief Returns an optional attribute of a specific type, or a fallback when absent.
   * @tparam T Expected alternative of AttributeValue.
   * @param name Attribute name.
   * @param fallback Value returned when the attribute is missing.
   * @return The stored value, or @p fallback.
   * @throws std::invalid_argument If the attribute exists but does not hold a @p T.
   */
  template <typename T>
  T value_or(const std::string& name, T fallback) const {
    const auto it = values_.find(name);
    if (it == values_.end()) return fallback;
    if (!std::holds_alternative<T>(it->second)) {
      throw std::invalid_argument("attribute has unexpected type: " + name);
    }
    return std::get<T>(it->second);
  }

  /// @return Read-only view of all attributes, sorted by name.
  const std::map<std::string, AttributeValue>& values() const { return values_; }

 private:
  /// Attribute storage keyed by name.
  std::map<std::string, AttributeValue> values_;
};

}  // namespace dli

#endif  // DLI_ATTRIBUTES_H_
