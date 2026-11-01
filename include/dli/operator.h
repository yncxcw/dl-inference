/**
 * @file operator.h
 * @brief Operator interface and the registry that creates operators by type name.
 */

#ifndef DLI_OPERATOR_H_
#define DLI_OPERATOR_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dli/attributes.h"
#include "dli/kv_cache.h"
#include "dli/tensor.h"

namespace dli {

/// Per-run state shared with every operator.
struct ExecutionContext {
  /// KV cache owned by the Engine; may be null when no cache is available.
  KVCache* kv_cache = nullptr;
};

/// Base class of all graph operators.
class Operator {
 public:
  virtual ~Operator() = default;

  /// @return The registered type name of this operator.
  virtual std::string type() const = 0;

  /**
   * @brief Computes the operator.
   * @param inputs Non-owning input tensors, in node input order.
   * @param outputs Non-owning output slots, in node output order; the operator assigns each one.
   * @param attrs Node attributes.
   * @param context Execution state such as the KV cache.
   * @throws std::invalid_argument On wrong arity, tensor types or attributes.
   * @throws std::runtime_error On backend failures.
   */
  virtual void compute(const std::vector<const Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs, const Attributes& attrs,
                       ExecutionContext& context) const = 0;
};

/// Maps operator type names to factories; can load additional operators from shared libraries.
class OperatorRegistry {
 public:
  /// Creates a fresh Operator instance.
  using Factory = std::function<std::unique_ptr<Operator>()>;

  /**
   * @brief Registers or replaces a factory.
   * @param type Operator type name.
   * @param factory Factory producing operators of that type.
   */
  void registerFactory(std::string type, Factory factory);

  /**
   * @brief Tests whether a type is registered.
   * @param type Operator type name.
   * @return True if a factory exists for @p type.
   */
  bool contains(const std::string& type) const;

  /**
   * @brief Instantiates an operator.
   * @param type Operator type name.
   * @return A new operator.
   * @throws std::invalid_argument If @p type is not registered.
   */
  std::unique_ptr<Operator> create(const std::string& type) const;

  /**
   * @brief Loads a plugin library and lets it register operators.
   *
   * The library must export `bool dli_register_operators(OperatorRegistry*)` (see
   * RegisterOperatorsFn). The library stays loaded for the lifetime of the registry.
   *
   * @param path Path to the shared library.
   * @throws std::runtime_error If loading fails, the symbol is missing, or registration returns
   *     false.
   */
  void loadLibrary(const std::string& path);

 private:
  /// Factories keyed by operator type.
  std::map<std::string, Factory> factories_;
  /// dlopen handles of loaded plugin libraries, kept open on purpose.
  std::vector<void*> library_handles_;
};

/// Signature of the `dli_register_operators` entry point; returns true on success.
using RegisterOperatorsFn = bool (*)(OperatorRegistry*);

}  // namespace dli

#endif  // DLI_OPERATOR_H_
