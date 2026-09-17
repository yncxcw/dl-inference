#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dli/attributes.h"
#include "dli/execution_state.h"
#include "dli/kv_cache.h"
#include "dli/tensor.h"

namespace dli {

struct ExecutionContext {
  // Keep the legacy cache pointer first so existing operator plugins continue
  // to see the field at the ABI-stable offset while newer plugins can use the
  // full execution state.
  KVCache* kv_cache = nullptr;
  ExecutionState* state = nullptr;
  std::int64_t position_offset = 0;
};

class Operator {
 public:
  virtual ~Operator() = default;
  virtual std::string type() const = 0;
  virtual void compute(const std::vector<const Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs, const Attributes& attrs,
                       ExecutionContext& context) const = 0;
};

class OperatorRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Operator>()>;
  void registerFactory(std::string type, Factory factory);
  bool contains(const std::string& type) const;
  std::unique_ptr<Operator> create(const std::string& type) const;
  void loadLibrary(const std::string& path);

 private:
  std::map<std::string, Factory> factories_;
  std::vector<void*> library_handles_;
};

using RegisterOperatorsFn = bool (*)(OperatorRegistry*);

}  // namespace dli
