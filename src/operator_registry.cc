#include <dlfcn.h>

#include <stdexcept>

#include "dli/operator.h"

namespace dli {

void OperatorRegistry::registerFactory(std::string type, Factory factory) {
  factories_[std::move(type)] = std::move(factory);
}

bool OperatorRegistry::contains(const std::string& type) const {
  return factories_.find(type) != factories_.end();
}

std::unique_ptr<Operator> OperatorRegistry::create(const std::string& type) const {
  const auto it = factories_.find(type);
  if (it == factories_.end()) throw std::invalid_argument("operator is not registered: " + type);
  return it->second();
}

void OperatorRegistry::loadLibrary(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* error = dlerror();
    throw std::runtime_error("failed to load operator library: " + path + ": " +
                             (error == nullptr ? "unknown error" : error));
  }
  auto* symbol = dlsym(handle, "dli_register_operators");
  if (symbol == nullptr) {
    dlclose(handle);
    throw std::runtime_error("operator library is missing dli_register_operators: " + path);
  }
  auto* fn = reinterpret_cast<RegisterOperatorsFn>(symbol);
  if (!fn(this)) {
    dlclose(handle);
    throw std::runtime_error("operator library registration failed: " + path);
  }
  library_handles_.push_back(handle);
}

}  // namespace dli
