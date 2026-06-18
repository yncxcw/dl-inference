#pragma once

#include <map>
#include <string>

#include "dli/tensor.h"

namespace dli {

struct KVCacheEntry {
  Tensor key;
  Tensor value;
};

class KVCache {
 public:
  void append(const std::string& name, const Tensor& key, const Tensor& value);
  const KVCacheEntry* get(const std::string& name) const;
  std::size_t size() const { return entries_.size(); }
  void clear() { entries_.clear(); }

 private:
  std::map<std::string, KVCacheEntry> entries_;
};

}  // namespace dli
