#include "test_support.h"

#include <ATen/ATen.h>

#include "dli/kv_cache.h"

int main() {
  return dli_test::run("KVCache", [] {
    dli::KVCache cache;
    dli_test::expect(cache.size() == 0, "kv cache starts empty");
    dli_test::expect(cache.get("missing") == nullptr, "kv cache missing entry");
    auto key = dli::Tensor(at::zeros({1, 1, 1, 2}, at::TensorOptions().dtype(at::kFloat)));
    auto value = dli::Tensor(at::zeros({1, 1, 1, 2}, at::TensorOptions().dtype(at::kFloat)));
    dli_test::expectThrows([&] { cache.append("layer", key, value); },
                           "kv cache rejects CPU tensors");
    cache.clear();
    dli_test::expect(cache.size() == 0, "kv cache clear");
  });
}

