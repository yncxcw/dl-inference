#include <ATen/ATen.h>

#include "dli/execution_state.h"
#include "dli/kv_cache.h"
#include "test_support.h"

namespace {

void testCudaAppend() {
  // batch * heads > 1, so every [batch, head] plane must grow independently.
  const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  const auto first_key = at::randn({2, 3, 4, 8}, options);
  const auto first_value = at::randn({2, 3, 4, 8}, options);
  const auto next_key = at::randn({2, 3, 1, 8}, options);
  const auto next_value = at::randn({2, 3, 1, 8}, options);

  dli::ExecutionState state;
  auto& cache = state.kvCache();
  cache.append("layer", dli::Tensor(first_key), dli::Tensor(first_value));
  const dli::KVCache copy = cache;
  const dli::ExecutionState snapshot = state.deepClone();

  // External buffers have no ATen storage of their own.
  const auto device = static_cast<int>(next_key.get_device());
  cache.append(
      "layer",
      dli::Tensor::externalCuda(dli::DType::Float32, {2, 3, 1, 8}, next_key.data_ptr(), device),
      dli::Tensor::externalCuda(dli::DType::Float32, {2, 3, 1, 8}, next_value.data_ptr(), device));

  dli_test::expect(cache.sequenceLength("layer") == 5, "CUDA append grows the sequence");
  dli_test::expectAllClose(cache.get("layer")->key.torchTensor(),
                           at::cat({first_key, next_key}, 2));
  dli_test::expectAllClose(cache.get("layer")->value.torchTensor(),
                           at::cat({first_value, next_value}, 2));

  // Engine snapshots share KV entries, which is only safe because append()
  // replaces an entry instead of writing into it.
  dli_test::expect(copy.sequenceLength("layer") == 4, "a copied cache keeps its entries");
  dli_test::expect(snapshot.kvCache().sequenceLength("layer") == 4,
                   "a deep-cloned state keeps its entries");
  dli_test::expectAllClose(snapshot.kvCache().get("layer")->key.torchTensor(), first_key);

  dli_test::expectThrows(
      [&] {
        cache.append("layer", dli::Tensor(at::randn({2, 2, 1, 8}, options)),
                     dli::Tensor(at::randn({2, 2, 1, 8}, options)));
      },
      "kv cache rejects a head-count mismatch");
}

}  // namespace

int main() {
  return dli_test::run("KVCache", [] {
    dli::KVCache cache;
    dli_test::expect(cache.size() == 0, "kv cache starts empty");
    dli_test::expect(cache.get("missing") == nullptr, "kv cache missing entry");
    dli_test::expect(cache.sequenceLength("missing") == 0,
                     "missing KV entry has zero sequence length");
    auto key = dli::Tensor(at::zeros({1, 1, 1, 2}, at::TensorOptions().dtype(at::kFloat)));
    auto value = dli::Tensor(at::zeros({1, 1, 1, 2}, at::TensorOptions().dtype(at::kFloat)));
    dli_test::expectThrows([&] { cache.append("layer", key, value); },
                           "kv cache rejects CPU tensors");
    cache.clear();
    dli_test::expect(cache.size() == 0, "kv cache clear");

    // CPU-only PyTorch builds (such as CI) skip the device checks.
    if (at::hasCUDA()) testCudaAppend();
  });
}
