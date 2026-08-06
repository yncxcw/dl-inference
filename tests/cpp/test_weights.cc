#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "dli/weights.h"
#include "test_support.h"

int main() {
  return dli_test::run("weights", [] {
    const auto bin_path = dli_test::tempPath("dli_test_weights", ".bin");
    const auto manifest_path = dli_test::tempPath("dli_test_weights", ".json");
    {
      const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
      const std::int64_t ids[] = {7, 8};
      std::ofstream bin(bin_path, std::ios::binary);
      bin.write(reinterpret_cast<const char*>(weight), sizeof(weight));
      bin.write(reinterpret_cast<const char*>(ids), sizeof(ids));
    }
    {
      std::ofstream manifest(manifest_path);
      manifest << R"({
  "format": "dli.weights.v1",
  "data": ")" << bin_path.filename().string()
               << R"(",
  "tensors": {
    "linear.weight": {"dtype": "float32", "shape": [2, 2], "offset": 0, "nbytes": 16},
    "ids": {"dtype": "int64", "shape": [2], "offset": 16, "nbytes": 16}
  }
})";
    }

    const auto weights = dli::loadWeights(manifest_path.string());
    dli_test::expect(weights.at("linear.weight").isCpu(), "weights load to CPU by default");
    dli_test::expect(weights.at("linear.weight").shape() == std::vector<std::int64_t>({2, 2}),
                     "weight shape");
    dli_test::expect(weights.at("linear.weight").data<float>()[3] == 4.0f, "weight value");
    dli_test::expect(weights.at("ids").data<std::int64_t>()[1] == 8, "int64 weight value");
    std::remove(bin_path.c_str());
    std::remove(manifest_path.c_str());

    const auto bad_manifest_path = dli_test::tempPath("dli_bad_weights", ".json");
    {
      std::ofstream manifest(bad_manifest_path);
      manifest << R"({"format":"bad","data":"missing.bin","tensors":{}})";
    }
    dli_test::expectThrows([&] { dli::loadWeights(bad_manifest_path.string()); },
                           "bad weights format should throw");
    std::remove(bad_manifest_path.c_str());
  });
}
