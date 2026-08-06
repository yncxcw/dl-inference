#include <cstdint>
#include <vector>

#include "dli/attributes.h"
#include "dli/utils.h"
#include "test_support.h"

int main() {
  return dli_test::run("utils", [] {
    dli_test::expect(dli::product({2, 3, 4}) == 24, "product full shape");
    dli_test::expect(dli::product({2, 3, 4}, 1) == 12, "product suffix shape");
    dli_test::expect(dli::product({2, 3, 4}, 1, 2) == 3, "product range shape");
    dli_test::expect(dli::ceilDiv(17, 8) == 3, "ceilDiv value");
    dli_test::expect(dli::formatShape({2, 3, 4}) == "[2,3,4]", "formatShape value");

    dli::Attributes attrs;
    attrs.set("as_int", std::int64_t{3});
    attrs.set("as_double", 2.5);
    attrs.set("ints", std::vector<std::int64_t>{2, 4});
    dli_test::expect(dli::attrDouble(attrs, "as_int", 0.0) == 3.0, "attrDouble int conversion");
    dli_test::expect(dli::attrDouble(attrs, "as_double", 0.0) == 2.5, "attrDouble double");
    dli_test::expect(dli::attrDouble(attrs, "missing", 1.25) == 1.25, "attrDouble fallback");
    dli_test::expect(dli::attrInts(attrs, "ints", {}) == std::vector<std::int64_t>({2, 4}),
                     "attrInts value");
    dli_test::expect(dli::attrInts(attrs, "missing", {1}) == std::vector<std::int64_t>({1}),
                     "attrInts fallback");
    dli_test::expect(dli::reshapeShape({2, 3, 4}, {6, -1}) == std::vector<std::int64_t>({6, 4}),
                     "reshape infer");
    dli_test::expectThrows([] { dli::reshapeShape({2, 3}, {4}); },
                           "reshape size mismatch should throw");
    dli_test::expectThrows([] { dli::reshapeShape({2, 3}, {-1, -1}); },
                           "reshape multiple infer should throw");
    dli_test::expectThrows([] { dli::reshapeShape({2, 3}, {0, 6}); },
                           "reshape zero dim should throw");
  });
}
