#include <cstdint>
#include <string>
#include <vector>

#include "dli/attributes.h"
#include "test_support.h"

int main() {
  return dli_test::run("Attributes", [] {
    dli::Attributes attrs;
    attrs.set("int", std::int64_t{7});
    attrs.set("double", 0.25);
    attrs.set("bool", true);
    attrs.set("string", std::string("value"));
    attrs.set("ints", std::vector<std::int64_t>{1, 2, 3});
    attrs.set("doubles", std::vector<double>{1.5, 2.5});
    attrs.set("strings", std::vector<std::string>{"a", "b"});

    dli_test::expect(attrs.contains("int"), "attribute contains int");
    dli_test::expect(!attrs.contains("missing"), "attribute missing");
    dli_test::expect(attrs.require<std::int64_t>("int") == 7, "attribute int value");
    dli_test::expect(attrs.require<double>("double") == 0.25, "attribute double value");
    dli_test::expect(attrs.require<bool>("bool"), "attribute bool value");
    dli_test::expect(attrs.require<std::string>("string") == "value", "attribute string value");
    dli_test::expect(attrs.require<std::vector<std::int64_t>>("ints").size() == 3,
                     "attribute int list value");
    dli_test::expect(attrs.require<std::vector<double>>("doubles")[1] == 2.5,
                     "attribute double list value");
    dli_test::expect(attrs.require<std::vector<std::string>>("strings")[0] == "a",
                     "attribute string list value");
    dli_test::expect(attrs.value_or<std::int64_t>("fallback", 11) == 11,
                     "attribute fallback value");
    dli_test::expectThrows([&] { attrs.require<double>("int"); },
                           "attribute wrong type should throw");
    dli_test::expectThrows([&] { attrs.require<std::int64_t>("missing"); },
                           "attribute missing should throw");
  });
}
