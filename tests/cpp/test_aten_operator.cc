#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "dli/attributes.h"
#include "dli/engine.h"
#include "dli/graph.h"
#include "dli/tensor.h"
#include "test_support.h"

int main() {
  return dli_test::run("AtenOperator", [] {
    {
      dli::Graph graph;
      graph.model_type = "aten_relu_test";
      graph.inputs = {"x"};
      graph.outputs = {"y"};
      dli::Attributes attrs;
      attrs.set("name", std::string("aten::relu"));
      graph.nodes.push_back({"relu", "aten", {"x"}, {"y"}, std::move(attrs)});

      auto input = at::tensor({-1.0f, 0.5f, 2.0f, -3.0f}, at::TensorOptions().dtype(at::kFloat));
      dli::Engine engine;
      auto outputs = engine.run(graph, {{"x", dli::Tensor(input)}});
      dli_test::expectAllClose(outputs.at("y").torchTensor(), at::relu(input));
    }

    {
      dli::Graph graph;
      graph.model_type = "aten_add_test";
      graph.inputs = {"lhs", "rhs"};
      graph.outputs = {"sum"};
      dli::Attributes attrs;
      attrs.set("name", std::string("aten::add"));
      attrs.set("overload", std::string("Tensor"));
      attrs.set("attr_order", std::vector<std::string>{"alpha"});
      attrs.set("alpha", 2.0);
      graph.nodes.push_back({"add", "aten", {"lhs", "rhs"}, {"sum"}, std::move(attrs)});

      auto lhs = at::tensor({1.0f, 2.0f, 3.0f}, at::TensorOptions().dtype(at::kFloat));
      auto rhs = at::tensor({10.0f, 20.0f, 30.0f}, at::TensorOptions().dtype(at::kFloat));
      dli::Engine engine;
      auto outputs = engine.run(graph, {{"lhs", dli::Tensor(lhs)}, {"rhs", dli::Tensor(rhs)}});
      dli_test::expectAllClose(outputs.at("sum").torchTensor(), at::add(lhs, rhs, 2.0));
    }

    {
      dli::Graph graph;
      graph.inputs = {"x"};
      graph.outputs = {"y"};
      graph.nodes.push_back({"aten_missing_name", "aten", {"x"}, {"y"}, {}});
      auto input = at::tensor({1.0f}, at::TensorOptions().dtype(at::kFloat));
      dli::Engine engine;
      dli_test::expectThrows([&] { engine.run(graph, {{"x", dli::Tensor(input)}}); },
                             "aten missing name should throw");
    }
  });
}
