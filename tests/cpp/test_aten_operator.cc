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
      graph.outputs = {"joined"};
      dli::Attributes attrs;
      attrs.set("name", std::string("aten::cat"));
      attrs.set("arguments", std::vector<std::string>{"tl:0,1", "i:0"});
      graph.nodes.push_back({"cat", "aten", {"lhs", "rhs"}, {"joined"}, std::move(attrs)});

      auto lhs = at::tensor({1, 2}, at::TensorOptions().dtype(at::kLong));
      auto rhs = at::tensor({3, 4}, at::TensorOptions().dtype(at::kLong));
      dli::Engine engine;
      auto outputs = engine.run(graph, {{"lhs", dli::Tensor(lhs)}, {"rhs", dli::Tensor(rhs)}});
      dli_test::expectAllClose(outputs.at("joined").torchTensor(), at::cat({lhs, rhs}));
    }

    {
      dli::Graph graph;
      graph.outputs = {"even"};
      dli::Attributes attrs;
      attrs.set("name", std::string("aten::slice"));
      attrs.set("overload", std::string("Tensor"));
      attrs.set("arguments", std::vector<std::string>{"t:0", "i:0", "n", "n", "i:2"});
      graph.nodes.push_back({"slice", "aten", {"x"}, {"even"}, std::move(attrs)});

      auto input = at::arange(0, 8, at::TensorOptions().dtype(at::kLong));
      dli::Engine engine;
      auto outputs = engine.run(graph, {{"x", dli::Tensor(input)}});
      dli_test::expectAllClose(outputs.at("even").torchTensor(), input.slice(0, 0, 8, 2));
    }

    {
      dli::Graph graph;
      graph.outputs = {"range"};
      dli::Attributes attrs;
      attrs.set("name", std::string("aten::arange"));
      attrs.set("overload", std::string("start_step"));
      attrs.set("arguments", std::vector<std::string>{"i:0", "i:7", "i:2", "dtype:int64",
                                                      "layout:strided", "device:0", "b:false"});
      graph.nodes.push_back({"arange", "aten", {"anchor"}, {"range"}, std::move(attrs)});

      auto anchor = at::zeros({1}, at::TensorOptions().dtype(at::kFloat));
      dli::Engine engine;
      auto outputs = engine.run(graph, {{"anchor", dli::Tensor(anchor)}});
      dli_test::expectAllClose(
          outputs.at("range").torchTensor(),
          at::arange(0, 7, 2, at::TensorOptions().dtype(at::kLong).device(anchor.device())));
    }

    {
      dli::Graph graph;
      graph.outputs = {"updated"};
      dli::Attributes attrs;
      attrs.set("name", std::string("aten::index_put"));
      attrs.set("arguments", std::vector<std::string>{"t:0", "otl:1", "t:2", "b:false"});
      graph.nodes.push_back(
          {"index_put", "aten", {"base", "index", "values"}, {"updated"}, std::move(attrs)});

      auto base = at::zeros({4}, at::TensorOptions().dtype(at::kFloat));
      auto index = at::tensor({1, 3}, at::TensorOptions().dtype(at::kLong));
      auto values = at::tensor({5.0f, 9.0f}, at::TensorOptions().dtype(at::kFloat));
      dli::Engine engine;
      auto outputs = engine.run(graph, {{"base", dli::Tensor(base)},
                                        {"index", dli::Tensor(index)},
                                        {"values", dli::Tensor(values)}});
      dli_test::expectAllClose(outputs.at("updated").torchTensor(),
                               at::index_put(base, {index}, values, false));
    }

    {
      dli::Graph graph;
      graph.outputs = {"values", "indices"};
      dli::Attributes attrs;
      attrs.set("name", std::string("aten::max"));
      attrs.set("overload", std::string("dim"));
      attrs.set("arguments", std::vector<std::string>{"t:0", "i:1", "b:false"});
      graph.nodes.push_back({"maximum", "aten", {"x"}, {"values", "indices"}, std::move(attrs)});

      auto input = at::tensor({1.0f, 4.0f, 2.0f, 8.0f, 3.0f, 7.0f}).view({2, 3});
      dli::Engine engine;
      auto outputs = engine.run(graph, {{"x", dli::Tensor(input)}});
      const auto expected = at::max(input, 1, false);
      dli_test::expectAllClose(outputs.at("values").torchTensor(), std::get<0>(expected));
      dli_test::expectAllClose(outputs.at("indices").torchTensor(), std::get<1>(expected));
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

    {
      dli::Graph graph;
      graph.outputs = {"updated"};
      dli::Attributes attrs;
      attrs.set("name", std::string("aten::add_"));
      attrs.set("overload", std::string("Tensor"));
      attrs.set("arguments", std::vector<std::string>{"t:0", "t:1", "i:1"});
      graph.nodes.push_back(
          {"in_place_add", "aten", {"state", "delta"}, {"updated"}, std::move(attrs)});
      auto state = at::zeros({1}, at::TensorOptions().dtype(at::kFloat));
      auto delta = at::ones({1}, at::TensorOptions().dtype(at::kFloat));
      dli::Engine engine;
      dli_test::expectThrows(
          [&] {
            engine.run(graph, {{"state", dli::Tensor(state)}, {"delta", dli::Tensor(delta)}});
          },
          "mutating aten schema should throw");
      dli_test::expect(state.item<float>() == 0.0f,
                       "rejected mutating schema must not modify its input");
    }
  });
}
