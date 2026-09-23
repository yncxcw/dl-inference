#include <cstdint>
#include <string>
#include <vector>

#include "dli/graph.h"
#include "test_support.h"

int main() {
  return dli_test::run("Graph", [] {
    const auto graph = dli::Graph::fromJson(R"({
      "format": "dli.graph.v1",
      "model_type": "graph_test",
      "weights": "weights.json",
      "inputs": ["x"],
      "outputs": ["y"],
      "state_inputs": {"cache_in": "layer.0.cache"},
      "state_outputs": {"cache_out": "layer.0.cache"},
      "state_initializers": {"layer.0.cache": "layer.0.cache.initial"},
      "nodes": [{
        "name": "node",
        "op_type": "aten",
        "inputs": ["x"],
        "outputs": ["y"],
        "attrs": {
          "name": "aten::relu",
          "flag": true,
          "int": 2,
          "number": 1.5,
          "ints": [1, 2],
          "numbers": [1, 2.5],
          "strings": ["a", "b"]
        }
      }]
    })");
    dli_test::expect(graph.model_type == "graph_test", "graph model type");
    dli_test::expect(graph.weights == "weights.json", "graph weights");
    dli_test::expect(graph.state_inputs.at("cache_in") == "layer.0.cache", "graph state input");
    dli_test::expect(graph.state_outputs.at("cache_out") == "layer.0.cache", "graph state output");
    dli_test::expect(graph.state_initializers.at("layer.0.cache") == "layer.0.cache.initial",
                     "graph state initializer");
    dli_test::expect(graph.nodes.size() == 1, "graph node count");
    dli_test::expect(graph.nodes[0].op_type == "aten", "graph op_type alias");
    dli_test::expect(graph.nodes[0].attributes.require<bool>("flag"), "graph bool attr");
    dli_test::expect(graph.nodes[0].attributes.require<std::int64_t>("int") == 2, "graph int attr");
    dli_test::expect(graph.nodes[0].attributes.require<double>("number") == 1.5,
                     "graph number attr");
    dli_test::expect(graph.nodes[0].attributes.require<std::vector<std::int64_t>>("ints")[1] == 2,
                     "graph ints attr");
    dli_test::expect(graph.nodes[0].attributes.require<std::vector<double>>("numbers")[1] == 2.5,
                     "graph numbers attr");
    dli_test::expect(
        graph.nodes[0].attributes.require<std::vector<std::string>>("strings")[0] == "a",
        "graph strings attr");

    const auto roundtrip = dli::Graph::fromJson(graph.toJson());
    dli_test::expect(roundtrip.nodes[0].attributes.require<std::string>("name") == "aten::relu",
                     "graph roundtrip attrs");
    dli_test::expect(roundtrip.state_inputs == graph.state_inputs, "graph roundtrip state inputs");
    dli_test::expect(roundtrip.state_outputs == graph.state_outputs,
                     "graph roundtrip state outputs");
    dli_test::expect(roundtrip.state_initializers == graph.state_initializers,
                     "graph roundtrip state initializers");
    dli_test::expectThrows([] { dli::Graph::fromJson(R"({"format":"bad","nodes":[]})"); },
                           "bad graph format should throw");
    dli_test::expectThrows([] { dli::Graph::fromJson(R"({"format":"dli.graph.v1"})"); },
                           "missing graph nodes should throw");
  });
}
