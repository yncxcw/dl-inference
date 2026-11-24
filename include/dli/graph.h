/**
 * @file graph.h
 * @brief In-memory representation of a "dli.graph.v1" computation graph and its JSON codec.
 */

#ifndef DLI_GRAPH_H_
#define DLI_GRAPH_H_

#include <map>
#include <string>
#include <vector>

#include "dli/attributes.h"

namespace dli {

/// One operator invocation in a Graph.
struct Node {
  /// Unique node name, used in diagnostics.
  std::string name;
  /// Registered operator type to instantiate (for example "aten" or "linear").
  std::string op_type;
  /// Names of the input tensors, in operator argument order.
  std::vector<std::string> inputs;
  /// Names of the tensors this node produces, in operator output order.
  std::vector<std::string> outputs;
  /// Operator configuration.
  Attributes attributes;
};

/// A topologically ordered list of nodes plus metadata, loadable from JSON.
class Graph {
 public:
  /// Schema identifier; only "dli.graph.v1" is accepted when parsing.
  std::string format = "dli.graph.v1";
  /// Free-form model family label, such as the Hugging Face model type.
  std::string model_type;
  /// Optional path to the weights manifest consumed by loadWeights().
  std::string weights;
  /// Names of the externally supplied input tensors.
  std::vector<std::string> inputs;
  /// Names of the tensors returned by Engine::run().
  std::vector<std::string> outputs;
  /**
   * @brief Input edges fed from the request state: edge name -> state key.
   *
   * With @ref state_outputs and @ref state_initializers, this carries per-request values, such
   * as a KV cache, from one run to the next. For example, an exported Qwen3.5 step graph has:
   *
   *     "state_inputs": {"state_6": "layer.3.keys"},
   *     "state_outputs": {"next_state.layer.3.keys": "layer.3.keys"},
   *     "state_initializers": {
   *       "layer.3.keys": "qwen3.5-0.8b_step.__state_initializer__.layer.3.keys"}
   *
   * The first run has no "layer.3.keys" yet, so Engine::run() feeds edge "state_6" from the
   * initializer, an all-zero [1, 2, 4096, 256] key cache stored with the weights. The graph
   * returns the cache with this step's keys written in as "next_state.layer.3.keys", and the
   * engine stores that as "layer.3.keys" only if the whole run succeeds. The next run feeds
   * "state_6" from the stored value. Inside the graph, state edges are ordinary tensors.
   */
  std::map<std::string, std::string> state_inputs;
  /// Produced edges stored into the request state after a successful run: edge name -> state key.
  std::map<std::string, std::string> state_outputs;
  /// Tensors used for state keys the request has no value for yet: state key -> tensor name.
  std::map<std::string, std::string> state_initializers;
  /// Nodes in execution order.
  std::vector<Node> nodes;

  /**
   * @brief Parses a graph from JSON text.
   * @param json Document text.
   * @return The parsed graph.
   * @throws std::invalid_argument If the JSON is malformed, the format is unsupported, or a
   *     required field is missing or has the wrong type.
   */
  static Graph fromJson(const std::string& json);

  /**
   * @brief Parses a graph from a JSON file.
   * @param path Path to the file.
   * @return The parsed graph.
   * @throws std::runtime_error If the file cannot be opened.
   * @throws std::invalid_argument As for fromJson().
   */
  static Graph fromJsonFile(const std::string& path);

  /// @return The graph serialized as JSON that fromJson() accepts.
  std::string toJson() const;
};

}  // namespace dli

#endif  // DLI_GRAPH_H_
