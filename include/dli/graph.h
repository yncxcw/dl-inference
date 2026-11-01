/**
 * @file graph.h
 * @brief In-memory representation of a "dli.graph.v1" computation graph and its JSON codec.
 */

#ifndef DLI_GRAPH_H_
#define DLI_GRAPH_H_

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
