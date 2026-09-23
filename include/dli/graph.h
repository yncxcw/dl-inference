#pragma once

#include <map>
#include <string>
#include <vector>

#include "dli/attributes.h"

namespace dli {

struct Node {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  Attributes attributes;
};

class Graph {
 public:
  std::string format = "dli.graph.v1";
  std::string model_type;
  std::string weights;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  // Stateful graph edges remain ordinary tensors. The engine injects the
  // declared inputs before execution and publishes outputs only on success.
  std::map<std::string, std::string> state_inputs;
  std::map<std::string, std::string> state_outputs;
  std::map<std::string, std::string> state_initializers;
  std::vector<Node> nodes;

  static Graph fromJson(const std::string& json);
  static Graph fromJsonFile(const std::string& path);
  std::string toJson() const;
};

}  // namespace dli
