#pragma once

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
  std::vector<Node> nodes;

  static Graph fromJson(const std::string& json);
  static Graph fromJsonFile(const std::string& path);
  std::string toJson() const;
};

}  // namespace dli
