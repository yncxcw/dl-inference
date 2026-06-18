#include "dli/graph.h"

#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>

namespace dli {
namespace {

struct JsonValue {
  enum class Type { Null, Bool, Int, Number, String, Array, Object };
  Type type = Type::Null;
  bool bool_value = false;
  std::int64_t int_value = 0;
  double number_value = 0.0;
  std::string string_value;
  std::vector<JsonValue> array_value;
  std::map<std::string, JsonValue> object_value;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}
  JsonValue parse() {
    auto value = parseValue();
    skipWhitespace();
    if (pos_ != text_.size()) throw std::invalid_argument("unexpected trailing JSON content");
    return value;
  }

 private:
  JsonValue parseValue() {
    skipWhitespace();
    if (pos_ >= text_.size()) throw std::invalid_argument("unexpected end of JSON");
    const char c = text_[pos_];
    if (c == '{') return parseObject();
    if (c == '[') return parseArray();
    if (c == '"') {
      JsonValue value;
      value.type = JsonValue::Type::String;
      value.string_value = parseString();
      return value;
    }
    if (c == 't' || c == 'f') return parseBool();
    if (c == 'n') return parseNull();
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parseNumber();
    throw std::invalid_argument("unexpected JSON token");
  }

  JsonValue parseObject() {
    expect('{');
    JsonValue object;
    object.type = JsonValue::Type::Object;
    skipWhitespace();
    if (consume('}')) return object;
    while (true) {
      skipWhitespace();
      if (peek() != '"') throw std::invalid_argument("JSON object key must be a string");
      auto key = parseString();
      skipWhitespace();
      expect(':');
      object.object_value.emplace(std::move(key), parseValue());
      skipWhitespace();
      if (consume('}')) return object;
      expect(',');
    }
  }

  JsonValue parseArray() {
    expect('[');
    JsonValue array;
    array.type = JsonValue::Type::Array;
    skipWhitespace();
    if (consume(']')) return array;
    while (true) {
      array.array_value.push_back(parseValue());
      skipWhitespace();
      if (consume(']')) return array;
      expect(',');
    }
  }

  std::string parseString() {
    expect('"');
    std::string result;
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') return result;
      if (c != '\\') {
        result.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) throw std::invalid_argument("unterminated JSON escape");
      const char escaped = text_[pos_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/': result.push_back(escaped); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        default: throw std::invalid_argument("unsupported JSON string escape");
      }
    }
    throw std::invalid_argument("unterminated JSON string");
  }

  JsonValue parseBool() {
    JsonValue value;
    value.type = JsonValue::Type::Bool;
    if (text_.substr(pos_, 4) == "true") {
      value.bool_value = true;
      pos_ += 4;
      return value;
    }
    if (text_.substr(pos_, 5) == "false") {
      value.bool_value = false;
      pos_ += 5;
      return value;
    }
    throw std::invalid_argument("invalid JSON boolean");
  }

  JsonValue parseNull() {
    if (text_.substr(pos_, 4) != "null") throw std::invalid_argument("invalid JSON null");
    pos_ += 4;
    return {};
  }

  JsonValue parseNumber() {
    const auto start = pos_;
    if (peek() == '-') ++pos_;
    consumeDigits();
    bool is_float = false;
    if (consume('.')) {
      is_float = true;
      consumeDigits();
    }
    if (peek() == 'e' || peek() == 'E') {
      is_float = true;
      ++pos_;
      if (peek() == '+' || peek() == '-') ++pos_;
      consumeDigits();
    }
    const auto token = std::string(text_.substr(start, pos_ - start));
    JsonValue value;
    if (is_float) {
      value.type = JsonValue::Type::Number;
      value.number_value = std::stod(token);
    } else {
      value.type = JsonValue::Type::Int;
      value.int_value = std::stoll(token);
    }
    return value;
  }

  void consumeDigits() {
    if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      throw std::invalid_argument("expected JSON digit");
    }
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
  }
  void skipWhitespace() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
  }
  char peek() const { return pos_ >= text_.size() ? '\0' : text_[pos_]; }
  bool consume(char expected) {
    if (peek() == expected) {
      ++pos_;
      return true;
    }
    return false;
  }
  void expect(char expected) {
    if (!consume(expected)) throw std::invalid_argument(std::string("expected JSON token: ") + expected);
  }

  std::string_view text_;
  std::size_t pos_ = 0;
};

const JsonValue& requireField(const JsonValue& object, const std::string& name) {
  if (object.type != JsonValue::Type::Object) throw std::invalid_argument("expected JSON object");
  const auto it = object.object_value.find(name);
  if (it == object.object_value.end()) throw std::invalid_argument("missing JSON field: " + name);
  return it->second;
}

const JsonValue* optionalField(const JsonValue& object, const std::string& name) {
  if (object.type != JsonValue::Type::Object) throw std::invalid_argument("expected JSON object");
  const auto it = object.object_value.find(name);
  return it == object.object_value.end() ? nullptr : &it->second;
}

std::string asString(const JsonValue& value, const std::string& field) {
  if (value.type != JsonValue::Type::String) throw std::invalid_argument("JSON field must be string: " + field);
  return value.string_value;
}

std::vector<std::string> asStringArray(const JsonValue& value, const std::string& field) {
  if (value.type != JsonValue::Type::Array) throw std::invalid_argument("JSON field must be string array: " + field);
  std::vector<std::string> result;
  for (const auto& item : value.array_value) result.push_back(asString(item, field));
  return result;
}

AttributeValue toAttributeValue(const JsonValue& value) {
  switch (value.type) {
    case JsonValue::Type::Bool: return value.bool_value;
    case JsonValue::Type::Int: return value.int_value;
    case JsonValue::Type::Number: return value.number_value;
    case JsonValue::Type::String: return value.string_value;
    case JsonValue::Type::Array: {
      if (value.array_value.empty()) return std::vector<std::int64_t>{};
      bool all_strings = true, all_ints = true, all_numbers = true;
      for (const auto& item : value.array_value) {
        all_strings = all_strings && item.type == JsonValue::Type::String;
        all_ints = all_ints && item.type == JsonValue::Type::Int;
        all_numbers = all_numbers && (item.type == JsonValue::Type::Int || item.type == JsonValue::Type::Number);
      }
      if (all_strings) {
        std::vector<std::string> out;
        for (const auto& item : value.array_value) out.push_back(item.string_value);
        return out;
      }
      if (all_ints) {
        std::vector<std::int64_t> out;
        for (const auto& item : value.array_value) out.push_back(item.int_value);
        return out;
      }
      if (all_numbers) {
        std::vector<double> out;
        for (const auto& item : value.array_value) {
          out.push_back(item.type == JsonValue::Type::Int ? static_cast<double>(item.int_value) : item.number_value);
        }
        return out;
      }
      break;
    }
    default: break;
  }
  throw std::invalid_argument("unsupported attribute JSON value");
}

std::string escapeJson(const std::string& value) {
  std::string out;
  for (const char c : value) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else out.push_back(c);
  }
  return out;
}

void appendStringArray(std::ostringstream& out, const std::vector<std::string>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << '"' << escapeJson(values[i]) << '"';
  }
  out << ']';
}

void appendAttributeValue(std::ostringstream& out, const AttributeValue& value) {
  std::visit(
      [&](const auto& item) {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::string>) {
          out << '"' << escapeJson(item) << '"';
        } else if constexpr (std::is_same_v<T, bool>) {
          out << (item ? "true" : "false");
        } else if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, double>) {
          out << item;
        } else {
          out << '[';
          for (std::size_t i = 0; i < item.size(); ++i) {
            if (i != 0) out << ',';
            if constexpr (std::is_same_v<typename T::value_type, std::string>) {
              out << '"' << escapeJson(item[i]) << '"';
            } else {
              out << item[i];
            }
          }
          out << ']';
        }
      },
      value);
}

}  // namespace

Graph Graph::fromJson(const std::string& json) {
  const auto root = JsonParser(json).parse();
  if (root.type != JsonValue::Type::Object) throw std::invalid_argument("graph JSON root must be an object");
  Graph graph;
  if (const auto* format = optionalField(root, "format")) graph.format = asString(*format, "format");
  if (graph.format != "dli.graph.v1") throw std::invalid_argument("unsupported graph format: " + graph.format);
  if (const auto* model_type = optionalField(root, "model_type")) graph.model_type = asString(*model_type, "model_type");
  if (const auto* weights = optionalField(root, "weights")) graph.weights = asString(*weights, "weights");
  if (const auto* inputs = optionalField(root, "inputs")) graph.inputs = asStringArray(*inputs, "inputs");
  if (const auto* outputs = optionalField(root, "outputs")) graph.outputs = asStringArray(*outputs, "outputs");

  const auto& nodes_json = requireField(root, "nodes");
  if (nodes_json.type != JsonValue::Type::Array) throw std::invalid_argument("graph nodes must be an array");
  for (const auto& node_json : nodes_json.array_value) {
    if (node_json.type != JsonValue::Type::Object) throw std::invalid_argument("graph node must be an object");
    Node node;
    node.name = asString(requireField(node_json, "name"), "name");
    if (const auto* op = optionalField(node_json, "op")) node.op_type = asString(*op, "op");
    else node.op_type = asString(requireField(node_json, "op_type"), "op_type");
    node.inputs = asStringArray(requireField(node_json, "inputs"), "inputs");
    node.outputs = asStringArray(requireField(node_json, "outputs"), "outputs");
    if (const auto* attrs = optionalField(node_json, "attrs")) {
      if (attrs->type != JsonValue::Type::Object) throw std::invalid_argument("node attrs must be an object");
      for (const auto& [name, value] : attrs->object_value) node.attributes.set(name, toAttributeValue(value));
    }
    graph.nodes.push_back(std::move(node));
  }
  return graph;
}

Graph Graph::fromJsonFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("failed to open graph file: " + path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return fromJson(buffer.str());
}

std::string Graph::toJson() const {
  std::ostringstream out;
  out << "{\n";
  out << "  \"format\":\"" << escapeJson(format) << "\",\n";
  out << "  \"model_type\":\"" << escapeJson(model_type) << "\",\n";
  if (!weights.empty()) out << "  \"weights\":\"" << escapeJson(weights) << "\",\n";
  out << "  \"inputs\":";
  appendStringArray(out, inputs);
  out << ",\n  \"outputs\":";
  appendStringArray(out, outputs);
  out << ",\n  \"nodes\":[\n";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    out << "    {\"name\":\"" << escapeJson(node.name) << "\",\"op\":\""
        << escapeJson(node.op_type) << "\",\"inputs\":";
    appendStringArray(out, node.inputs);
    out << ",\"outputs\":";
    appendStringArray(out, node.outputs);
    out << ",\"attrs\":{";
    std::size_t attr_index = 0;
    for (const auto& [name, value] : node.attributes.values()) {
      if (attr_index++ != 0) out << ',';
      out << '"' << escapeJson(name) << "\":";
      appendAttributeValue(out, value);
    }
    out << "}}";
    if (i + 1 != nodes.size()) out << ',';
    out << "\n";
  }
  out << "  ]\n}\n";
  return out.str();
}

}  // namespace dli
