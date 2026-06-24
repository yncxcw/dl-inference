#include "dli/weights.h"

#include "dli/cuda_runtime.h"

#include <ATen/ATen.h>

#include <cctype>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace dli {
namespace {

at::ScalarType torchDType(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return at::kFloat;
    case DType::Int64:
      return at::kLong;
  }
  throw std::invalid_argument("unknown dtype");
}

struct JsonValue {
  enum class Type { Null, Bool, Int, Number, String, Array, Object };
  Type type = Type::Null;
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
    if (c == 'n') return parseNull();
    if (c == 't' || c == 'f') return parseBool();
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
      if (escaped == '"' || escaped == '\\' || escaped == '/') result.push_back(escaped);
      else if (escaped == 'n') result.push_back('\n');
      else if (escaped == 't') result.push_back('\t');
      else throw std::invalid_argument("unsupported JSON string escape");
    }
    throw std::invalid_argument("unterminated JSON string");
  }
  JsonValue parseNull() {
    if (text_.substr(pos_, 4) != "null") throw std::invalid_argument("invalid JSON null");
    pos_ += 4;
    return {};
  }
  JsonValue parseBool() {
    if (text_.substr(pos_, 4) == "true") {
      pos_ += 4;
      return {};
    }
    if (text_.substr(pos_, 5) == "false") {
      pos_ += 5;
      return {};
    }
    throw std::invalid_argument("invalid JSON boolean");
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

std::string asString(const JsonValue& value, const std::string& field) {
  if (value.type != JsonValue::Type::String) throw std::invalid_argument("JSON field must be string: " + field);
  return value.string_value;
}

std::int64_t asInt(const JsonValue& value, const std::string& field) {
  if (value.type != JsonValue::Type::Int) throw std::invalid_argument("JSON field must be integer: " + field);
  return value.int_value;
}

std::vector<std::int64_t> asIntArray(const JsonValue& value, const std::string& field) {
  if (value.type != JsonValue::Type::Array) throw std::invalid_argument("JSON field must be integer array: " + field);
  std::vector<std::int64_t> result;
  for (const auto& item : value.array_value) result.push_back(asInt(item, field));
  return result;
}

std::string readTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("failed to open weights manifest: " + path.string());
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

std::vector<std::byte> readBytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("failed to open weights data: " + path.string());
  const std::vector<char> chars{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  std::vector<std::byte> bytes(chars.size());
  if (!chars.empty()) std::memcpy(bytes.data(), chars.data(), chars.size());
  return bytes;
}

Tensor tensorFromBytes(DType dtype, std::vector<std::int64_t> shape,
                       const std::byte* data, std::size_t nbytes,
                       DeviceType device, int device_id) {
  auto host = at::empty(shape, at::TensorOptions().dtype(torchDType(dtype)).device(at::kCPU));
  const auto expected_nbytes = static_cast<std::size_t>(host.numel()) * byteSize(dtype);
  if (expected_nbytes != nbytes) {
    throw std::invalid_argument("weights tensor byte count does not match dtype and shape");
  }
  std::memcpy(host.data_ptr(), data, nbytes);
  if (device == DeviceType::Cpu) return Tensor(host);
  return Tensor(host.to(at::Device(at::kCUDA, device_id)));
}

}  // namespace

TensorMap loadWeights(const std::string& manifest_path, DeviceType device, int device_id) {
  const std::filesystem::path manifest(manifest_path);
  const auto root = JsonParser(readTextFile(manifest)).parse();
  if (root.type != JsonValue::Type::Object) throw std::invalid_argument("weights manifest root must be an object");
  const auto format = asString(requireField(root, "format"), "format");
  if (format != "dli.weights.v1") throw std::invalid_argument("unsupported weights format: " + format);
  auto data_path = std::filesystem::path(asString(requireField(root, "data"), "data"));
  if (data_path.is_relative()) data_path = manifest.parent_path() / data_path;
  const auto bytes = readBytes(data_path);
  const auto& tensors = requireField(root, "tensors");
  if (tensors.type != JsonValue::Type::Object) throw std::invalid_argument("weights tensors must be an object");

  TensorMap result;
  for (const auto& [name, metadata] : tensors.object_value) {
    if (metadata.type != JsonValue::Type::Object) throw std::invalid_argument("weights tensor metadata must be an object: " + name);
    const auto dtype = dtypeFromString(asString(requireField(metadata, "dtype"), "dtype"));
    const auto shape = asIntArray(requireField(metadata, "shape"), "shape");
    const auto offset = static_cast<std::size_t>(asInt(requireField(metadata, "offset"), "offset"));
    const auto nbytes = static_cast<std::size_t>(asInt(requireField(metadata, "nbytes"), "nbytes"));
    if (offset > bytes.size() || nbytes > bytes.size() - offset) {
      throw std::invalid_argument("weights tensor range exceeds data file: " + name);
    }
    result.emplace(name, tensorFromBytes(dtype, shape, bytes.data() + offset, nbytes, device, device_id));
  }
  return result;
}

}  // namespace dli
