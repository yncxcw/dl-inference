#pragma once

#include <string>

namespace dli::grpc {

class Client {
 public:
  explicit Client(std::string endpoint);
  const std::string& endpoint() const { return endpoint_; }

 private:
  std::string endpoint_;
};

}  // namespace dli::grpc
