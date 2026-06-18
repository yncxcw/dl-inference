#pragma once

#include <string>

namespace dli::grpc {

class Server {
 public:
  explicit Server(std::string endpoint);
  const std::string& endpoint() const { return endpoint_; }

 private:
  std::string endpoint_;
};

}  // namespace dli::grpc
