#include "dli/grpc/server.h"

#include <utility>

namespace dli::grpc {

Server::Server(std::string endpoint) : endpoint_(std::move(endpoint)) {}

}  // namespace dli::grpc
