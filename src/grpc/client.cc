#include "dli/grpc/client.h"

#include <utility>

namespace dli::grpc {

Client::Client(std::string endpoint) : endpoint_(std::move(endpoint)) {}

}  // namespace dli::grpc
