/**
 * @file test_grpc_server.cc
 * @brief Tests that the gRPC server stores its endpoint.
 */

#include "dli/grpc/server.h"
#include "test_support.h"

// Verifies: the server reports the endpoint it was constructed with.
int main() {
  return dli_test::run("grpc_server", [] {
    dli::grpc::Server server("0.0.0.0:50051");
    dli_test::expect(server.endpoint() == "0.0.0.0:50051", "server endpoint mismatch");
  });
}
