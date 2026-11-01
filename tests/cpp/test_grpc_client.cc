/**
 * @file test_grpc_client.cc
 * @brief Tests that the gRPC client stores its endpoint.
 */

#include "dli/grpc/client.h"
#include "test_support.h"

// Verifies: the client reports the endpoint it was constructed with.
int main() {
  return dli_test::run("grpc_client", [] {
    dli::grpc::Client client("127.0.0.1:50051");
    dli_test::expect(client.endpoint() == "127.0.0.1:50051", "client endpoint mismatch");
  });
}
