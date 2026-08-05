#include "dli/grpc/server.h"

#include "test_support.h"

int main() {
  return dli_test::run("grpc_server", [] {
    dli::grpc::Server server("0.0.0.0:50051");
    dli_test::expect(server.endpoint() == "0.0.0.0:50051", "server endpoint mismatch");
  });
}
