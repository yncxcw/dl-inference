#include "dli/grpc/client.h"
#include "test_support.h"

int main() {
  return dli_test::run("grpc_client", [] {
    dli::grpc::Client client("127.0.0.1:50051");
    dli_test::expect(client.endpoint() == "127.0.0.1:50051", "client endpoint mismatch");
  });
}
