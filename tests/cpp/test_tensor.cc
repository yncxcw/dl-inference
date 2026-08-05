#include "test_support.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <vector>

#include "dli/tensor.h"

int main() {
  return dli_test::run("Tensor", [] {
    dli_test::expect(dli::toString(dli::DeviceType::Cpu) == "cpu", "cpu device string");
    dli_test::expect(dli::toString(dli::DeviceType::Cuda) == "cuda", "cuda device string");
    dli_test::expect(dli::deviceFromString("gpu") == dli::DeviceType::Cuda, "gpu device alias");
    dli_test::expect(dli::toString(dli::DType::Float32) == "float32", "float32 dtype string");
    dli_test::expect(dli::dtypeFromString("i64") == dli::DType::Int64, "int64 dtype alias");
    dli_test::expect(dli::byteSize(dli::DType::Int64) == sizeof(std::int64_t), "int64 byte size");

    auto torch_tensor = at::tensor({1.0f, 2.0f, 3.0f, 4.0f},
                                   at::TensorOptions().dtype(at::kFloat)).view({2, 2});
    auto tensor = dli::Tensor(torch_tensor);
    dli_test::expect(tensor.isCpu(), "host tensor device");
    dli_test::expect(tensor.numel() == 4, "host tensor numel");
    dli_test::expect(tensor.nbytes() == 16, "host tensor nbytes");
    dli_test::expect(tensor.data<float>()[2] == 3.0f, "host tensor value");

    auto ints = dli::Tensor::zeros(dli::DType::Int64, {2, 2});
    dli_test::expect(ints.isCpu(), "zeros tensor is CPU");
    dli_test::expect(ints.data<std::int64_t>()[3] == 0, "zeros tensor value");
    auto host_view = ints.withShape({4});
    dli_test::expect(host_view.shape() == std::vector<std::int64_t>({4}), "host tensor view shape");
    dli_test::expect(dli::sameShape(ints, dli::Tensor::zeros(dli::DType::Int64, {2, 2})),
                     "same shape helper");
    dli_test::expect(dli::contiguousOffset({2, 3, 4}, {1, 2, 3}) == 23,
                     "contiguous offset");

    auto* fake_ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1000));
    auto external = dli::Tensor::externalCuda(dli::DType::Float32, {2, 3}, fake_ptr, 0);
    dli_test::expect(external.isCuda(), "external tensor is CUDA");
    dli_test::expect(external.deviceData() == fake_ptr, "external CUDA pointer");
    dli_test::expect(external.nbytes() == 24, "external CUDA nbytes");
    auto cuda_view = external.withShape({3, 2});
    dli_test::expect(cuda_view.isCuda(), "CUDA view device");
    dli_test::expect(cuda_view.deviceData() == fake_ptr, "CUDA view pointer");
    dli_test::expect(cuda_view.shape() == std::vector<std::int64_t>({3, 2}), "CUDA view shape");

    dli_test::expectThrows([] { dli::Tensor(dli::DType::Float32, {-1}); },
                           "negative tensor dim should throw");
    dli_test::expectThrows([&] { ints.dim(2); }, "tensor dim out of range should throw");
    dli_test::expectThrows([&] { ints.deviceData(); }, "CPU tensor deviceData should throw");
    dli_test::expectThrows([&] { ints.withShape({3}); }, "bad tensor view shape should throw");
    dli_test::expectThrows([] { dli::dtypeFromString("bad"); }, "bad dtype should throw");
    dli_test::expectThrows([] { dli::deviceFromString("bad"); }, "bad device should throw");
  });
}

