/**
 * @file template.cc
 * @brief Code-generation template for the AOT Triton operator "matmul".
 *
 * This is a fragment, not a standalone translation unit: tools/build_triton_aot_ops.py
 * replaces every double-brace HASH placeholder with the compiled kernel's hash suffix
 * and splices the fragment into the generated plugin source. It is excluded from
 * clang-format because of those placeholders.
 */

/// Returns the AOT matmul kernel specialized for the reduction size @p k; throws if none exists.
dli::CudaAotKernel& matmulKernel(std::int64_t k) {
  switch (k) {
    case 2: return matmul_k2_{{HASH_matmul_k2}}_kernel();
    case 4: return matmul_k4_{{HASH_matmul_k4}}_kernel();
    case 8: return matmul_k8_{{HASH_matmul_k8}}_kernel();
    case 64: return matmul_k64_{{HASH_matmul_k64}}_kernel();
    case 128: return matmul_k128_{{HASH_matmul_k128}}_kernel();
    case 1024: return matmul_k1024_{{HASH_matmul_k1024}}_kernel();
    case 4096: return matmul_k4096_{{HASH_matmul_k4096}}_kernel();
  }
  throw std::invalid_argument("no AOT matmul specialization for k=" + std::to_string(k));
}

/// Operator "matmul": Matrix multiplication of float32 CUDA tensors.
class MatmulOp final : public dli::Operator {
 public:
  std::string type() const override { return "matmul"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes&, dli::ExecutionContext&) const override {
    if (inputs.size() != 2 || outputs.size() != 1) throw std::invalid_argument("matmul arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type()); requireFloat(*inputs[1], type());
    const int m = static_cast<int>(inputs[0]->dim(0));
    const int n = static_cast<int>(inputs[1]->dim(1));
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, {m, n}, inputs[0]->deviceId());
    void* a = ptr(*inputs[0]); void* b = ptr(*inputs[1]); void* out = ptr(*outputs[0]);
    void* triton_scratch = nullptr;
    void* triton_profile_scratch = nullptr;
    void* args[] = {&a, &b, &out, const_cast<int*>(&m), const_cast<int*>(&n), &triton_scratch, &triton_profile_scratch};
    matmulKernel(inputs[0]->dim(1)).launch(args, ceilDiv(m, 16), ceilDiv(n, 16), 1, 4 * 32);
  }
};

/// Registers the "matmul" operator factory with @p registry.
void register_matmul(dli::OperatorRegistry* registry) {
  registry->registerFactory("matmul", [] { return std::make_unique<MatmulOp>(); });
}
