/**
 * @file template.cc
 * @brief Code-generation template for the AOT Triton operator "embedding".
 *
 * This is a fragment, not a standalone translation unit: tools/build_triton_aot_ops.py
 * replaces every double-brace HASH placeholder with the compiled kernel's hash suffix
 * and splices the fragment into the generated plugin source. It is excluded from
 * clang-format because of those placeholders.
 */

/// Returns the AOT embedding kernel specialized for the given hidden size; throws if none exists.
dli::CudaAotKernel& embeddingKernel(std::int64_t hidden) {
  switch (hidden) {
    case 2: return embedding_h2_{{HASH_embedding_h2}}_kernel();
    case 4: return embedding_h4_{{HASH_embedding_h4}}_kernel();
    case 8: return embedding_h8_{{HASH_embedding_h8}}_kernel();
    case 128: return embedding_h128_{{HASH_embedding_h128}}_kernel();
    case 4096: return embedding_h4096_{{HASH_embedding_h4096}}_kernel();
  }
  throw std::invalid_argument("no AOT embedding specialization for hidden=" + std::to_string(hidden));
}

/// Operator "embedding": Embedding table lookup, specialized per hidden size.
class EmbeddingOp final : public dli::Operator {
 public:
  std::string type() const override { return "embedding"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes&, dli::ExecutionContext&) const override {
    if (inputs.size() != 2 || outputs.size() != 1) throw std::invalid_argument("embedding arity");
    requireCudaInputs(inputs, type());
    if (inputs[0]->dtype() != dli::DType::Int64) throw std::invalid_argument("embedding ids must be int64");
    requireFloat(*inputs[1], type());
    auto shape = inputs[0]->shape();
    shape.push_back(inputs[1]->dim(1));
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, shape, inputs[0]->deviceId());
    int total = static_cast<int>(inputs[0]->numel());
    void* ids = ptr(*inputs[0]); void* table = ptr(*inputs[1]); void* out = ptr(*outputs[0]);
    void* triton_scratch = nullptr;
    void* triton_profile_scratch = nullptr;
    void* args[] = {&ids, &table, &out, &total, &triton_scratch, &triton_profile_scratch};
    embeddingKernel(inputs[1]->dim(1)).launch(args, ceilDiv(total, 16), ceilDiv(inputs[1]->dim(1), 32), 1, 4 * 32);
  }
};

/// Registers the "embedding" operator factory with @p registry.
void register_embedding(dli::OperatorRegistry* registry) {
  registry->registerFactory("embedding", [] { return std::make_unique<EmbeddingOp>(); });
}
