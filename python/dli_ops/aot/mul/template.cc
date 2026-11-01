/**
 * @file template.cc
 * @brief Code-generation template for the AOT Triton operator "mul".
 *
 * This is a fragment, not a standalone translation unit: tools/build_triton_aot_ops.py
 * replaces every double-brace HASH placeholder with the compiled kernel's hash suffix
 * and splices the fragment into the generated plugin source. It is excluded from
 * clang-format because of those placeholders.
 */

/// Operator "mul": Element-wise float32 multiplication of two CUDA tensors.
class MulOp final : public dli::Operator {
 public:
  std::string type() const override { return "mul"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes&, dli::ExecutionContext&) const override {
    if (inputs.size() != 2 || outputs.size() != 1) throw std::invalid_argument("mul arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type()); requireFloat(*inputs[1], type());
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, inputs[0]->shape(), inputs[0]->deviceId());
    int total = static_cast<int>(inputs[0]->numel()); int width = static_cast<int>(inputs[1]->numel());
    void* a = ptr(*inputs[0]); void* b = ptr(*inputs[1]); void* out = ptr(*outputs[0]);
    void* triton_scratch = nullptr;
    void* triton_profile_scratch = nullptr;
    void* args[] = {&a, &b, &out, &total, &width, &triton_scratch, &triton_profile_scratch};
    mul_{{HASH_mul}}_kernel().launch(args, ceilDiv(total, 256), 1, 1, 4 * 32);
  }
};

/// Registers the "mul" operator factory with @p registry.
void register_mul(dli::OperatorRegistry* registry) {
  registry->registerFactory("mul", [] { return std::make_unique<MulOp>(); });
}
