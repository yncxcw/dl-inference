/**
 * @file template.cc
 * @brief Code-generation template for the AOT Triton operator "silu".
 *
 * This is a fragment, not a standalone translation unit: tools/build_triton_aot_ops.py
 * replaces every double-brace HASH placeholder with the compiled kernel's hash suffix
 * and splices the fragment into the generated plugin source. It is excluded from
 * clang-format because of those placeholders.
 */

/// Operator "silu": Element-wise SiLU (swish) activation.
class SiluOp final : public dli::Operator {
 public:
  std::string type() const override { return "silu"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes&, dli::ExecutionContext&) const override {
    if (inputs.size() != 1 || outputs.size() != 1) throw std::invalid_argument("silu arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type());
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, inputs[0]->shape(), inputs[0]->deviceId());
    int total = static_cast<int>(inputs[0]->numel());
    void* x = ptr(*inputs[0]); void* out = ptr(*outputs[0]);
    void* triton_scratch = nullptr;
    void* triton_profile_scratch = nullptr;
    void* args[] = {&x, &out, &total, &triton_scratch, &triton_profile_scratch};
    silu_{{HASH_silu}}_kernel().launch(args, ceilDiv(total, 256), 1, 1, 4 * 32);
  }
};

/// Registers the "silu" operator factory with @p registry.
void register_silu(dli::OperatorRegistry* registry) {
  registry->registerFactory("silu", [] { return std::make_unique<SiluOp>(); });
}
