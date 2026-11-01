/**
 * @file template.cc
 * @brief Code-generation template for the AOT Triton operator "transpose".
 *
 * This is a fragment, not a standalone translation unit: tools/build_triton_aot_ops.py
 * replaces every double-brace HASH placeholder with the compiled kernel's hash suffix
 * and splices the fragment into the generated plugin source. It is excluded from
 * clang-format because of those placeholders.
 */

/// Operator "transpose": 2-D transpose selected by the "perm" attribute.
class TransposeOp final : public dli::Operator {
 public:
  std::string type() const override { return "transpose"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext&) const override {
    if (inputs.size() != 1 || outputs.size() != 1) throw std::invalid_argument("transpose arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type());
    const auto perm = attrs.require<std::vector<std::int64_t>>("perm");
    if (inputs[0]->rank() != 2 || perm != std::vector<std::int64_t>({1, 0})) {
      throw std::invalid_argument("AOT transpose currently supports 2D perm [1,0]");
    }
    int rows = static_cast<int>(inputs[0]->dim(0)); int cols = static_cast<int>(inputs[0]->dim(1));
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, {cols, rows}, inputs[0]->deviceId());
    void* x = ptr(*inputs[0]); void* out = ptr(*outputs[0]);
    void* triton_scratch = nullptr;
    void* triton_profile_scratch = nullptr;
    void* args[] = {&x, &out, &rows, &cols, &triton_scratch, &triton_profile_scratch};
    transpose2d_{{HASH_transpose2d}}_kernel().launch(args, ceilDiv(rows, 16), ceilDiv(cols, 16), 1, 4 * 32);
  }
};

/// Registers the "transpose" operator factory with @p registry.
void register_transpose(dli::OperatorRegistry* registry) {
  registry->registerFactory("transpose", [] { return std::make_unique<TransposeOp>(); });
}
