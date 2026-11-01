/**
 * @file template.cc
 * @brief Code-generation template for the AOT Triton operator "reshape".
 *
 * This is a fragment, not a standalone translation unit: tools/build_triton_aot_ops.py
 * replaces every double-brace HASH placeholder with the compiled kernel's hash suffix
 * and splices the fragment into the generated plugin source. It is excluded from
 * clang-format because of those placeholders.
 */

/// Operator "reshape": Reshape that returns a view over the input storage; launches no kernel.
class ReshapeOp final : public dli::Operator {
 public:
  std::string type() const override { return "reshape"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext&) const override {
    if (inputs.size() != 1 || outputs.size() != 1) throw std::invalid_argument("reshape arity");
    requireCudaInputs(inputs, type());
    *outputs[0] = inputs[0]->withShape(reshapeShape(inputs[0]->shape(), attrs.require<std::vector<std::int64_t>>("shape")));
  }
};

/// Registers the "reshape" operator factory with @p registry.
void register_reshape(dli::OperatorRegistry* registry) {
  registry->registerFactory("reshape", [] { return std::make_unique<ReshapeOp>(); });
}
