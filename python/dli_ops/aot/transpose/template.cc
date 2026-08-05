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
    void* args[] = {&x, &out, &rows, &cols, &triton_scratch};
    transpose2d_{{HASH_transpose2d}}_kernel().launch(args, ceilDiv(rows, 16), ceilDiv(cols, 16), 1, 4 * 32);
  }
};

void register_transpose(dli::OperatorRegistry* registry) {
  registry->registerFactory("transpose", [] { return std::make_unique<TransposeOp>(); });
}
