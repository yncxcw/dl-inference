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

void register_reshape(dli::OperatorRegistry* registry) {
  registry->registerFactory("reshape", [] { return std::make_unique<ReshapeOp>(); });
}
