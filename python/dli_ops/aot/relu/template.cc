class ReluOp final : public dli::Operator {
 public:
  std::string type() const override { return "relu"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes&, dli::ExecutionContext&) const override {
    if (inputs.size() != 1 || outputs.size() != 1) throw std::invalid_argument("relu arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type());
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, inputs[0]->shape(), inputs[0]->deviceId());
    int total = static_cast<int>(inputs[0]->numel());
    void* x = ptr(*inputs[0]); void* out = ptr(*outputs[0]);
    void* args[] = {&x, &out, &total};
    relu_{{HASH_relu}}_kernel().launch(args, ceilDiv(total, 256), 1, 1, 4 * 32);
  }
};

void register_relu(dli::OperatorRegistry* registry) {
  registry->registerFactory("relu", [] { return std::make_unique<ReluOp>(); });
}
