class BatchNorm2dOp final : public dli::Operator {
 public:
  std::string type() const override { return "batch_norm2d"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext&) const override {
    if (inputs.size() != 5 || outputs.size() != 1) throw std::invalid_argument("batch_norm2d arity");
    requireCudaInputs(inputs, type());
    for (const auto* input : inputs) requireFloat(*input, type());
    if (inputs[0]->shape().size() != 4) throw std::invalid_argument("batch_norm2d expects NCHW input");
    const int channels = static_cast<int>(inputs[0]->dim(1));
    for (std::size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i]->numel() != channels) {
        throw std::invalid_argument("batch_norm2d parameter size must match input channels");
      }
    }
    const int total = static_cast<int>(inputs[0]->numel());
    const int spatial = static_cast<int>(inputs[0]->dim(2) * inputs[0]->dim(3));
    const float eps = static_cast<float>(attrDouble(attrs, "eps", 1e-5));
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, inputs[0]->shape(), inputs[0]->deviceId());
    void* x = ptr(*inputs[0]);
    void* weight = ptr(*inputs[1]);
    void* bias = ptr(*inputs[2]);
    void* running_mean = ptr(*inputs[3]);
    void* running_var = ptr(*inputs[4]);
    void* out = ptr(*outputs[0]);
    void* triton_scratch = nullptr;
    void* args[] = {&x, &weight, &bias, &running_mean, &running_var, &out,
                    const_cast<int*>(&total), const_cast<int*>(&channels),
                    const_cast<int*>(&spatial), const_cast<float*>(&eps), &triton_scratch};
    batch_norm2d_{{HASH_batch_norm2d}}_kernel().launch(args, ceilDiv(total, 256), 1, 1, 4 * 32);
  }
};

void register_batch_norm2d(dli::OperatorRegistry* registry) {
  registry->registerFactory("batch_norm2d", [] { return std::make_unique<BatchNorm2dOp>(); });
}
