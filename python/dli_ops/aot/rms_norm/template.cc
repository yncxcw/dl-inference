dli::CudaAotKernel& rmsNormKernel(std::int64_t hidden) {
  switch (hidden) {
    case 2: return rms_norm_h2_{{HASH_rms_norm_h2}}_kernel();
    case 4: return rms_norm_h4_{{HASH_rms_norm_h4}}_kernel();
    case 8: return rms_norm_h8_{{HASH_rms_norm_h8}}_kernel();
    case 128: return rms_norm_h128_{{HASH_rms_norm_h128}}_kernel();
    case 4096: return rms_norm_h4096_{{HASH_rms_norm_h4096}}_kernel();
  }
  throw std::invalid_argument("no AOT rms_norm specialization for hidden=" + std::to_string(hidden));
}

class RmsNormOp final : public dli::Operator {
 public:
  std::string type() const override { return "rms_norm"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext&) const override {
    if (inputs.size() != 2 || outputs.size() != 1) throw std::invalid_argument("rms_norm arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type()); requireFloat(*inputs[1], type());
    const int rows = static_cast<int>(inputs[0]->numel() / inputs[0]->shape().back());
    const float eps = static_cast<float>(attrDouble(attrs, "eps", 1e-6));
    *outputs[0] = dli::Tensor::cuda(inputs[0]->dtype(), inputs[0]->shape(), inputs[0]->deviceId());
    void* x = ptr(*inputs[0]); void* w = ptr(*inputs[1]); void* out = ptr(*outputs[0]);
    void* args[] = {&x, &w, &out, const_cast<int*>(&rows), const_cast<float*>(&eps)};
    rmsNormKernel(inputs[0]->shape().back()).launch(args, rows, 1, 1, 4 * 32);
  }
};

void register_rms_norm(dli::OperatorRegistry* registry) {
  registry->registerFactory("rms_norm", [] { return std::make_unique<RmsNormOp>(); });
}
