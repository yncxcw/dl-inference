dli::CudaAotKernel& linearKernel(std::int64_t k, bool has_bias) {
  switch (k) {
    case 2: return has_bias ? linear_k2_b1_{{HASH_linear_k2_b1}}_kernel() : linear_k2_b0_{{HASH_linear_k2_b0}}_kernel();
    case 4: return has_bias ? linear_k4_b1_{{HASH_linear_k4_b1}}_kernel() : linear_k4_b0_{{HASH_linear_k4_b0}}_kernel();
    case 8: return has_bias ? linear_k8_b1_{{HASH_linear_k8_b1}}_kernel() : linear_k8_b0_{{HASH_linear_k8_b0}}_kernel();
    case 64: return has_bias ? linear_k64_b1_{{HASH_linear_k64_b1}}_kernel() : linear_k64_b0_{{HASH_linear_k64_b0}}_kernel();
    case 128: return has_bias ? linear_k128_b1_{{HASH_linear_k128_b1}}_kernel() : linear_k128_b0_{{HASH_linear_k128_b0}}_kernel();
    case 1024: return has_bias ? linear_k1024_b1_{{HASH_linear_k1024_b1}}_kernel() : linear_k1024_b0_{{HASH_linear_k1024_b0}}_kernel();
    case 4096: return has_bias ? linear_k4096_b1_{{HASH_linear_k4096_b1}}_kernel() : linear_k4096_b0_{{HASH_linear_k4096_b0}}_kernel();
  }
  throw std::invalid_argument("no AOT linear specialization for k=" + std::to_string(k));
}

class LinearOp final : public dli::Operator {
 public:
  std::string type() const override { return "linear"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes&, dli::ExecutionContext&) const override {
    if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1) throw std::invalid_argument("linear arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type()); requireFloat(*inputs[1], type());
    const bool has_bias = inputs.size() == 3;
    const int m = static_cast<int>(product(inputs[0]->shape(), 0, inputs[0]->rank() - 1));
    const int n = static_cast<int>(inputs[1]->dim(0));
    std::vector<std::int64_t> out_shape = inputs[0]->shape(); out_shape.back() = n;
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, out_shape, inputs[0]->deviceId());
    void* x = ptr(*inputs[0]); void* w = ptr(*inputs[1]);
    void* bias = has_bias ? ptr(*inputs[2]) : ptr(*outputs[0]);
    void* out = ptr(*outputs[0]);
    void* args[] = {&x, &w, &bias, &out, const_cast<int*>(&m), const_cast<int*>(&n)};
    linearKernel(inputs[1]->dim(1), has_bias).launch(args, ceilDiv(m, 16), ceilDiv(n, 16), 1, 4 * 32);
  }
};

void register_linear(dli::OperatorRegistry* registry) {
  registry->registerFactory("linear", [] { return std::make_unique<LinearOp>(); });
}
