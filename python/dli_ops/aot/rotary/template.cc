class RotaryOp final : public dli::Operator {
 public:
  std::string type() const override { return "rotary_embedding"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext&) const override {
    if ((inputs.size() != 4 && inputs.size() != 5) || outputs.size() != 2) throw std::invalid_argument("rotary_embedding arity");
    requireCudaInputs(inputs, type());
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, inputs[0]->shape(), inputs[0]->deviceId());
    *outputs[1] = dli::Tensor::cuda(dli::DType::Float32, inputs[1]->shape(), inputs[1]->deviceId());
    dli::cudaMemcpyBytes(outputs[0]->deviceData(), inputs[0]->deviceData(), inputs[0]->nbytes(), dli::CudaMemcpyKind::DeviceToDevice);
    dli::cudaMemcpyBytes(outputs[1]->deviceData(), inputs[1]->deviceData(), inputs[1]->nbytes(), dli::CudaMemcpyKind::DeviceToDevice);
    const int head_dim = inputs[0]->dim(3), seq = inputs[0]->dim(2), pairs = inputs[2]->dim(1);
    const int total_pairs = static_cast<int>(product(inputs[0]->shape(), 0, inputs[0]->rank() - 1) * pairs);
    int start_pos = static_cast<int>(attrs.value_or<std::int64_t>("start_pos", 0));
    void* q = ptr(*inputs[0]); void* k = ptr(*inputs[1]); void* cos = ptr(*inputs[2]); void* sin = ptr(*inputs[3]);
    void* out_q = ptr(*outputs[0]); void* out_k = ptr(*outputs[1]);
    void* triton_scratch = nullptr;
    void* args[] = {&q, &k, &cos, &sin, &out_q, &out_k, const_cast<int*>(&total_pairs), &start_pos, &triton_scratch};
    dli::CudaAotKernel* kernel = nullptr;
    if (head_dim == 2 && seq == 1 && pairs == 1) kernel = &rotary_d2_s1_p1_{{HASH_rotary_d2_s1_p1}}_kernel();
    if (head_dim == 2 && seq == 2 && pairs == 1) kernel = &rotary_d2_s2_p1_{{HASH_rotary_d2_s2_p1}}_kernel();
    if (head_dim == 128 && seq == 1 && pairs == 64) kernel = &rotary_d128_s1_p64_{{HASH_rotary_d128_s1_p64}}_kernel();
    if (head_dim == 128 && seq == 128 && pairs == 64) kernel = &rotary_d128_s128_p64_{{HASH_rotary_d128_s128_p64}}_kernel();
    if (kernel == nullptr) throw std::invalid_argument("no AOT rotary specialization");
    kernel->launch(args, ceilDiv(total_pairs, 128), 1, 1, 4 * 32);
  }
};

void register_rotary(dli::OperatorRegistry* registry) {
  registry->registerFactory("rotary_embedding", [] { return std::make_unique<RotaryOp>(); });
}
