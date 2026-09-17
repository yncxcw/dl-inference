class RotaryOp final : public dli::Operator {
 public:
  std::string type() const override { return "rotary_embedding"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext& context) const override {
    if (inputs.size() != 4 || outputs.size() != 2) throw std::invalid_argument("rotary_embedding arity");
    requireCudaInputs(inputs, type());
    for (const auto* input : inputs) requireFloat(*input, type());
    if (inputs[0]->rank() != 4 || inputs[1]->rank() != 4 || inputs[2]->rank() != 2 ||
        inputs[3]->rank() != 2) {
      throw std::invalid_argument("rotary_embedding expects rank-4 Q/K and rank-2 tables");
    }
    if (inputs[0]->dim(0) != inputs[1]->dim(0) ||
        inputs[0]->dim(2) != inputs[1]->dim(2) ||
        inputs[0]->dim(3) != inputs[1]->dim(3)) {
      throw std::invalid_argument("rotary_embedding Q/K shape mismatch");
    }
    for (std::size_t index = 1; index < inputs.size(); ++index) {
      if (inputs[index]->deviceId() != inputs[0]->deviceId())
        throw std::invalid_argument("rotary_embedding device mismatch");
    }
    for (const auto* input : {inputs[0], inputs[1], inputs[2], inputs[3]}) {
      for (std::size_t dim = 0; dim < input->rank(); ++dim) {
        if (input->dim(dim) <= 0 || input->dim(dim) > std::numeric_limits<int>::max())
          throw std::invalid_argument("rotary_embedding dimensions must be positive runtime int values");
      }
    }
    const int head_dim = inputs[0]->dim(3), seq = inputs[0]->dim(2), pairs = inputs[2]->dim(1);
    if (pairs <= 0 || pairs > head_dim / 2)
      throw std::invalid_argument("rotary_embedding invalid rotary dimension");
    const auto checked_total_pairs = [&](const dli::Tensor& tensor) {
      std::int64_t total = pairs;
      for (std::size_t dim = 0; dim + 1 < tensor.rank(); ++dim) {
        if (tensor.dim(dim) > std::numeric_limits<int>::max() / total)
          throw std::invalid_argument("rotary_embedding input is too large");
        total *= tensor.dim(dim);
      }
      return static_cast<int>(total);
    };
    int q_total_pairs = checked_total_pairs(*inputs[0]);
    int k_total_pairs = checked_total_pairs(*inputs[1]);
    const int total_pairs = q_total_pairs > k_total_pairs ? q_total_pairs : k_total_pairs;
    const auto graph_offset = attrs.value_or<std::int64_t>("start_pos", 0);
    const auto runtime_offset = context.position_offset;
    if (graph_offset < 0 || runtime_offset < 0 ||
        graph_offset > std::numeric_limits<std::int64_t>::max() - runtime_offset) {
      throw std::invalid_argument("rotary_embedding invalid position offset");
    }
    const auto start_pos_64 = graph_offset + runtime_offset;
    if (start_pos_64 > std::numeric_limits<int>::max() ||
        start_pos_64 + seq > inputs[2]->dim(0) ||
        inputs[2]->shape() != inputs[3]->shape()) {
      throw std::invalid_argument("rotary_embedding position exceeds rotary table");
    }
    int start_pos = static_cast<int>(start_pos_64);
    dli::CudaAotKernel* kernel = nullptr;
    if (head_dim == 2 && seq == 1 && pairs == 1) kernel = &rotary_d2_s1_p1_{{HASH_rotary_d2_s1_p1}}_kernel();
    if (head_dim == 2 && seq == 2 && pairs == 1) kernel = &rotary_d2_s2_p1_{{HASH_rotary_d2_s2_p1}}_kernel();
    if (head_dim == 128 && seq == 1 && pairs == 64) kernel = &rotary_d128_s1_p64_{{HASH_rotary_d128_s1_p64}}_kernel();
    if (head_dim == 128 && seq == 128 && pairs == 64) kernel = &rotary_d128_s128_p64_{{HASH_rotary_d128_s128_p64}}_kernel();
    if (head_dim == 256 && seq == 1 && pairs == 32) kernel = &rotary_d256_s1_p32_{{HASH_rotary_d256_s1_p32}}_kernel();
    if (kernel == nullptr) throw std::invalid_argument("no AOT rotary specialization");
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, inputs[0]->shape(), inputs[0]->deviceId());
    *outputs[1] = dli::Tensor::cuda(dli::DType::Float32, inputs[1]->shape(), inputs[1]->deviceId());
    dli::cudaMemcpyBytes(outputs[0]->deviceData(), inputs[0]->deviceData(), inputs[0]->nbytes(), dli::CudaMemcpyKind::DeviceToDevice);
    dli::cudaMemcpyBytes(outputs[1]->deviceData(), inputs[1]->deviceData(), inputs[1]->nbytes(), dli::CudaMemcpyKind::DeviceToDevice);
    void* q = ptr(*inputs[0]); void* k = ptr(*inputs[1]); void* cos = ptr(*inputs[2]); void* sin = ptr(*inputs[3]);
    void* out_q = ptr(*outputs[0]); void* out_k = ptr(*outputs[1]);
    void* triton_scratch = nullptr;
    void* args[] = {&q, &k, &cos, &sin, &out_q, &out_k, &q_total_pairs, &k_total_pairs, &start_pos, &triton_scratch};
    kernel->launch(args, ceilDiv(total_pairs, 128), 1, 1, 4 * 32);
  }
};

void register_rotary(dli::OperatorRegistry* registry) {
  registry->registerFactory("rotary_embedding", [] { return std::make_unique<RotaryOp>(); });
}
