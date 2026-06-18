class AttentionOp final : public dli::Operator {
 public:
  std::string type() const override { return "attention"; }
  void compute(const std::vector<const dli::Tensor*>& raw_inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext& context) const override {
    if (raw_inputs.size() != 3 || outputs.size() != 1) throw std::invalid_argument("attention arity");
    requireCudaInputs(raw_inputs, type()); requireFloat(*raw_inputs[0], type());
    std::vector<const dli::Tensor*> inputs = raw_inputs;
    if (attrs.contains("kv_cache")) {
      const auto& name = attrs.require<std::string>("kv_cache");
      if (!name.empty()) {
        if (context.kv_cache == nullptr) throw std::invalid_argument("attention missing KV cache");
        context.kv_cache->append(name, *raw_inputs[1], *raw_inputs[2]);
        const auto* entry = context.kv_cache->get(name);
        inputs[1] = &entry->key; inputs[2] = &entry->value;
      }
    }
    const int batch_heads = static_cast<int>(inputs[0]->dim(0) * inputs[0]->dim(1));
    const int seq_q = static_cast<int>(inputs[0]->dim(2));
    const int seq_k = static_cast<int>(inputs[1]->dim(2));
    const int head_dim = static_cast<int>(inputs[0]->dim(3));
    const bool causal = attrs.value_or<bool>("causal", false);
    if (attrs.contains("scale")) throw std::invalid_argument("AOT attention currently supports default scale only");
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, inputs[0]->shape(), inputs[0]->deviceId());
    void* q = ptr(*inputs[0]); void* k = ptr(*inputs[1]); void* v = ptr(*inputs[2]); void* out = ptr(*outputs[0]);
    void* args[] = {&q, &k, &v, &out};
    dli::CudaAotKernel* kernel = nullptr;
    if (seq_q == 1 && seq_k == 1 && head_dim == 2 && !causal) kernel = &attention_q1_k1_d2_c0_{{HASH_attention_q1_k1_d2_c0}}_kernel();
    if (seq_q == 1 && seq_k == 1 && head_dim == 2 && causal) kernel = &attention_q1_k1_d2_c1_{{HASH_attention_q1_k1_d2_c1}}_kernel();
    if (seq_q == 1 && seq_k == 2 && head_dim == 2 && !causal) kernel = &attention_q1_k2_d2_c0_{{HASH_attention_q1_k2_d2_c0}}_kernel();
    if (seq_q == 1 && seq_k == 2 && head_dim == 2 && causal) kernel = &attention_q1_k2_d2_c1_{{HASH_attention_q1_k2_d2_c1}}_kernel();
    if (seq_q == 2 && seq_k == 2 && head_dim == 2 && !causal) kernel = &attention_q2_k2_d2_c0_{{HASH_attention_q2_k2_d2_c0}}_kernel();
    if (seq_q == 2 && seq_k == 2 && head_dim == 2 && causal) kernel = &attention_q2_k2_d2_c1_{{HASH_attention_q2_k2_d2_c1}}_kernel();
    if (seq_q == 1 && seq_k == 1 && head_dim == 128 && !causal) kernel = &attention_q1_k1_d128_c0_{{HASH_attention_q1_k1_d128_c0}}_kernel();
    if (seq_q == 1 && seq_k == 1 && head_dim == 128 && causal) kernel = &attention_q1_k1_d128_c1_{{HASH_attention_q1_k1_d128_c1}}_kernel();
    if (seq_q == 1 && seq_k == 128 && head_dim == 128 && !causal) kernel = &attention_q1_k128_d128_c0_{{HASH_attention_q1_k128_d128_c0}}_kernel();
    if (seq_q == 1 && seq_k == 128 && head_dim == 128 && causal) kernel = &attention_q1_k128_d128_c1_{{HASH_attention_q1_k128_d128_c1}}_kernel();
    if (seq_q == 1 && seq_k == 512 && head_dim == 128 && !causal) kernel = &attention_q1_k512_d128_c0_{{HASH_attention_q1_k512_d128_c0}}_kernel();
    if (seq_q == 1 && seq_k == 512 && head_dim == 128 && causal) kernel = &attention_q1_k512_d128_c1_{{HASH_attention_q1_k512_d128_c1}}_kernel();
    if (seq_q == 1 && seq_k == 2048 && head_dim == 128 && !causal) kernel = &attention_q1_k2048_d128_c0_{{HASH_attention_q1_k2048_d128_c0}}_kernel();
    if (seq_q == 1 && seq_k == 2048 && head_dim == 128 && causal) kernel = &attention_q1_k2048_d128_c1_{{HASH_attention_q1_k2048_d128_c1}}_kernel();
    if (seq_q == 128 && seq_k == 128 && head_dim == 128 && !causal) kernel = &attention_q128_k128_d128_c0_{{HASH_attention_q128_k128_d128_c0}}_kernel();
    if (seq_q == 128 && seq_k == 128 && head_dim == 128 && causal) kernel = &attention_q128_k128_d128_c1_{{HASH_attention_q128_k128_d128_c1}}_kernel();
    if (kernel == nullptr) throw std::invalid_argument("no AOT attention specialization");
    kernel->launch(args, batch_heads, ceilDiv(seq_q, 16), 1, 4 * 32);
  }
};

void register_attention(dli::OperatorRegistry* registry) {
  registry->registerFactory("attention", [] { return std::make_unique<AttentionOp>(); });
}
