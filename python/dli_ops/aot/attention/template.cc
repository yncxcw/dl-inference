class AttentionOp final : public dli::Operator {
 public:
  std::string type() const override { return "attention"; }
  void compute(const std::vector<const dli::Tensor*>& raw_inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext& context) const override {
    if (raw_inputs.size() != 3 || outputs.size() != 1) throw std::invalid_argument("attention arity");
    requireCudaInputs(raw_inputs, type());
    for (const auto* input : raw_inputs) requireFloat(*input, type());

    const auto validate_shapes = [](const dli::Tensor& q, const dli::Tensor& k,
                                    const dli::Tensor& v) {
      if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4) {
        throw std::invalid_argument("attention expects rank-4 [batch, heads, sequence, head_dim] tensors");
      }
      if (k.shape() != v.shape()) throw std::invalid_argument("attention key/value shape mismatch");
      for (const auto* tensor : {&q, &k, &v}) {
        for (std::size_t dim = 0; dim < tensor->rank(); ++dim) {
          if (tensor->dim(dim) <= 0 ||
              tensor->dim(dim) > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("attention dimensions must be positive runtime int values");
          }
        }
      }
      if (q.dim(0) != k.dim(0)) throw std::invalid_argument("attention batch mismatch");
      if (q.dim(3) != k.dim(3)) throw std::invalid_argument("attention head_dim mismatch");
      if (q.dim(1) % k.dim(1) != 0) {
        throw std::invalid_argument("attention query heads must be a multiple of KV heads");
      }
      if (q.deviceId() != k.deviceId() || q.deviceId() != v.deviceId()) {
        throw std::invalid_argument("attention input device mismatch");
      }
    };

    validate_shapes(*raw_inputs[0], *raw_inputs[1], *raw_inputs[2]);
    const bool causal = attrs.value_or<bool>("causal", false);
    if (attrs.contains("scale")) throw std::invalid_argument("AOT attention currently supports default scale only");

    std::string cache_name;
    const dli::KVCacheEntry* existing_cache = nullptr;
    if (attrs.contains("kv_cache")) cache_name = attrs.require<std::string>("kv_cache");
    if (!cache_name.empty()) {
      if (context.kv_cache == nullptr) throw std::invalid_argument("attention missing KV cache");
      existing_cache = context.kv_cache->get(cache_name);
      if (existing_cache != nullptr) {
        validate_shapes(*raw_inputs[0], existing_cache->key, existing_cache->value);
        if (existing_cache->key.dim(0) != raw_inputs[1]->dim(0) ||
            existing_cache->key.dim(1) != raw_inputs[1]->dim(1) ||
            existing_cache->key.dim(3) != raw_inputs[1]->dim(3) ||
            existing_cache->key.deviceId() != raw_inputs[1]->deviceId()) {
          throw std::invalid_argument("attention KV cache append shape or device mismatch");
        }
      }
    }

    const auto batch_heads_64 = raw_inputs[0]->dim(0) * raw_inputs[0]->dim(1);
    if (batch_heads_64 > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("attention batch-head product exceeds runtime limit");
    }
    const auto cached_seq = existing_cache == nullptr ? 0 : existing_cache->key.dim(2);
    if (cached_seq > std::numeric_limits<int>::max() - raw_inputs[1]->dim(2)) {
      throw std::invalid_argument("attention KV sequence length exceeds runtime limit");
    }
    const int batch_heads = static_cast<int>(batch_heads_64);
    const int seq_q = static_cast<int>(raw_inputs[0]->dim(2));
    const int seq_k = static_cast<int>(cached_seq + raw_inputs[1]->dim(2));
    const int q_heads = static_cast<int>(raw_inputs[0]->dim(1));
    const int kv_heads = static_cast<int>(raw_inputs[1]->dim(1));
    const int head_dim = static_cast<int>(raw_inputs[0]->dim(3));

    dli::CudaAotKernel* kernel = nullptr;
    if (seq_q == 1 && head_dim == 2 && !causal) kernel = &attention_decode_d2_c0_{{HASH_attention_decode_d2_c0}}_kernel();
    if (seq_q == 1 && head_dim == 2 && causal) kernel = &attention_decode_d2_c1_{{HASH_attention_decode_d2_c1}}_kernel();
    if (seq_q == 1 && head_dim == 128 && !causal) kernel = &attention_decode_d128_c0_{{HASH_attention_decode_d128_c0}}_kernel();
    if (seq_q == 1 && head_dim == 128 && causal) kernel = &attention_decode_d128_c1_{{HASH_attention_decode_d128_c1}}_kernel();
    if (seq_q == 1 && head_dim == 256 && !causal) kernel = &attention_decode_d256_c0_{{HASH_attention_decode_d256_c0}}_kernel();
    if (seq_q == 1 && head_dim == 256 && causal) kernel = &attention_decode_d256_c1_{{HASH_attention_decode_d256_c1}}_kernel();
    if (seq_q == 2 && seq_k == 2 && head_dim == 2 && !causal) kernel = &attention_q2_k2_d2_c0_{{HASH_attention_q2_k2_d2_c0}}_kernel();
    if (seq_q == 2 && seq_k == 2 && head_dim == 2 && causal) kernel = &attention_q2_k2_d2_c1_{{HASH_attention_q2_k2_d2_c1}}_kernel();
    if (seq_q == 128 && seq_k == 128 && head_dim == 128 && !causal) kernel = &attention_q128_k128_d128_c0_{{HASH_attention_q128_k128_d128_c0}}_kernel();
    if (seq_q == 128 && seq_k == 128 && head_dim == 128 && causal) kernel = &attention_q128_k128_d128_c1_{{HASH_attention_q128_k128_d128_c1}}_kernel();
    if (kernel == nullptr) throw std::invalid_argument("no AOT attention specialization");

    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, raw_inputs[0]->shape(), raw_inputs[0]->deviceId());
    std::vector<const dli::Tensor*> inputs = raw_inputs;
    if (!cache_name.empty()) {
      context.kv_cache->append(cache_name, *raw_inputs[1], *raw_inputs[2]);
      const auto* entry = context.kv_cache->get(cache_name);
      inputs[1] = &entry->key;
      inputs[2] = &entry->value;
    }
    void* q = ptr(*inputs[0]); void* k = ptr(*inputs[1]); void* v = ptr(*inputs[2]); void* out = ptr(*outputs[0]);
    void* triton_scratch = nullptr;
    int runtime_seq_q = seq_q;
    int runtime_seq_k = seq_k;
    void* fixed_args[] = {&q, &k, &v, &out, const_cast<int*>(&q_heads), const_cast<int*>(&kv_heads), &triton_scratch};
    void* decode_args[] = {&q, &k, &v, &out, &runtime_seq_q, &runtime_seq_k,
                           const_cast<int*>(&q_heads), const_cast<int*>(&kv_heads), &triton_scratch};
    kernel->launch(seq_q == 1 ? decode_args : fixed_args, batch_heads, ceilDiv(seq_q, 16), 1, 4 * 32);
  }
};

void register_attention(dli::OperatorRegistry* registry) {
  registry->registerFactory("attention", [] { return std::make_unique<AttentionOp>(); });
}
