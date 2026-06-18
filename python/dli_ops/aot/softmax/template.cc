dli::CudaAotKernel& softmaxKernel(std::int64_t block) {
  switch (block) {
    case 8: return softmax_b8_{{HASH_softmax_b8}}_kernel();
    case 16: return softmax_b16_{{HASH_softmax_b16}}_kernel();
    case 32: return softmax_b32_{{HASH_softmax_b32}}_kernel();
    case 64: return softmax_b64_{{HASH_softmax_b64}}_kernel();
    case 128: return softmax_b128_{{HASH_softmax_b128}}_kernel();
    case 256: return softmax_b256_{{HASH_softmax_b256}}_kernel();
    case 512: return softmax_b512_{{HASH_softmax_b512}}_kernel();
  }
  throw std::invalid_argument("no AOT softmax specialization for block=" + std::to_string(block));
}

std::int64_t nextPowerOf2(std::int64_t value) {
  std::int64_t out = 1;
  while (out < value) out *= 2;
  return out;
}

class SoftmaxOp final : public dli::Operator {
 public:
  std::string type() const override { return "softmax"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes&, dli::ExecutionContext&) const override {
    if (inputs.size() != 1 || outputs.size() != 1) throw std::invalid_argument("softmax arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type());
    const int width = static_cast<int>(inputs[0]->shape().back());
    const int rows = static_cast<int>(inputs[0]->numel() / width);
    const auto block = nextPowerOf2(width);
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, inputs[0]->shape(), inputs[0]->deviceId());
    void* x = ptr(*inputs[0]); void* out = ptr(*outputs[0]);
    void* args[] = {&x, &out, const_cast<int*>(&rows), const_cast<int*>(&width)};
    softmaxKernel(block).launch(args, rows, 1, 1, 4 * 32);
  }
};

void register_softmax(dli::OperatorRegistry* registry) {
  registry->registerFactory("softmax", [] { return std::make_unique<SoftmaxOp>(); });
}
