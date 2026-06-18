class Conv2dOp final : public dli::Operator {
 public:
  std::string type() const override { return "conv2d"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext&) const override {
    if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1) throw std::invalid_argument("conv2d arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type()); requireFloat(*inputs[1], type());
    const bool has_bias = inputs.size() == 3;
    const auto stride = attrInts(attrs, "stride", {1, 1}); const auto pad = attrInts(attrs, "padding", {0, 0});
    const int n = inputs[0]->dim(0), h = inputs[0]->dim(2), w = inputs[0]->dim(3);
    const int out_c = inputs[1]->dim(0), in_c = inputs[1]->dim(1), kh = inputs[1]->dim(2), kw = inputs[1]->dim(3);
    const int out_h = (h + 2 * pad[0] - kh) / stride[0] + 1;
    const int out_w = (w + 2 * pad[1] - kw) / stride[1] + 1;
    const int total = n * out_c * out_h * out_w;
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, {n, out_c, out_h, out_w}, inputs[0]->deviceId());
    void* x = ptr(*inputs[0]); void* weight = ptr(*inputs[1]); void* bias = has_bias ? ptr(*inputs[2]) : ptr(*outputs[0]); void* out = ptr(*outputs[0]);
    int sh = stride[0], sw = stride[1], ph = pad[0], pw = pad[1];
    void* args[] = {&x, &weight, &bias, &out, const_cast<int*>(&total), const_cast<int*>(&n), const_cast<int*>(&h),
                    const_cast<int*>(&w), const_cast<int*>(&out_c), const_cast<int*>(&out_h), const_cast<int*>(&out_w),
                    &sh, &sw, &ph, &pw};
    dli::CudaAotKernel* kernel = nullptr;
    if (in_c == 1 && kh == 2 && kw == 2 && !has_bias) kernel = &conv_c1_k2x2_b0_{{HASH_conv_c1_k2x2_b0}}_kernel();
    if (in_c == 1 && kh == 2 && kw == 2 && has_bias) kernel = &conv_c1_k2x2_b1_{{HASH_conv_c1_k2x2_b1}}_kernel();
    if (in_c == 1 && kh == 3 && kw == 3 && !has_bias) kernel = &conv_c1_k3x3_b0_{{HASH_conv_c1_k3x3_b0}}_kernel();
    if (in_c == 1 && kh == 3 && kw == 3 && has_bias) kernel = &conv_c1_k3x3_b1_{{HASH_conv_c1_k3x3_b1}}_kernel();
    if (in_c == 3 && kh == 3 && kw == 3 && !has_bias) kernel = &conv_c3_k3x3_b0_{{HASH_conv_c3_k3x3_b0}}_kernel();
    if (in_c == 3 && kh == 3 && kw == 3 && has_bias) kernel = &conv_c3_k3x3_b1_{{HASH_conv_c3_k3x3_b1}}_kernel();
    if (in_c == 8 && kh == 3 && kw == 3 && !has_bias) kernel = &conv_c8_k3x3_b0_{{HASH_conv_c8_k3x3_b0}}_kernel();
    if (in_c == 8 && kh == 3 && kw == 3 && has_bias) kernel = &conv_c8_k3x3_b1_{{HASH_conv_c8_k3x3_b1}}_kernel();
    if (in_c == 16 && kh == 3 && kw == 3 && !has_bias) kernel = &conv_c16_k3x3_b0_{{HASH_conv_c16_k3x3_b0}}_kernel();
    if (in_c == 16 && kh == 3 && kw == 3 && has_bias) kernel = &conv_c16_k3x3_b1_{{HASH_conv_c16_k3x3_b1}}_kernel();
    if (kernel == nullptr) throw std::invalid_argument("no AOT conv2d specialization");
    kernel->launch(args, ceilDiv(total, 128), 1, 1, 4 * 32);
  }
};

void register_conv2d(dli::OperatorRegistry* registry) {
  registry->registerFactory("conv2d", [] { return std::make_unique<Conv2dOp>(); });
}
