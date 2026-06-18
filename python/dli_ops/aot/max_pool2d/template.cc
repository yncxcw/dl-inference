class MaxPool2dOp final : public dli::Operator {
 public:
  std::string type() const override { return "max_pool2d"; }
  void compute(const std::vector<const dli::Tensor*>& inputs,
               const std::vector<dli::Tensor*>& outputs,
               const dli::Attributes& attrs, dli::ExecutionContext&) const override {
    if (inputs.size() != 1 || outputs.size() != 1) throw std::invalid_argument("max_pool2d arity");
    requireCudaInputs(inputs, type()); requireFloat(*inputs[0], type());
    const auto kernel_size = attrInts(attrs, "kernel_size", {2, 2});
    const auto stride = attrInts(attrs, "stride", kernel_size);
    const auto pad = attrInts(attrs, "padding", {0, 0});
    const int n = inputs[0]->dim(0), c = inputs[0]->dim(1), h = inputs[0]->dim(2), w = inputs[0]->dim(3);
    const int kh = kernel_size[0], kw = kernel_size[1];
    const int out_h = (h + 2 * pad[0] - kh) / stride[0] + 1;
    const int out_w = (w + 2 * pad[1] - kw) / stride[1] + 1;
    const int total = n * c * out_h * out_w;
    *outputs[0] = dli::Tensor::cuda(dli::DType::Float32, {n, c, out_h, out_w}, inputs[0]->deviceId());
    void* x = ptr(*inputs[0]); void* out = ptr(*outputs[0]);
    int sh = stride[0], sw = stride[1], ph = pad[0], pw = pad[1];
    void* args[] = {&x, &out, const_cast<int*>(&total), const_cast<int*>(&n), const_cast<int*>(&c),
                    const_cast<int*>(&h), const_cast<int*>(&w), const_cast<int*>(&out_h),
                    const_cast<int*>(&out_w), &sh, &sw, &ph, &pw};
    dli::CudaAotKernel* kernel = nullptr;
    if (kh == 2 && kw == 2) kernel = &maxpool_k2x2_{{HASH_maxpool_k2x2}}_kernel();
    if (kh == 3 && kw == 3) kernel = &maxpool_k3x3_{{HASH_maxpool_k3x3}}_kernel();
    if (kernel == nullptr) throw std::invalid_argument("no AOT max_pool2d specialization");
    kernel->launch(args, ceilDiv(total, 128), 1, 1, 4 * 32);
  }
};

void register_max_pool2d(dli::OperatorRegistry* registry) {
  registry->registerFactory("max_pool2d", [] { return std::make_unique<MaxPool2dOp>(); });
}
