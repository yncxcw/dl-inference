#pragma once

#include <string>

#include "dli/engine.h"
#include "dli/tensor.h"

namespace dli {

TensorMap loadWeights(const std::string& manifest_path,
                      DeviceType device = DeviceType::Cpu,
                      int device_id = 0);

}  // namespace dli
