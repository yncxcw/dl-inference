/**
 * @file weights.h
 * @brief Loader for "dli.weights.v1" weight manifests.
 */

#ifndef DLI_WEIGHTS_H_
#define DLI_WEIGHTS_H_

#include <string>

#include "dli/engine.h"
#include "dli/tensor.h"

namespace dli {

/**
 * @brief Loads all tensors described by a weights manifest.
 *
 * The manifest is a JSON file naming a binary data file (relative paths resolve against the
 * manifest directory) and, per tensor, its dtype, shape, byte offset and byte count.
 *
 * @param manifest_path Path to the JSON manifest.
 * @param device Device on which to place the tensors.
 * @param device_id CUDA device ordinal; ignored for CPU.
 * @return Tensors keyed by name.
 * @throws std::runtime_error If the manifest or data file cannot be opened.
 * @throws std::invalid_argument If the manifest is malformed, the format is unsupported, or a
 *     tensor range or byte count is inconsistent.
 */
TensorMap loadWeights(const std::string& manifest_path, DeviceType device = DeviceType::Cpu,
                      int device_id = 0);

}  // namespace dli

#endif  // DLI_WEIGHTS_H_
