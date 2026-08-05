#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/ATen.h>

#include <map>
#include <string>
#include <utility>

#include "dli/engine.h"
#include "dli/graph.h"
#include "dli/tensor.h"
#include "dli/weights.h"

namespace py = pybind11;

namespace dli {
namespace {

TensorMap tensorMapFromPython(const py::dict& tensors) {
  TensorMap result;
  for (const auto& item : tensors) {
    auto name = py::cast<std::string>(item.first);
    auto tensor = py::cast<at::Tensor>(item.second);
    result.emplace(std::move(name), Tensor(std::move(tensor)));
  }
  return result;
}

py::dict tensorMapToPython(const TensorMap& tensors) {
  py::dict result;
  for (const auto& [name, tensor] : tensors) {
    result[py::str(name)] = py::cast(tensor.torchTensor());
  }
  return result;
}

TensorMap runGraph(Engine& engine, const Graph& graph, const py::dict& inputs) {
  auto tensor_inputs = tensorMapFromPython(inputs);
  py::gil_scoped_release release;
  return engine.run(graph, std::move(tensor_inputs));
}

DeviceType parseDevice(const std::string& device) {
  return deviceFromString(device);
}

}  // namespace
}  // namespace dli

PYBIND11_MODULE(_dli_native, m) {
  m.doc() = "Native bindings for the DLI C++ graph engine";

  py::class_<dli::Graph>(m, "Graph")
      .def(py::init<>())
      .def_static("from_json", &dli::Graph::fromJson, py::arg("json"))
      .def_static("from_json_file", &dli::Graph::fromJsonFile, py::arg("path"))
      .def("to_json", &dli::Graph::toJson)
      .def_readwrite("format", &dli::Graph::format)
      .def_readwrite("model_type", &dli::Graph::model_type)
      .def_readwrite("weights", &dli::Graph::weights)
      .def_readwrite("inputs", &dli::Graph::inputs)
      .def_readwrite("outputs", &dli::Graph::outputs);

  py::class_<dli::Engine>(m, "Engine")
      .def(py::init<>())
      .def(
          "load_library",
          [](dli::Engine& engine, const std::string& path) -> dli::Engine& {
            engine.registry().loadLibrary(path);
            return engine;
          },
          py::arg("path"),
          py::return_value_policy::reference_internal)
      .def(
          "run",
          [](dli::Engine& engine, const dli::Graph& graph, const py::dict& inputs) {
            return dli::tensorMapToPython(dli::runGraph(engine, graph, inputs));
          },
          py::arg("graph"),
          py::arg("inputs"));

  m.def(
      "load_weights",
      [](const std::string& path, const std::string& device) {
        return dli::tensorMapToPython(dli::loadWeights(path, dli::parseDevice(device)));
      },
      py::arg("path"),
      py::arg("device") = "cpu");
}
