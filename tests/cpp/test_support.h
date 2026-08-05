#pragma once

#include <ATen/ATen.h>

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace dli_test {

inline void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Fn>
void expectThrows(Fn&& fn, const std::string& message) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

inline void expectAllClose(const at::Tensor& actual, const at::Tensor& expected,
                           double atol = 1.0e-6) {
  expect(at::allclose(actual, expected, atol, 0.0), "tensor values differ");
}

inline std::filesystem::path tempPath(const std::string& stem, const std::string& extension) {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() / (stem + "_" + std::to_string(now) + extension);
}

template <typename Fn>
int run(const std::string& name, Fn&& fn) {
  try {
    fn();
    std::cout << name << " passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << name << " failed: " << error.what() << "\n";
    return 1;
  }
}

}  // namespace dli_test

