/**
 * @file test_support.h
 * @brief Tiny assertion and runner helpers shared by the C++ unit tests.
 */

#ifndef DLI_TESTS_CPP_TEST_SUPPORT_H_
#define DLI_TESTS_CPP_TEST_SUPPORT_H_

#include <ATen/ATen.h>

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace dli_test {

/**
 * @brief Fails the test if a condition is false.
 * @param condition Condition that must hold.
 * @param message Failure description.
 * @throws std::runtime_error If @p condition is false.
 */
inline void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

/**
 * @brief Fails the test unless a callable throws a std::exception.
 * @tparam Fn Callable type.
 * @param fn Callable expected to throw.
 * @param message Failure description.
 * @throws std::runtime_error If @p fn returns normally.
 */
template <typename Fn>
void expectThrows(Fn&& fn, const std::string& message) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

/**
 * @brief Fails the test unless two tensors are element-wise equal within an absolute tolerance.
 * @param actual Computed tensor.
 * @param expected Reference tensor.
 * @param atol Absolute tolerance; the relative tolerance is zero.
 * @throws std::runtime_error If the tensors differ.
 */
inline void expectAllClose(const at::Tensor& actual, const at::Tensor& expected,
                           double atol = 1.0e-6) {
  expect(at::allclose(actual, expected, atol, 0.0), "tensor values differ");
}

/**
 * @brief Builds a unique path in the system temporary directory.
 * @param stem File name prefix.
 * @param extension File extension including the dot.
 * @return `<tmp>/<stem>_<clock ticks><extension>`.
 */
inline std::filesystem::path tempPath(const std::string& stem, const std::string& extension) {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() / (stem + "_" + std::to_string(now) + extension);
}

/**
 * @brief Runs a test body and converts exceptions into a process exit code.
 * @tparam Fn Callable type.
 * @param name Test name printed with the result.
 * @param fn Test body.
 * @return 0 if @p fn completes, 1 if it throws a std::exception.
 */
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

#endif  // DLI_TESTS_CPP_TEST_SUPPORT_H_
