cmake_minimum_required(VERSION 3.20)
project(dl_inference LANGUAGES CXX)

option(DLI_BUILD_TESTS "Build unit tests" ON)
option(DLI_BUILD_EXAMPLES "Build runnable examples" ON)
option(DLI_ENABLE_GRPC "Build gRPC client/server support" OFF)
option(DLI_ENABLE_TRITON_AOT "Build Triton AOT GPU operator plugin" ON)
option(DLI_USE_TORCH_TENSOR "Back dli::Tensor with PyTorch/ATen tensors" ON)
option(DLI_BUILD_PYTHON_BINDINGS "Build Python bindings for the C++ graph engine" ON)

find_package(Python3 COMPONENTS Interpreter Development.Module)

if(NOT DLI_USE_TORCH_TENSOR)
  message(FATAL_ERROR "DLI_USE_TORCH_TENSOR=OFF is not supported; dli::Tensor is PyTorch/ATen-backed")
endif()

if(DLI_USE_TORCH_TENSOR)
  if(NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR "DLI_USE_TORCH_TENSOR requires Python3 to locate the installed PyTorch CMake package")
  endif()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE DLI_TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  get_filename_component(DLI_TORCH_PACKAGE_DIR "${DLI_TORCH_CMAKE_PREFIX}/../.." ABSOLUTE)
  list(APPEND CMAKE_PREFIX_PATH ${DLI_TORCH_CMAKE_PREFIX})
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND AND NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
  find_package(Torch REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dli_core SHARED
  src/aten_operator.cc
  src/attributes.cc
  src/cuda_driver.cc
  src/cuda_runtime.cc
  src/engine.cc
  src/graph.cc
  src/kv_cache.cc
  src/logging.cc
  src/operator_registry.cc
  src/tensor.cc
  src/utils.cc
  src/weights.cc
)

target_include_directories(dli_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(dli_core PUBLIC cxx_std_20)
set_target_properties(dli_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DLI_USE_TORCH_TENSOR)
  target_compile_definitions(dli_core PUBLIC DLI_USE_TORCH_TENSOR=1)
  target_include_directories(dli_core PUBLIC ${TORCH_INCLUDE_DIRS})
  target_link_libraries(dli_core PUBLIC ${TORCH_LIBRARIES})
endif()

if(CMAKE_DL_LIBS)
  target_link_libraries(dli_core PUBLIC ${CMAKE_DL_LIBS})
endif()

if(DLI_BUILD_PYTHON_BINDINGS)
  if(NOT Python3_Interpreter_FOUND OR NOT Python3_Development.Module_FOUND)
    message(FATAL_ERROR "DLI_BUILD_PYTHON_BINDINGS requires Python3 Interpreter and Development.Module")
  endif()
  find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS ${DLI_TORCH_PACKAGE_DIR}/lib
    NO_DEFAULT_PATH
  )
  if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "DLI_BUILD_PYTHON_BINDINGS requires libtorch_python from the installed PyTorch package")
  endif()

  Python3_add_library(_dli_native MODULE src/python_bindings.cc WITH_SOABI)
  target_link_libraries(_dli_native PRIVATE dli_core ${TORCH_PYTHON_LIBRARY})
  target_include_directories(_dli_native PRIVATE ${TORCH_INCLUDE_DIRS})
  target_compile_features(_dli_native PRIVATE cxx_std_20)
  set_target_properties(_dli_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/dli
    BUILD_RPATH "$<TARGET_FILE_DIR:dli_core>;${DLI_TORCH_PACKAGE_DIR}/lib"
  )
  add_custom_command(TARGET _dli_native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/dli
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_SOURCE_DIR}/python/dli/__init__.py
      ${CMAKE_CURRENT_BINARY_DIR}/python/dli/__init__.py
    VERBATIM
  )
endif()

set(DLI_AOT_OPERATOR_NAMES
  attention
  add
  conv2d
  embedding
  linear
  matmul
  max_pool2d
  mul
  reshape
  relu
  rms_norm
  rotary
  silu
  softmax
  transpose
)

set(DLI_AOT_KERNEL_SOURCES)
foreach(operator_name IN LISTS DLI_AOT_OPERATOR_NAMES)
  list(APPEND DLI_AOT_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/python/dli_ops/aot/${operator_name}/kernel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/dli_ops/aot/${operator_name}/template.cc
  )
endforeach()

if(DLI_ENABLE_TRITON_AOT AND Python3_Interpreter_FOUND)
  set(DLI_AOT_OP_DIR ${CMAKE_CURRENT_BINARY_DIR}/operators)
  set(DLI_AOT_PLUGIN ${DLI_AOT_OP_DIR}/libdli_triton_aot_ops.so)
  set(DLI_AOT_EXTRA_INCLUDE_ARGS)
  if(DLI_USE_TORCH_TENSOR)
    foreach(torch_include_dir IN LISTS TORCH_INCLUDE_DIRS)
      list(APPEND DLI_AOT_EXTRA_INCLUDE_ARGS --extra-include-dir ${torch_include_dir})
    endforeach()
  endif()
  add_custom_command(
    OUTPUT ${DLI_AOT_PLUGIN}
    COMMAND ${CMAKE_COMMAND} -E env
      TRITON_CACHE_DIR=${DLI_AOT_OP_DIR}/.triton-cache
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/build_triton_aot_ops.py
        --core-library-dir ${CMAKE_CURRENT_BINARY_DIR}
        --include-dir ${CMAKE_CURRENT_SOURCE_DIR}/include
        --output-dir ${DLI_AOT_OP_DIR}
        --library-name dli_triton_aot_ops
        --kernel-source ${CMAKE_CURRENT_SOURCE_DIR}/python/dli_ops/aot
        ${DLI_AOT_EXTRA_INCLUDE_ARGS}
    DEPENDS dli_core
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/build_triton_aot_ops.py
      ${DLI_AOT_KERNEL_SOURCES}
    VERBATIM
  )
  add_custom_target(dli_triton_aot_ops ALL DEPENDS ${DLI_AOT_PLUGIN})
elseif(DLI_ENABLE_TRITON_AOT)
  message(WARNING "Python3 interpreter not found; Triton AOT plugin will not be built")
endif()

if(DLI_BUILD_EXAMPLES)
  add_custom_target(dli_examples ALL)
  if(TARGET _dli_native)
    add_dependencies(dli_examples _dli_native)
  endif()
  if(TARGET dli_triton_aot_ops)
    add_dependencies(dli_examples dli_triton_aot_ops)
  endif()
endif()

if(DLI_ENABLE_GRPC)
  find_package(Protobuf REQUIRED)
  find_package(gRPC REQUIRED)
  set(DLI_PROTO ${CMAKE_CURRENT_SOURCE_DIR}/proto/dli/inference.proto)
  set(DLI_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${DLI_PROTO_GEN_DIR})
  add_custom_command(
    OUTPUT
      ${DLI_PROTO_GEN_DIR}/dli/inference.pb.cc
      ${DLI_PROTO_GEN_DIR}/dli/inference.pb.h
      ${DLI_PROTO_GEN_DIR}/dli/inference.grpc.pb.cc
      ${DLI_PROTO_GEN_DIR}/dli/inference.grpc.pb.h
    COMMAND protobuf::protoc
      --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
      --cpp_out=${DLI_PROTO_GEN_DIR}
      --grpc_out=${DLI_PROTO_GEN_DIR}
      --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
      ${DLI_PROTO}
    DEPENDS ${DLI_PROTO}
    VERBATIM
  )
  add_library(dli_grpc
    src/grpc/client.cc
    src/grpc/server.cc
    ${DLI_PROTO_GEN_DIR}/dli/inference.pb.cc
    ${DLI_PROTO_GEN_DIR}/dli/inference.grpc.pb.cc
  )
  target_include_directories(dli_grpc PUBLIC include ${DLI_PROTO_GEN_DIR})
  target_link_libraries(dli_grpc PUBLIC dli_core gRPC::grpc++ protobuf::libprotobuf)
  target_compile_features(dli_grpc PUBLIC cxx_std_20)
endif()

if(DLI_BUILD_TESTS)
  enable_testing()

  function(dli_add_cpp_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE dli_core)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
    add_test(NAME ${target} COMMAND ${target})
  endfunction()

  dli_add_cpp_test(dli_test_attributes tests/cpp/test_attributes.cc)
  dli_add_cpp_test(dli_test_aten_operator tests/cpp/test_aten_operator.cc)
  dli_add_cpp_test(dli_test_cuda_aot_kernel tests/cpp/test_cuda_aot_kernel.cc)
  dli_add_cpp_test(dli_test_cuda_runtime tests/cpp/test_cuda_runtime.cc)
  dli_add_cpp_test(dli_test_engine tests/cpp/test_engine.cc)
  dli_add_cpp_test(dli_test_graph tests/cpp/test_graph.cc)
  dli_add_cpp_test(dli_test_kv_cache tests/cpp/test_kv_cache.cc)
  dli_add_cpp_test(dli_test_log_message tests/cpp/test_log_message.cc)
  dli_add_cpp_test(dli_test_tensor tests/cpp/test_tensor.cc)
  dli_add_cpp_test(dli_test_utils tests/cpp/test_utils.cc)
  dli_add_cpp_test(dli_test_weights tests/cpp/test_weights.cc)

  if(TARGET dli_grpc)
    add_executable(dli_test_grpc_client tests/cpp/test_grpc_client.cc)
    target_link_libraries(dli_test_grpc_client PRIVATE dli_grpc)
    target_include_directories(dli_test_grpc_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
    add_test(NAME dli_test_grpc_client COMMAND dli_test_grpc_client)

    add_executable(dli_test_grpc_server tests/cpp/test_grpc_server.cc)
    target_link_libraries(dli_test_grpc_server PRIVATE dli_grpc)
    target_include_directories(dli_test_grpc_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
    add_test(NAME dli_test_grpc_server COMMAND dli_test_grpc_server)
  endif()

  add_executable(dli_test_operator_registry tests/cpp/test_operator_registry.cc)
  target_link_libraries(dli_test_operator_registry PRIVATE dli_core)
  target_include_directories(dli_test_operator_registry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
  if(TARGET dli_triton_aot_ops)
    add_dependencies(dli_test_operator_registry dli_triton_aot_ops)
    add_test(NAME dli_test_operator_registry COMMAND dli_test_operator_registry --plugin ${DLI_AOT_PLUGIN})
  else()
    add_test(NAME dli_test_operator_registry COMMAND dli_test_operator_registry)
  endif()

  add_executable(dli_operator_engine_runner tests/operator_engine_runner.cc)
  target_link_libraries(dli_operator_engine_runner PRIVATE dli_core)
  if(TARGET dli_triton_aot_ops)
    add_dependencies(dli_operator_engine_runner dli_triton_aot_ops)
  endif()

  if(Python3_Interpreter_FOUND)
    set(DLI_PYTHON_MODULE_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/python:${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/examples/alexnet:${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2"
    )
    set(DLI_PYTHON_NATIVE_PATH
      "${CMAKE_CURRENT_BINARY_DIR}/python:${DLI_PYTHON_MODULE_PATH}"
    )

    function(dli_add_python_module_test name script)
      add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script})
      set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${DLI_PYTHON_MODULE_PATH};PYTHONDONTWRITEBYTECODE=1"
      )
    endfunction()

    add_test(
      NAME dli_triton_aot_operator_specs
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/build_triton_aot_ops.py --list
    )

    dli_add_python_module_test(dli_py_dli_export_init tests/python/test_dli_export_init.py)
    dli_add_python_module_test(dli_py_dli_export_weights tests/python/test_dli_export_weights.py)
    dli_add_python_module_test(dli_py_dli_export_export tests/python/test_dli_export_export.py)
    dli_add_python_module_test(dli_py_dli_ops_init tests/python/test_dli_ops_init.py)
    dli_add_python_module_test(dli_py_dli_ops_spec tests/python/test_dli_ops_spec.py)
    dli_add_python_module_test(dli_py_dli_ops_aot_init tests/python/test_dli_ops_aot_init.py)
    dli_add_python_module_test(dli_py_aot_add_kernel tests/python/aot/test_add_kernel.py)
    dli_add_python_module_test(dli_py_aot_attention_kernel tests/python/aot/test_attention_kernel.py)
    dli_add_python_module_test(dli_py_aot_conv2d_kernel tests/python/aot/test_conv2d_kernel.py)
    dli_add_python_module_test(dli_py_aot_embedding_kernel tests/python/aot/test_embedding_kernel.py)
    dli_add_python_module_test(dli_py_aot_linear_kernel tests/python/aot/test_linear_kernel.py)
    dli_add_python_module_test(dli_py_aot_matmul_kernel tests/python/aot/test_matmul_kernel.py)
    dli_add_python_module_test(dli_py_aot_max_pool2d_kernel tests/python/aot/test_max_pool2d_kernel.py)
    dli_add_python_module_test(dli_py_aot_mul_kernel tests/python/aot/test_mul_kernel.py)
    dli_add_python_module_test(dli_py_aot_relu_kernel tests/python/aot/test_relu_kernel.py)
    dli_add_python_module_test(dli_py_aot_reshape_kernel tests/python/aot/test_reshape_kernel.py)
    dli_add_python_module_test(dli_py_aot_rms_norm_kernel tests/python/aot/test_rms_norm_kernel.py)
    dli_add_python_module_test(dli_py_aot_rotary_kernel tests/python/aot/test_rotary_kernel.py)
    dli_add_python_module_test(dli_py_aot_silu_kernel tests/python/aot/test_silu_kernel.py)
    dli_add_python_module_test(dli_py_aot_softmax_kernel tests/python/aot/test_softmax_kernel.py)
    dli_add_python_module_test(dli_py_aot_transpose_kernel tests/python/aot/test_transpose_kernel.py)

    if(TARGET _dli_native)
      add_test(
        NAME dli_py_dli_module
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_dli_module.py
      )
      set_tests_properties(dli_py_dli_module PROPERTIES
        ENVIRONMENT "PYTHONPATH=${DLI_PYTHON_NATIVE_PATH};PYTHONDONTWRITEBYTECODE=1"
      )
    endif()

    if(TARGET dli_triton_aot_ops)
      function(dli_add_aot_operator_numeric_test name script)
        add_test(
          NAME ${name}
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        )
        set_tests_properties(${name} PROPERTIES
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tests/operators;DLI_OP_TEST_RUNNER=$<TARGET_FILE:dli_operator_engine_runner>;DLI_AOT_PLUGIN=${DLI_AOT_PLUGIN};PYTHONDONTWRITEBYTECODE=1"
        )
      endfunction()

      dli_add_aot_operator_numeric_test(dli_aot_add_numerics tests/operators/test_add.py)
      dli_add_aot_operator_numeric_test(dli_aot_attention_numerics tests/operators/test_attention.py)
      dli_add_aot_operator_numeric_test(dli_aot_conv2d_numerics tests/operators/test_conv2d.py)
      dli_add_aot_operator_numeric_test(dli_aot_embedding_numerics tests/operators/test_embedding.py)
      dli_add_aot_operator_numeric_test(dli_aot_linear_numerics tests/operators/test_linear.py)
      dli_add_aot_operator_numeric_test(dli_aot_matmul_numerics tests/operators/test_matmul.py)
      dli_add_aot_operator_numeric_test(dli_aot_max_pool2d_numerics tests/operators/test_max_pool2d.py)
      dli_add_aot_operator_numeric_test(dli_aot_mul_numerics tests/operators/test_mul.py)
      dli_add_aot_operator_numeric_test(dli_aot_relu_numerics tests/operators/test_relu.py)
      dli_add_aot_operator_numeric_test(dli_aot_reshape_numerics tests/operators/test_reshape.py)
      dli_add_aot_operator_numeric_test(dli_aot_rms_norm_numerics tests/operators/test_rms_norm.py)
      dli_add_aot_operator_numeric_test(dli_aot_rotary_embedding_numerics tests/operators/test_rotary_embedding.py)
      dli_add_aot_operator_numeric_test(dli_aot_silu_numerics tests/operators/test_silu.py)
      dli_add_aot_operator_numeric_test(dli_aot_softmax_numerics tests/operators/test_softmax.py)
      dli_add_aot_operator_numeric_test(dli_aot_transpose_numerics tests/operators/test_transpose.py)

      add_test(
        NAME dli_no_python_runtime_link
        COMMAND ${CMAKE_COMMAND}
          -DDLI_CORE=$<TARGET_FILE:dli_core>
          -DDLI_PLUGIN=${DLI_AOT_PLUGIN}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_no_libpython.cmake
      )

      if(TARGET _dli_native)
        add_test(
          NAME dli_e2e_alexnet
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/test_models.py --model alexnet
        )
        set_tests_properties(dli_e2e_alexnet PROPERTIES
          ENVIRONMENT "PYTHONPATH=${DLI_PYTHON_NATIVE_PATH};DLI_AOT_PLUGIN=${DLI_AOT_PLUGIN};PYTHONDONTWRITEBYTECODE=1"
        )

        add_test(
          NAME dli_e2e_qwen2
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/test_models.py --model qwen2
        )
        set_tests_properties(dli_e2e_qwen2 PROPERTIES
          ENVIRONMENT "PYTHONPATH=${DLI_PYTHON_NATIVE_PATH};DLI_AOT_PLUGIN=${DLI_AOT_PLUGIN};PYTHONDONTWRITEBYTECODE=1"
        )
      endif()
    endif()
  endif()
endif()
