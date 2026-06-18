cmake_minimum_required(VERSION 3.20)
project(dl_inference LANGUAGES CXX)

option(DLI_BUILD_TESTS "Build unit tests" ON)
option(DLI_BUILD_EXAMPLES "Build runnable examples" ON)
option(DLI_ENABLE_GRPC "Build gRPC client/server support" OFF)
option(DLI_ENABLE_TRITON_AOT "Build Triton AOT GPU operator plugin" ON)

find_package(Python3 COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dli_core SHARED
  src/attributes.cc
  src/cuda_driver.cc
  src/cuda_runtime.cc
  src/engine.cc
  src/graph.cc
  src/kv_cache.cc
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

if(CMAKE_DL_LIBS)
  target_link_libraries(dli_core PUBLIC ${CMAKE_DL_LIBS})
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
  add_executable(dli_operator_linear examples/operators/linear/main.cpp)
  target_link_libraries(dli_operator_linear PRIVATE dli_core)
  if(TARGET dli_triton_aot_ops)
    add_dependencies(dli_operator_linear dli_triton_aot_ops)
  endif()

  if(Python3_Interpreter_FOUND)
    set(DLI_ALEXNET_EXPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/examples/alexnet)
    add_custom_command(
      OUTPUT
        ${DLI_ALEXNET_EXPORT_DIR}/alexnet.dli.json
        ${DLI_ALEXNET_EXPORT_DIR}/alexnet.dli.weights.json
        ${DLI_ALEXNET_EXPORT_DIR}/alexnet.dli.weights.bin
      COMMAND ${CMAKE_COMMAND} -E make_directory ${DLI_ALEXNET_EXPORT_DIR}
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/examples/alexnet
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE} -m dli_export.export
          --model alexnet:create_model
          --example-shape 1,3,32,32
          --output-dir ${DLI_ALEXNET_EXPORT_DIR}
          --model-type alexnet_tiny
          --stem alexnet
      DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/alexnet/alexnet.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/dli_export/export.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/dli_export/weights.py
      VERBATIM
    )
    add_custom_target(dli_alexnet_export ALL
      DEPENDS
        ${DLI_ALEXNET_EXPORT_DIR}/alexnet.dli.json
        ${DLI_ALEXNET_EXPORT_DIR}/alexnet.dli.weights.json
        ${DLI_ALEXNET_EXPORT_DIR}/alexnet.dli.weights.bin
    )

    set(DLI_QWEN2_EXPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/examples/qwen2)
    add_custom_command(
      OUTPUT
        ${DLI_QWEN2_EXPORT_DIR}/qwen2.dli.json
        ${DLI_QWEN2_EXPORT_DIR}/qwen2.dli.weights.json
        ${DLI_QWEN2_EXPORT_DIR}/qwen2.dli.weights.bin
      COMMAND ${CMAKE_COMMAND} -E make_directory ${DLI_QWEN2_EXPORT_DIR}
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE} -m dli_export.export
          --model qwen2:create_model
          --output-dir ${DLI_QWEN2_EXPORT_DIR}
          --model-type qwen2
          --stem qwen2
      DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2/qwen2.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/dli_export/export.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/dli_export/weights.py
      VERBATIM
    )
    add_custom_target(dli_qwen2_export ALL
      DEPENDS
        ${DLI_QWEN2_EXPORT_DIR}/qwen2.dli.json
        ${DLI_QWEN2_EXPORT_DIR}/qwen2.dli.weights.json
        ${DLI_QWEN2_EXPORT_DIR}/qwen2.dli.weights.bin
    )
  endif()

  add_executable(dli_alexnet examples/alexnet/main.cpp)
  target_link_libraries(dli_alexnet PRIVATE dli_core)
  if(TARGET dli_alexnet_export)
    add_dependencies(dli_alexnet dli_alexnet_export)
  endif()
  if(TARGET dli_triton_aot_ops)
    add_dependencies(dli_alexnet dli_triton_aot_ops)
  endif()

  add_executable(dli_qwen2 examples/qwen2/main.cpp)
  target_link_libraries(dli_qwen2 PRIVATE dli_core)
  if(TARGET dli_qwen2_export)
    add_dependencies(dli_qwen2 dli_qwen2_export)
  endif()
  if(TARGET dli_triton_aot_ops)
    add_dependencies(dli_qwen2 dli_triton_aot_ops)
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

  add_executable(dli_tests tests/test_main.cc)
  target_link_libraries(dli_tests PRIVATE dli_core)
  if(TARGET dli_triton_aot_ops)
    add_dependencies(dli_tests dli_triton_aot_ops)
    set(DLI_TEST_PLUGIN_ARG ${DLI_AOT_PLUGIN})
  else()
    set(DLI_TEST_PLUGIN_ARG "")
  endif()
  add_test(NAME dli_tests COMMAND dli_tests --plugin ${DLI_TEST_PLUGIN_ARG})

  add_executable(dli_operator_engine_runner tests/operator_engine_runner.cc)
  target_link_libraries(dli_operator_engine_runner PRIVATE dli_core)
  if(TARGET dli_triton_aot_ops)
    add_dependencies(dli_operator_engine_runner dli_triton_aot_ops)
  endif()

  if(Python3_Interpreter_FOUND)
    add_test(
      NAME dli_triton_aot_operator_specs
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/build_triton_aot_ops.py --list
    )

    add_test(
      NAME dli_export_linear
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_export_linear.py
    )
    set_tests_properties(dli_export_linear PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python;PYTHONDONTWRITEBYTECODE=1"
    )

    add_test(
      NAME dli_export_alexnet
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_export_alexnet.py
    )
    set_tests_properties(dli_export_alexnet PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/examples/alexnet;PYTHONDONTWRITEBYTECODE=1"
    )

    add_test(
      NAME dli_export_qwen2
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_export_qwen2.py
    )
    set_tests_properties(dli_export_qwen2 PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/examples/qwen2;PYTHONDONTWRITEBYTECODE=1"
    )

    if(TARGET dli_triton_aot_ops)
      add_test(
        NAME dli_aot_operator_numerics
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_aot_operator_numerics.py
      )
      set_tests_properties(dli_aot_operator_numerics PROPERTIES
        ENVIRONMENT "DLI_OP_TEST_RUNNER=$<TARGET_FILE:dli_operator_engine_runner>;DLI_AOT_PLUGIN=${DLI_AOT_PLUGIN};PYTHONDONTWRITEBYTECODE=1"
      )

      add_test(
        NAME dli_no_python_runtime_link
        COMMAND ${CMAKE_COMMAND}
          -DDLI_CORE=$<TARGET_FILE:dli_core>
          -DDLI_PLUGIN=${DLI_AOT_PLUGIN}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_no_libpython.cmake
      )
    endif()
  endif()
endif()
