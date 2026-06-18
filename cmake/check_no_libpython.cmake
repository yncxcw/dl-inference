if(NOT DEFINED DLI_CORE OR NOT DEFINED DLI_PLUGIN)
  message(FATAL_ERROR "DLI_CORE and DLI_PLUGIN are required")
endif()

execute_process(
  COMMAND ldd ${DLI_CORE}
  RESULT_VARIABLE core_result
  OUTPUT_VARIABLE core_output
  ERROR_VARIABLE core_error
)
if(NOT core_result EQUAL 0)
  message(FATAL_ERROR "ldd failed for dli_core: ${core_error}")
endif()

execute_process(
  COMMAND ldd ${DLI_PLUGIN}
  RESULT_VARIABLE plugin_result
  OUTPUT_VARIABLE plugin_output
  ERROR_VARIABLE plugin_error
)
if(NOT plugin_result EQUAL 0)
  message(FATAL_ERROR "ldd failed for AOT plugin: ${plugin_error}")
endif()

string(FIND "${core_output}" "libpython" core_python)
string(FIND "${plugin_output}" "libpython" plugin_python)
if(NOT core_python EQUAL -1)
  message(FATAL_ERROR "dli_core links libpython:\n${core_output}")
endif()
if(NOT plugin_python EQUAL -1)
  message(FATAL_ERROR "AOT plugin links libpython:\n${plugin_output}")
endif()

message(STATUS "No libpython dependency found in dli_core or AOT plugin")
