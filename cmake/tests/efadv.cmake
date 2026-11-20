# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# Exercise complete and partial EFA SDKs without requiring RDMA hardware.
cmake_minimum_required(VERSION 3.21)

foreach(
  _missing IN
  ITEMS NONE
        create_mem_comp_action
        destroy_comp_action
        get_max_inline_data
        action_block_offset
)
  set(_build "${BINARY_DIR}/efadv-test/${_missing}")
  file(REMOVE_RECURSE "${_build}")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -S "${SOURCE_DIR}/cmake/tests/efadv" -B "${_build}" -G
      "${GENERATOR}" "-DCMAKE_C_COMPILER=${C_COMPILER}"
      "-DMODULE_DIR=${SOURCE_DIR}/cmake" "-DMISSING=${_missing}"
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  if(NOT _status EQUAL 0)
    message(
      FATAL_ERROR "EFA SDK fixture ${_missing} failed:\n${_stdout}\n${_stderr}"
    )
  endif()
endforeach()
