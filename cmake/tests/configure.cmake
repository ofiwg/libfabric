# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
cmake_minimum_required(VERSION 3.21)
function(configure expected)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${BINARY_DIR}/configure-test" -G
      "${GENERATOR}" -DLIBFABRIC_BUILD_TESTING=OFF ${ARGN}
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  if(expected STREQUAL "success" AND NOT _status EQUAL 0)
    message(FATAL_ERROR "Configuration failed:\n${_stdout}\n${_stderr}")
  elseif(expected STREQUAL "failure" AND _status EQUAL 0)
    message(FATAL_ERROR "Invalid configuration was accepted: ${ARGN}")
  endif()
endfunction()
file(REMOVE_RECURSE "${BINARY_DIR}/configure-test")
# This must fail even on machines with RDMA development packages installed.
configure(
  failure -DLIBFABRIC_PROVIDER_VERBS=ON
  -DCMAKE_DISABLE_FIND_PACKAGE_IBVerbs=TRUE
)
configure(success -DLIBFABRIC_PROVIDER_VERBS=OFF -DLIBFABRIC_PROVIDER_TCP=ON)
configure(success -DLIBFABRIC_PROVIDER_TCP=OFF)
file(READ "${BINARY_DIR}/configure-test/config.h" _config)
if(NOT _config MATCHES "#define HAVE_TCP 0")
  message(
    FATAL_ERROR "Disabling TCP retained the previous feature configuration"
  )
endif()
configure(failure -DLIBFABRIC_PROVIDER_TCP=invalid)
configure(
  failure -DLIBFABRIC_PROVIDER_TCP=ON -DBUILD_SHARED_LIBS=OFF
  -DLIBFABRIC_TCP_DLOPEN=ON
)
configure(
  failure -DBUILD_SHARED_LIBS=ON -DLIBFABRIC_TCP_DLOPEN=OFF
  -DLIBFABRIC_PERF_DLOPEN=ON
)
configure(success -DLIBFABRIC_PERF_DLOPEN=OFF)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  configure(
    success -DLIBFABRIC_ENABLE_ATOMICS=OFF
    -DCMAKE_DISABLE_FIND_PACKAGE_Epoll=ON
  )
  file(READ "${BINARY_DIR}/configure-test/config.h" _config)
  if(_config
     MATCHES
     "#define HAVE_(ATOMICS|ATOMICS_LEAST_TYPES|BUILTIN_ATOMICS|BUILTIN_MM_ATOMICS|BUILTIN_MM_INT128_ATOMICS|EPOLL) 1"
  )
    message(FATAL_ERROR "Disabling atomics/epoll retained cached features")
  endif()
  configure(
    success -DLIBFABRIC_ENABLE_ATOMICS=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_Epoll=OFF
  )
  # Exercise the older-glibc link requirement on current Linux hosts too.
  configure(
    success -DBUILD_SHARED_LIBS=ON -DLIBFABRIC_TCP_DLOPEN=OFF
    -DLIBFABRIC_HAVE_SHM_OPEN=FALSE
  )
  file(GLOB_RECURSE _pc_files "${BINARY_DIR}/configure-test/pkgconfig/*.pc")
  if(NOT _pc_files)
    message(FATAL_ERROR "No pkg-config metadata was generated")
  endif()
  foreach(_pc IN LISTS _pc_files)
    file(READ "${_pc}" _metadata)
    if(NOT _metadata MATCHES "Libs.private:[^\n]*-lrt")
      message(FATAL_ERROR "shm_open's librt requirement was not exported")
    endif()
  endforeach()
  configure(failure -DLIBFABRIC_HAVE_SHM_OPEN_RT=FALSE)

  # Selecting a target system explicitly enters cross-compilation mode, even
  # with a host compiler. Such builds must not register native execution tests.
  set(_cross_build "${BINARY_DIR}/cross-test")
  file(REMOVE_RECURSE "${_cross_build}")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" --preset minimal -S "${SOURCE_DIR}" -B
      "${_cross_build}" -G "${GENERATOR}" -DCMAKE_SYSTEM_NAME=Linux
      -DLIBFABRIC_BUILD_TESTING=ON
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  if(NOT _status EQUAL 0)
    message(
      FATAL_ERROR "Cross-mode configuration failed:\n${_stdout}\n${_stderr}"
    )
  endif()
  execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${_cross_build}"
            --show-only=json-v1
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _tests
    ERROR_VARIABLE _stderr
  )
  if(NOT _status EQUAL 0)
    message(FATAL_ERROR "Cannot enumerate cross-mode tests: ${_stderr}")
  endif()
  string(JSON _test_count LENGTH "${_tests}" tests)
  if(NOT _test_count EQUAL 0)
    message(
      FATAL_ERROR "Cross build registered native execution tests: ${_tests}"
    )
  endif()
endif()
