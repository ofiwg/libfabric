# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# Configure OPX/ROCr with synthetic complete and partial kernel SDKs, without HIP.
# Placeholder libraries exercise discovery and generation, not provider linking.
cmake_minimum_required(VERSION 3.21)

foreach(_missing NONE SDMA TID IOCTL)
  set(_root "${BINARY_DIR}/opx-test/${_missing}")
  file(REMOVE_RECURSE "${_root}")
  set(_sdk "${_root}/sdk")
  file(
    MAKE_DIRECTORY
    "${_sdk}/include/rdma/hfi"
    "${_sdk}/include/hsa"
    "${_sdk}/include/uuid"
    "${_sdk}/lib"
  )
  file(WRITE "${_sdk}/include/numa.h" "")
  file(WRITE "${_sdk}/include/uuid/uuid.h" "")
  file(WRITE "${_sdk}/include/hsa/hsa_ext_amd.h" "")
  file(WRITE "${_sdk}/lib/libnuma.a" "")
  file(WRITE "${_sdk}/lib/libuuid.a" "")
  file(WRITE "${_sdk}/include/rdma/hfi/hfi1_user.h"
       "#define HFI1_CAP_TID_RDMA 1\n"
  )
  if(NOT _missing STREQUAL "SDMA")
    file(APPEND "${_sdk}/include/rdma/hfi/hfi1_user.h"
         "struct sdma_req_meminfo { unsigned long types, context[1]; };\n"
    )
  endif()
  file(WRITE "${_sdk}/include/rdma/hfi/hfi1_ioctl.h" "")
  if(NOT _missing STREQUAL "TID")
    file(
      APPEND "${_sdk}/include/rdma/hfi/hfi1_ioctl.h"
      "struct hfi1_tid_info_v3 { unsigned long vaddr, length, tidlist, tidcnt, flags, context; };\n"
    )
  endif()
  file(WRITE "${_sdk}/include/rdma/rdma_user_ioctl.h" "")
  if(NOT _missing STREQUAL "IOCTL")
    file(APPEND "${_sdk}/include/rdma/rdma_user_ioctl.h"
         "#define HFI1_IOCTL_TID_UPDATE_V3 1\n"
    )
  endif()
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" --preset minimal -S "${SOURCE_DIR}" -B "${_root}/build"
      -G "${GENERATOR}" "-DCMAKE_C_COMPILER=${C_COMPILER}"
      -DLIBFABRIC_BUILD_TESTING=OFF -DLIBFABRIC_PROVIDER_OPX=ON
      -DLIBFABRIC_ENABLE_ROCR=ON -DLIBFABRIC_OPX_TARGET_CLONES=OFF
      -DLIBFABRIC_OPX_HFISVC=OFF -DCMAKE_DISABLE_FIND_PACKAGE_hip=TRUE
      -DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=TRUE
      "-DOPX_HFI1_INCLUDE_DIR=${_sdk}/include"
      "-DROCm_INCLUDE_DIR=${_sdk}/include" "-DNuma_INCLUDE_DIR=${_sdk}/include"
      "-DNuma_LIBRARY=${_sdk}/lib/libnuma.a"
      "-DUUID_INCLUDE_DIR=${_sdk}/include"
      "-DUUID_LIBRARY=${_sdk}/lib/libuuid.a"
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  if(_missing STREQUAL "NONE")
    if(NOT _status EQUAL 0)
      message(FATAL_ERROR "Complete OPX SDK failed:\n${_stdout}\n${_stderr}")
    endif()
  elseif(_status EQUAL 0 OR NOT _stdout MATCHES "HMEM requires")
    message(
      FATAL_ERROR
        "Missing OPX ${_missing} API was not diagnosed:\n${_stdout}\n${_stderr}"
    )
  endif()
endforeach()
