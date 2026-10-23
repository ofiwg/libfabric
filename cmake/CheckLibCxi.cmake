# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
CheckLibCxi
-----------

Probe optional CXI service and communication-profile functions.

Requires LibCxi and CXIDriverHeaders discovery; sets LibCxi_HAVE_*.

Unguarded: repeated callers need results in their own scope. CMake caches
individual probes internally. Required-check state is saved and restored.
This module is build-only and is not installed for consumers.
]]
include(CheckSymbolExists)
include(CMakePushCheckState)
if(LibCxi_FOUND AND CXIDriverHeaders_FOUND)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES LibCxi::LibCxi
                               CXIDriverHeaders::CXIDriverHeaders
  )
  foreach(_feature MODIFY_CP SVC_GET_VNI_RANGE ALLOC_TRIG_CP SVC_GET_NETNS)
    string(TOLOWER "${_feature}" _symbol)
    check_symbol_exists(
      cxil_${_symbol} "string.h;libcxi/libcxi.h" LibCxi_HAVE_${_feature}
    )
  endforeach()
  cmake_pop_check_state()
endif()
