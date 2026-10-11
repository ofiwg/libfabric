# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
CheckRdmaCM
-----------

Validate the RDMA connection-manager API.

Requires RdmaCM and IBVerbs discovery; sets RdmaCM_HAVE_* and clears
RdmaCM_FOUND when the baseline API cannot be linked.

Unguarded: repeated callers need results in their own scope. CMake caches
individual probes internally. Required-check state is saved and restored.
This module is build-only and is not installed for consumers.
]]
include(CheckCSourceCompiles)
include(CheckSymbolExists)
include(CMakePushCheckState)

if(RdmaCM_FOUND AND IBVerbs_FOUND)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${RdmaCM_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES RdmaCM::RdmaCM IBVerbs::IBVerbs)

  check_symbol_exists(rdma_create_qp "rdma/rdma_cma.h" RdmaCM_HAVE_CREATE_QP)
  if(NOT RdmaCM_HAVE_CREATE_QP)
    set(RdmaCM_FOUND FALSE)
    cmake_pop_check_state()
    return()
  endif()

  # Check for rdma_establish (modern rdma-core)
  check_symbol_exists(
    rdma_establish "rdma/rdma_cma.h" _RdmaCM_HAVE_RDMA_ESTABLISH
  )
  if(_RdmaCM_HAVE_RDMA_ESTABLISH)
    set(RdmaCM_HAVE_RDMA_ESTABLISH TRUE)
  else()
    set(RdmaCM_HAVE_RDMA_ESTABLISH FALSE)
  endif()

  # Check for rdma_create_qp_ex (required for XRC support)
  check_symbol_exists(
    rdma_create_qp_ex "rdma/rdma_cma.h" _RdmaCM_HAVE_CREATE_QP_EX
  )
  if(_RdmaCM_HAVE_CREATE_QP_EX)
    set(RdmaCM_HAVE_CREATE_QP_EX TRUE)
  else()
    set(RdmaCM_HAVE_CREATE_QP_EX FALSE)
  endif()

  cmake_pop_check_state()
endif()
