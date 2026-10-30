# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
CheckIBVerbs
------------

Validate baseline verbs symbols and optional verbs features.

Requires IBVerbs discovery; sets IBVerbs_HAVE_* and clears IBVerbs_FOUND
when the provider's baseline API cannot be linked.

Unguarded: repeated callers need results in their own scope. CMake caches
individual probes internally. Required-check state is saved and restored.
This module is build-only and is not installed for consumers.
]]
include(CheckCSourceCompiles)
include(CheckSymbolExists)
include(CMakePushCheckState)

if(IBVerbs_FOUND)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${IBVerbs_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES IBVerbs::IBVerbs)

  check_symbol_exists(
    ibv_open_device "infiniband/verbs.h" IBVerbs_HAVE_OPEN_DEVICE
  )
  if(NOT IBVerbs_HAVE_OPEN_DEVICE)
    set(IBVerbs_FOUND FALSE)
    cmake_pop_check_state()
    return()
  endif()

  # Check for ibv_open_xrcd (XRC support)
  check_symbol_exists(ibv_open_xrcd "infiniband/verbs.h" IBVerbs_HAVE_XRC)

  # Check for IBV_QPT_XRC_SEND (XRC queue pair type)
  check_c_source_compiles(
    "
    #include <infiniband/verbs.h>
    int main() {
      int x = IBV_QPT_XRC_SEND;
      return x;
    }
  "
    IBVerbs_HAVE_XRC_SEND
  )

  # Check for ibv_query_device_ex (extended query support)
  check_symbol_exists(
    ibv_query_device_ex "infiniband/verbs.h" IBVerbs_HAVE_QUERY_EX
  )

  # Check for ibv_reg_dmabuf_mr (dmabuf support)
  check_symbol_exists(
    ibv_reg_dmabuf_mr "infiniband/verbs.h" IBVerbs_HAVE_DMABUF_MR
  )

  # Check for ibv_is_fork_initialized
  check_symbol_exists(
    ibv_is_fork_initialized "infiniband/verbs.h"
    IBVerbs_HAVE_IS_FORK_INITIALIZED
  )

  # Check for IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES
  check_c_source_compiles(
    "
    #include <infiniband/verbs.h>
    int main() {
      int x = IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES;
      return x;
    }
  "
    IBVerbs_HAVE_DATA_IN_ORDER_ALIGNED_128_BYTES
  )

  # Check for IBV_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY
  check_c_source_compiles(
    "
    #include <infiniband/verbs.h>
    int main() {
      int x = IBV_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY;
      return x;
    }
  "
    IBVerbs_HAVE_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY
  )

  # Check for ibv_create_comp_channel (CQ notification)
  check_symbol_exists(
    ibv_create_comp_channel "infiniband/verbs.h"
    IBVerbs_HAVE_CREATE_COMP_CHANNEL
  )

  # Check for ibv_get_cq_event (CQ notification)
  check_symbol_exists(
    ibv_get_cq_event "infiniband/verbs.h" IBVerbs_HAVE_GET_CQ_EVENT
  )

  cmake_pop_check_state()

  # Compute derived features
  if(IBVerbs_HAVE_CREATE_COMP_CHANNEL AND IBVerbs_HAVE_GET_CQ_EVENT)
    set(IBVerbs_HAVE_CQ_NOTIFICATION TRUE)
  else()
    set(IBVerbs_HAVE_CQ_NOTIFICATION FALSE)
  endif()
endif()
