# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckSymbolExists)

# Try pkg-config first (preferred for rdma-core)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_IBVERBS QUIET IMPORTED_TARGET libibverbs)
endif()

if(_IBVERBS_FOUND)
  set(IBVerbs_FOUND TRUE)
  set(IBVerbs_VERSION "${_IBVERBS_VERSION}")
  set(IBVerbs_INCLUDE_DIRS "${_IBVERBS_INCLUDE_DIRS}")
  set(IBVerbs_LIBRARIES "${_IBVERBS_LIBRARIES}")
  set(IBVerbs_LIBRARY_DIRS "${_IBVERBS_LIBRARY_DIRS}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET IBVerbs::IBVerbs)
    add_library(IBVerbs::IBVerbs ALIAS PkgConfig::_IBVERBS)
  endif()
else()
  # Fallback: manual search
  find_path(
    IBVerbs_INCLUDE_DIR
    NAMES infiniband/verbs.h
    HINTS ${IBVerbs_ROOT} ENV IBVerbs_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    IBVerbs_LIBRARY
    NAMES ibverbs
    HINTS ${IBVerbs_ROOT} ENV IBVerbs_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    IBVerbs REQUIRED_VARS IBVerbs_LIBRARY IBVerbs_INCLUDE_DIR
  )

  if(IBVerbs_FOUND)
    set(IBVerbs_INCLUDE_DIRS "${IBVerbs_INCLUDE_DIR}")
    set(IBVerbs_LIBRARIES "${IBVerbs_LIBRARY}")

    if(NOT TARGET IBVerbs::IBVerbs)
      add_library(IBVerbs::IBVerbs UNKNOWN IMPORTED)
      set_target_properties(
        IBVerbs::IBVerbs
        PROPERTIES IMPORTED_LOCATION "${IBVerbs_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${IBVerbs_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

if(IBVerbs_FOUND)
  include(CheckCSourceCompiles)
  include(CMakePushCheckState)

  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${IBVerbs_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${IBVerbs_LIBRARIES})

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

mark_as_advanced(IBVerbs_INCLUDE_DIR IBVerbs_LIBRARY)
