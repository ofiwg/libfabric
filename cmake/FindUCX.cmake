# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_path(
  UCX_INCLUDE_DIR
  NAMES ucp/api/ucp.h
  HINTS ${UCX_ROOT} ENV UCX_ROOT
  PATH_SUFFIXES include
)

find_library(
  UCX_LIBRARY
  NAMES ucp
  HINTS ${UCX_ROOT} ENV UCX_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(UCX REQUIRED_VARS UCX_LIBRARY UCX_INCLUDE_DIR)

if(UCX_FOUND)
  set(UCX_INCLUDE_DIRS "${UCX_INCLUDE_DIR}")
  set(UCX_LIBRARIES "${UCX_LIBRARY}")

  if(NOT TARGET UCX::UCX)
    add_library(UCX::UCX UNKNOWN IMPORTED)
    set_target_properties(
      UCX::UCX PROPERTIES IMPORTED_LOCATION "${UCX_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${UCX_INCLUDE_DIR}"
    )
  endif()

  # =============================================================================
  # Feature Detection
  # =============================================================================
  include(CheckCSourceCompiles)
  include(CMakePushCheckState)

  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${UCX_INCLUDE_DIRS})

  # Check for UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK
  check_c_source_compiles(
    "
    #include <ucp/api/ucp.h>
    int main() {
      int x = UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK;
      return x;
    }
  "
    UCX_HAVE_WORKER_FLAG_IGNORE_REQUEST_LEAK
  )

  cmake_pop_check_state()
endif()

mark_as_advanced(UCX_INCLUDE_DIR UCX_LIBRARY)
