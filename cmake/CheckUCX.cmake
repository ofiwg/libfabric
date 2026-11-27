# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
CheckUCX
--------

Validate the UCX API used by the provider.

Requires UCX discovery. Sets UCX_HAVE_WORKER_FLAG_IGNORE_REQUEST_LEAK and
LIBFABRIC_UCX_MEMTYPE_API; clears UCX_FOUND for an unsupported memtype API.

Unguarded: repeated callers need results in their own scope. CMake caches
individual probes internally. Required-check state is saved and restored.
This module is build-only and is not installed for consumers.
]]
if(UCX_FOUND)
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

  set(CMAKE_REQUIRED_LIBRARIES UCX::UCX)
  check_c_source_compiles(
    "
    #include <ucs/memory/memtype_cache.h>
    int main(void) {
      ucs_memtype_cache_update(0, 0, UCS_MEMORY_TYPE_UNKNOWN, UCS_SYS_DEVICE_ID_UNKNOWN);
      return 0;
    }"
    LIBFABRIC_UCX_MEMTYPE_API
  )
  if(NOT LIBFABRIC_UCX_MEMTYPE_API)
    message(STATUS "UCX provider requires the four-argument memtype cache API")
    set(UCX_FOUND FALSE)
  endif()
  cmake_pop_check_state()
endif()
