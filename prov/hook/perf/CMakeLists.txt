# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Performance monitoring hook
libfabric_add_provider(
  HOOK
  NAME perf
  SOURCES src/hook_perf.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# The core hook wrappers dispatch through the performance hook's operations.
if(TARGET libfabric_provider_perf)
  target_include_directories(
    libfabric_hooks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
endif()
