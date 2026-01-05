# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Performance monitoring hook
libfabric_add_hook_provider(
  NAME perf
  SOURCES src/hook_perf.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
)
