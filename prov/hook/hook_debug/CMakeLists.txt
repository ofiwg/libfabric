# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Debug hook
libfabric_add_hook_provider(
  NAME hook_debug
  SOURCES src/hook_debug.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
)
