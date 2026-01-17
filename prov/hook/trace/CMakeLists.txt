# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Tracing hook
libfabric_add_hook_provider(
  NAME trace
  SOURCES src/hook_trace.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
)
