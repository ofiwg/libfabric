# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Monitor hook (Unix only - uses fstab.h and statfs)
libfabric_add_hook_provider(
  NAME monitor
  CONDITION "NOT LIBFABRIC_WINDOWS"
  SOURCES src/hook_monitor.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
)
