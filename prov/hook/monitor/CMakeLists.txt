# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Monitor hook (Unix only - uses fstab.h and statfs)
if(WIN32)
  return()
endif()
libfabric_add_provider(
  HOOK
  NAME monitor
  SOURCES src/hook_monitor.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
)
