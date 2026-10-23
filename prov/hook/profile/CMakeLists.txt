# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Profile hook
libfabric_add_provider(
  HOOK
  NAME profile
  SOURCES src/hook_profile.c src/prof_report.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
)
