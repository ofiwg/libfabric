# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# HMEM (heterogeneous memory) hook
libfabric_add_hook_provider(
  NAME hook_hmem
  SOURCES src/hook_hmem.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
)
