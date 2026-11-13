# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# SM2 (shared memory v2) provider
if(NOT LIBFABRIC_LINUX)
  return()
endif()
if(NOT LIBFABRIC_X86_64 AND NOT CMA_FOUND)
  message(STATUS "SM2 provider: disabled (CMA support not found)")
  return()
endif()
if(NOT HAVE_ATOMICS AND NOT HAVE_BUILTIN_MM_ATOMICS)
  message(STATUS "SM2 provider: disabled (memory-order atomics not found)")
  return()
endif()

libfabric_add_provider(
  NAME sm2
  SOURCES src/sm2_atomic.c
          src/sm2_attr.c
          src/sm2_av.c
          src/sm2_cntr.c
          src/sm2_comp.c
          src/sm2_coordination.c
          src/sm2_cq.c
          src/sm2_domain.c
          src/sm2_ep.c
          src/sm2_fabric.c
          src/sm2_init.c
          src/sm2_msg.c
          src/sm2_progress.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
  LINUX_ONLY
)
