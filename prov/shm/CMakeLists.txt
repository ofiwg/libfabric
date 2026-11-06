# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# SHM provider (shared memory) Requires CMA (Cross Memory Attach) support and
# shm_open

# SHM provider is Linux-only
if(NOT LIBFABRIC_LINUX)
  return()
endif()

if(NOT HAVE_ATOMICS AND NOT HAVE_BUILTIN_MM_ATOMICS)
  message(STATUS "SHM provider: disabled (memory-order atomics not found)")
  return()
endif()

# x86_64 has syscall fallbacks; other architectures need the libc CMA API.
if(NOT LIBFABRIC_X86_64 AND NOT CMA_FOUND)
  message(STATUS "SHM provider: disabled (CMA support not found)")
  return()
endif()

set(SHM_LINK_LIBRARIES "")
set(SHM_INCLUDE_DIRS "")
set(SHM_HAVE_DSA FALSE)

# shm_open and its librt requirement are checked for the common library.
if(NOT LIBFABRIC_SHM_DSA STREQUAL "OFF" AND DSA_FOUND)
  # DSA also requires libnuma and linux/idxd.h
  check_include_file("linux/idxd.h" SHM_HAVE_IDXD_H)
  if(Numa_FOUND AND SHM_HAVE_IDXD_H)
    # libaccel-config is loaded with dlopen; only its headers are needed here.
    list(APPEND SHM_INCLUDE_DIRS ${DSA_INCLUDE_DIRS})
    list(APPEND SHM_LINK_LIBRARIES Numa::Numa)
    set(SHM_HAVE_DSA TRUE)
    message(STATUS "SHM provider: DSA support enabled")
  endif()
endif()
if(LIBFABRIC_SHM_DSA STREQUAL "ON" AND NOT SHM_HAVE_DSA)
  message(
    FATAL_ERROR
      "SHM DSA support requires accel-config headers, NUMA, and linux/idxd.h"
  )
endif()

libfabric_add_provider(
  NAME shm
  SOURCES src/smr_atomic.c
          src/smr_attr.c
          src/smr_av.c
          src/smr_cntr.c
          src/smr_comp.c
          src/smr_cq.c
          src/smr_domain.c
          src/smr_dsa.c
          src/smr_ep.c
          src/smr_fabric.c
          src/smr_init.c
          src/smr_msg.c
          src/smr_progress.c
          src/smr_rma.c
          src/smr_util.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src ${SHM_INCLUDE_DIRS}
  COMPILE_DEFINITIONS SHM_HAVE_DSA=$<BOOL:${SHM_HAVE_DSA}>
  LINK_LIBRARIES ${SHM_LINK_LIBRARIES}
  LINUX_ONLY
)
