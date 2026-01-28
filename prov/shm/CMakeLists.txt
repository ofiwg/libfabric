# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# SHM provider (shared memory) Requires CMA (Cross Memory Attach) support and
# shm_open

include(CheckSymbolExists)
include(CheckCSourceCompiles)

# SHM provider is Linux-only
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  return()
endif()

# Check for CMA support (process_vm_readv) On x86_64 Linux, we assume CMA is
# available (kernel syscall) On other architectures, we check for the function
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(SHM_HAVE_CMA TRUE)
else()
  check_symbol_exists(process_vm_readv "sys/uio.h" SHM_HAVE_CMA)
endif()

if(NOT SHM_HAVE_CMA)
  message(STATUS "SHM provider: disabled (CMA support not found)")
  return()
endif()

# Check for shm_open (in libc or librt)
check_symbol_exists(shm_open "sys/mman.h" SHM_HAVE_SHM_OPEN)
if(NOT SHM_HAVE_SHM_OPEN)
  # Try with -lrt
  set(CMAKE_REQUIRED_LIBRARIES rt)
  check_symbol_exists(shm_open "sys/mman.h" SHM_HAVE_SHM_OPEN_RT)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT SHM_HAVE_SHM_OPEN_RT)
    message(STATUS "SHM provider: disabled (shm_open not found)")
    return()
  endif()
endif()

set(SHM_LINK_LIBRARIES "")
set(SHM_COMPILE_DEFINITIONS "")

# Check for DSA support
if(DSA_FOUND)
  # DSA also requires libnuma and linux/idxd.h
  check_include_file("linux/idxd.h" SHM_HAVE_IDXD_H)
  if(Numa_FOUND AND SHM_HAVE_IDXD_H)
    list(APPEND SHM_LINK_LIBRARIES DSA::DSA Numa::Numa)
    list(APPEND SHM_COMPILE_DEFINITIONS "SHM_HAVE_DSA=1")
    message(STATUS "SHM provider: DSA support enabled")
  else()
    list(APPEND SHM_COMPILE_DEFINITIONS "SHM_HAVE_DSA=0")
  endif()
else()
  list(APPEND SHM_COMPILE_DEFINITIONS "SHM_HAVE_DSA=0")
endif()

# Link to librt if needed for shm_open
if(RT_FOUND)
  list(APPEND SHM_LINK_LIBRARIES RT::RT)
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
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
  COMPILE_DEFINITIONS ${SHM_COMPILE_DEFINITIONS}
  LINK_LIBRARIES ${SHM_LINK_LIBRARIES}
  LINUX_ONLY
)
