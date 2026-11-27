# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# OPX provider (OmniPath Express)
#
# OPX compile-time configuration options (matching autotools configure.m4
# defaults) These can be overridden via cmake -DOPX_AV=FI_AV_MAP etc.
#
set(OPX_PROGRESS
    "FI_PROGRESS_UNSPEC"
    CACHE STRING "OPX progress mode"
)
set(OPX_AV
    "FI_AV_UNSPEC"
    CACHE STRING "OPX address vector type"
)
set(OPX_MR
    "OFI_MR_SCALABLE"
    CACHE STRING "OPX memory region mode"
)
set(OPX_THREAD
    "FI_THREAD_DOMAIN"
    CACHE STRING "OPX thread mode"
)
set(OPX_RELIABILITY
    "OFI_RELIABILITY_KIND_ONLOAD"
    CACHE STRING "OPX reliability mode"
)

# OPX requires Linux and x86_64 or riscv64 architecture
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  return()
endif()

if(NOT LIBFABRIC_X86_64 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES
                            "^(riscv|riscv64)$"
)
  message(
    STATUS "OPX provider: disabled (only supported on x86_64 and riscv64)"
  )
  return()
endif()

if(NOT Numa_FOUND
   OR NOT UUID_FOUND
   OR NOT HAVE_ATOMICS
)
  message(STATUS "Provider opx: requires libnuma, libuuid, and native atomics")
  return()
endif()

include(CheckIncludeFile)
include(CheckCSourceCompiles)
include(CMakePushCheckState)

# Keep alternate kernel and rdma-core headers private to OPX.
find_path(
  OPX_HFI1_INCLUDE_DIR
  NAMES rdma/hfi/hfi1_user.h
  HINTS ${OPXS_ROOT}
        ENV
        OPXS_ROOT
        ENV
        OPXS_KERNEL
  PATH_SUFFIXES include/uapi include
)
find_path(
  OPX_HFI1DV_INCLUDE_DIR
  NAMES infiniband/hfi1dv.h
  HINTS ${OPXHfiSvc_ROOT} ENV OPXHfiSvc_ROOT
  PATH_SUFFIXES include
)
if(NOT LIBFABRIC_OPX_HFISVC STREQUAL "OFF")
  find_path(
    OPX_HFISVC_INCLUDE_DIR
    NAMES infiniband/hfisvc_client.h
    HINTS ${OPXHfiSvc_ROOT} ENV OPXHfiSvc_ROOT
    PATH_SUFFIXES include
  )
endif()
mark_as_advanced(
  OPX_HFI1_INCLUDE_DIR OPX_HFI1DV_INCLUDE_DIR OPX_HFISVC_INCLUDE_DIR
)
if(NOT OPX_HFI1_INCLUDE_DIR)
  message(STATUS "Provider opx: rdma/hfi/hfi1_user.h not found")
  return()
endif()
set(OPX_INCLUDE_DIRS ${OPX_HFI1_INCLUDE_DIR})
if(OPX_HFI1DV_INCLUDE_DIR)
  list(APPEND OPX_INCLUDE_DIRS ${OPX_HFI1DV_INCLUDE_DIR})
endif()
if(OPX_HFISVC_INCLUDE_DIR AND NOT LIBFABRIC_OPX_HFISVC STREQUAL "OFF")
  list(APPEND OPX_INCLUDE_DIRS ${OPX_HFISVC_INCLUDE_DIR})
endif()
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES ${OPX_INCLUDE_DIRS})
check_c_source_compiles(
  "#include <rdma/hfi/hfi1_user.h>
   #ifndef HFI1_CAP_TID_RDMA
   #error HFI1_CAP_TID_RDMA not defined
   #endif
   int main(void) { return 0; }"
  OPX_HAVE_HFI1_CAP_TID_RDMA
)
if(NOT OPX_HAVE_HFI1_CAP_TID_RDMA)
  cmake_pop_check_state()
  message(STATUS "Provider opx: requires HFI1_CAP_TID_RDMA")
  return()
endif()

check_include_file("infiniband/hfi1dv.h" OPX_HAVE_HFI1DV_H)
set(OPX_HAVE_HFISVC FALSE)
if(OPX_HAVE_HFI1DV_H AND NOT LIBFABRIC_OPX_HFISVC STREQUAL "OFF")
  check_include_file("infiniband/hfisvc_client.h" OPX_HAVE_HFISVC_H)
  set(OPX_HAVE_HFISVC ${OPX_HAVE_HFISVC_H})
endif()
if(LIBFABRIC_OPX_HFISVC STREQUAL "ON" AND NOT OPX_HAVE_HFISVC)
  message(FATAL_ERROR "OPX hfisvc requires hfi1dv.h and hfisvc_client.h")
endif()
if(OPX_HAVE_HFISVC)
  check_c_source_compiles(
    "#include <infiniband/hfisvc_client.h>
     int main(void) {
       enum hfisvc_client_hmem_iface i = HFISVC_CLIENT_HMEM_IFACE_DMABUF;
       return (int)i;
     }"
    OPX_HAVE_HFISVC_DMABUF
  )
  check_c_source_compiles(
    "#include <infiniband/hfisvc_client.h>
     int main(void) { return HFISVC_CLIENT_COMPLETION_FLAG_CQ; }"
    OPX_HAVE_HFISVC_CLIENT_COMPLETION_FLAG_CQ
  )
endif()

check_c_source_compiles(
  "#include <rdma/hfi/hfi1_user.h>
   int main(void) { struct hfi1_status_v2 s; return 0; }"
  OPX_HAVE_HFI1_STATUS_V2
)
set(OPX_COMPILE_DEFINITIONS
    OPX_PROGRESS=${OPX_PROGRESS}
    OPX_AV=${OPX_AV}
    OPX_MR=${OPX_MR}
    OPX_THREAD=${OPX_THREAD}
    OPX_RELIABILITY=${OPX_RELIABILITY}
    HAVE_HFI1_DIRECT_VERBS=$<BOOL:${OPX_HAVE_HFI1DV_H}>
    HAVE_HFISVC=$<BOOL:${OPX_HAVE_HFISVC}>
    HAVE_HFISVC_DMABUF=$<AND:$<BOOL:${OPX_HAVE_HFISVC}>,$<BOOL:${OPX_HAVE_HFISVC_DMABUF}>>
    HAVE_HFISVC_CLIENT_COMPLETION_FLAG_CQ=$<AND:$<BOOL:${OPX_HAVE_HFISVC}>,$<BOOL:${OPX_HAVE_HFISVC_CLIENT_COMPLETION_FLAG_CQ}>>
)
if(OPX_HAVE_HFI1_STATUS_V2)
  list(APPEND OPX_COMPILE_DEFINITIONS OPX_JKR_SUPPORT)
endif()

if(HAVE_CUDA OR HAVE_ROCR)
  check_c_source_compiles(
    "#include <rdma/hfi/hfi1_user.h>
     int main(void) {
       struct sdma_req_meminfo m = {0};
       m.types = 0;
       m.context[0] = 0;
       return m.context[0] != 0;
     }"
    OPX_HAVE_SDMA_REQ_MEMINFO
  )
  check_c_source_compiles(
    "#include <sys/ioctl.h>
     #include <rdma/hfi/hfi1_ioctl.h>
     #include <rdma/rdma_user_ioctl.h>
     int main(void) {
       struct hfi1_tid_info_v3 info = {0};
       info.vaddr = 0;
       info.length = 0;
       info.tidlist = 0;
       info.tidcnt = 0;
       info.flags = 0;
       info.context = 0;
       return info.context != 0 || HFI1_IOCTL_TID_UPDATE_V3 == 0;
     }"
    OPX_HAVE_HMEM_TID
  )
  if(NOT OPX_HAVE_SDMA_REQ_MEMINFO OR NOT OPX_HAVE_HMEM_TID)
    cmake_pop_check_state()
    message(
      STATUS
        "Provider opx: HMEM requires sdma_req_meminfo and TID-update-v3 kernel APIs"
    )
    return()
  endif()
  list(APPEND OPX_COMPILE_DEFINITIONS OPX_HMEM)
  check_c_source_compiles(
    "#include <rdma/hfi/hfi1_user.h>
     #ifndef HFI1_MEMINFO_TYPE_DMABUF
     #error HFI1_MEMINFO_TYPE_DMABUF not defined
     #endif
     #ifndef HFI1_CAP_DMABUF
     #error HFI1_CAP_DMABUF not defined
     #endif
     int main(void) { return 0; }"
    OPX_HAVE_DMABUF
  )
  list(APPEND OPX_COMPILE_DEFINITIONS
       OPX_HAVE_DMABUF=$<BOOL:${OPX_HAVE_DMABUF}>
  )
endif()
cmake_pop_check_state()

# Probe the full compiler/linker contract before enabling function dispatch.
set(OPX_TARGET_CLONES FALSE)
if(NOT LIBFABRIC_OPX_TARGET_CLONES STREQUAL "OFF" AND LIBFABRIC_X86_64)
  cmake_push_check_state(RESET)
  check_c_source_compiles(
    "#include <stddef.h>
     #if defined(__clang__)
     #if !__has_attribute(target_clones)
     #error target_clones not supported
     #endif
     #elif !defined(__GNUC__) || (__GNUC__ < 6)
     #error target_clones not supported
     #endif
     __attribute__((target_clones(\"avx512f\", \"avx2\", \"sse2\", \"default\")))
     static void copy(const char *src, char *dst, size_t n) {
       for (size_t i = 0; i < n; ++i) dst[i] = src[i];
     }
     int main(void) { char src[1] = {1}, dst[1] = {0}; copy(src, dst, 1); return dst[0] != 1; }"
    OPX_HAVE_TARGET_CLONES
  )
  cmake_pop_check_state()
  set(OPX_TARGET_CLONES ${OPX_HAVE_TARGET_CLONES})
endif()
if(LIBFABRIC_OPX_TARGET_CLONES STREQUAL "ON" AND NOT OPX_TARGET_CLONES)
  message(
    FATAL_ERROR "OPX target_clones requires x86_64 compiler/linker support"
  )
endif()
list(APPEND OPX_COMPILE_DEFINITIONS
     HAVE_OPX_TARGET_CLONES=$<BOOL:${OPX_TARGET_CLONES}>
)

if(LIBFABRIC_OPX_TRACER)
  set(_opx_tracer_categories
      TX
      RX
      RELI
      SDMA
      PIO
      CQ
      MR
      TID
      PROGRESS
      HMEM
      ATOMIC
      RMA
      LOCK
  )
  string(REPLACE "," ";" _opx_tracer_requested "${LIBFABRIC_OPX_TRACER}")
  set(_opx_tracer_enabled "")
  foreach(_category IN LISTS _opx_tracer_requested)
    if(_category STREQUAL "all")
      list(APPEND _opx_tracer_enabled ${_opx_tracer_categories})
    elseif(_category IN_LIST _opx_tracer_categories)
      list(APPEND _opx_tracer_enabled ${_category})
    elseif(NOT _category MATCHES "^\\^")
      message(FATAL_ERROR "Unknown OPX tracer subsystem: ${_category}")
    endif()
  endforeach()
  foreach(_category IN LISTS _opx_tracer_requested)
    if(_category MATCHES "^\\^(.+)$")
      if(NOT CMAKE_MATCH_1 IN_LIST _opx_tracer_categories)
        message(FATAL_ERROR "Unknown OPX tracer subsystem: ${_category}")
      endif()
      list(REMOVE_ITEM _opx_tracer_enabled ${CMAKE_MATCH_1})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _opx_tracer_enabled)
  list(APPEND OPX_COMPILE_DEFINITIONS OPX_TRACER_ENABLED)
  foreach(_category IN LISTS _opx_tracer_enabled)
    list(APPEND OPX_COMPILE_DEFINITIONS OPX_TRACER_${_category})
  endforeach()
endif()

set(OPX_SOURCES
    src/fi_opx_atomic.c
    src/fi_opx_atomic_clones.c
    src/fi_opx_av.c
    src/fi_opx_cm.c
    src/fi_opx_cntr.c
    src/fi_opx_cq.c
    src/fi_opx_cq_ops_table_locking.c
    src/fi_opx_cq_ops_table_locking_8192.c
    src/fi_opx_cq_ops_table_locking_runtime.c
    src/fi_opx_cq_ops_table_non_locking.c
    src/fi_opx_cq_ops_table_non_locking_8192.c
    src/fi_opx_cq_ops_table_non_locking_runtime.c
    src/fi_opx_domain.c
    src/fi_opx_tid_domain.c
    src/opx_hmem_domain.c
    src/opx_hfisvc.c
    src/opx_hfisvc_rma.c
    src/opx_hfisvc_mr.c
    src/opx_tracer.c
    src/opx_ipc.c
    src/opx_debug.c
    src/fi_opx_ep.c
    src/fi_opx_fabric.c
    src/fi_opx_info.c
    src/fi_opx_init.c
    src/fi_opx_mr.c
    src/fi_opx_msg.c
    src/fi_opx_rma.c
    src/fi_opx_hfi1.c
    src/fi_opx_hfi1_jkr.c
    src/fi_opx_hfi1_wfr.c
    src/opx_hfi1_rdma_core.c
    src/fi_opx_hfi1_sdma.c
    src/fi_opx_reliability.c
    src/fi_opx_tagged.c
    src/fi_opx_sep.c
    src/fi_opx_sysfs.c
    src/fi_opx_service.c
    src/fi_opx_shm.c
    src/fi_opx_tid_cache.c
    src/opx_hmem_cache.c
    src/opx_shm_cache.c
    src/fi_opx_hfi_select.c
    src/opa_utils_gen1.c
    src/opa_proto.c
    src/fi_opx_progress.c
)

set(OPX_LINK_LIBRARIES "")
if(Numa_FOUND)
  list(APPEND OPX_LINK_LIBRARIES Numa::Numa)
endif()
if(UUID_FOUND)
  list(APPEND OPX_LINK_LIBRARIES UUID::UUID)
endif()

libfabric_add_provider(
  NAME opx
  SOURCES ${OPX_SOURCES}
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/linux-i386 ${OPX_INCLUDE_DIRS}
  COMPILE_DEFINITIONS ${OPX_COMPILE_DEFINITIONS}
  LINK_LIBRARIES ${OPX_LINK_LIBRARIES}
  DEPENDS Numa UUID
  LINUX_ONLY
)
