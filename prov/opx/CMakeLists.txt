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

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|riscv64)$")
  message(
    STATUS "OPX provider: disabled (only supported on x86_64 and riscv64)"
  )
  return()
endif()

# Check for required HFI1 headers
include(CheckIncludeFile)
include(CheckCSourceCompiles)

# Check for rdma/hfi/hfi1_user.h
check_include_file("rdma/hfi/hfi1_user.h" OPX_HAVE_HFI1_USER_H)
if(NOT OPX_HAVE_HFI1_USER_H)
  message(STATUS "OPX provider: disabled (rdma/hfi/hfi1_user.h not found)")
  return()
endif()

# Check for HFI1_CAP_TID_RDMA in hfi1_user.h
check_c_source_compiles(
  "
  #include <rdma/hfi/hfi1_user.h>
  #ifndef HFI1_CAP_TID_RDMA
  #error HFI1_CAP_TID_RDMA not defined
  #endif
  int main() { return 0; }
"
  OPX_HAVE_HFI1_CAP_TID_RDMA
)

if(NOT OPX_HAVE_HFI1_CAP_TID_RDMA)
  message(
    STATUS
      "OPX provider: disabled (HFI1_CAP_TID_RDMA not defined in hfi1_user.h)"
  )
  return()
endif()

# Check for hfi1 direct verbs support (optional)
check_include_file("infiniband/hfi1dv.h" OPX_HAVE_HFI1DV_H)

# Check for hfisvc support (optional, requires hfi1dv)
check_include_file("infiniband/hfisvc_client.h" OPX_HAVE_HFISVC_H)

# hfisvc requires hfi1dv
if(OPX_HAVE_HFISVC_H AND NOT OPX_HAVE_HFI1DV_H)
  set(OPX_HAVE_HFISVC_H FALSE)
endif()

# Check for hfi1_status_v2 (CN5000/JKR support)
check_c_source_compiles(
  "
  #include <rdma/hfi/hfi1_user.h>
  int main() { struct hfi1_status_v2 s; return 0; }
"
  OPX_HAVE_HFI1_STATUS_V2
)

set(OPX_COMPILE_DEFINITIONS
    OPX_PROGRESS=${OPX_PROGRESS} OPX_AV=${OPX_AV} OPX_MR=${OPX_MR}
    OPX_THREAD=${OPX_THREAD} OPX_RELIABILITY=${OPX_RELIABILITY}
)

# Add feature detection defines
if(OPX_HAVE_HFI1DV_H)
  list(APPEND OPX_COMPILE_DEFINITIONS "HAVE_HFI1_DIRECT_VERBS=1")
else()
  list(APPEND OPX_COMPILE_DEFINITIONS "HAVE_HFI1_DIRECT_VERBS=0")
endif()

if(OPX_HAVE_HFISVC_H AND OPX_HAVE_HFI1DV_H)
  list(APPEND OPX_COMPILE_DEFINITIONS "HAVE_HFISVC=1")
else()
  list(APPEND OPX_COMPILE_DEFINITIONS "HAVE_HFISVC=0")
endif()

if(OPX_HAVE_HFI1_STATUS_V2)
  list(APPEND OPX_COMPILE_DEFINITIONS "OPX_JKR_SUPPORT")
endif()

# HMEM (GPU) support - enabled if CUDA or ROCr is available
if(HAVE_CUDA OR HAVE_ROCR)
  # Check for struct sdma_req_meminfo in hfi1_user.h (required for GPU support)
  check_c_source_compiles(
    "
    #include <rdma/hfi/hfi1_user.h>
    int main() { struct sdma_req_meminfo m; return 0; }
  "
    OPX_HAVE_SDMA_REQ_MEMINFO
  )

  if(OPX_HAVE_SDMA_REQ_MEMINFO)
    list(APPEND OPX_COMPILE_DEFINITIONS "OPX_HMEM")
    message(STATUS "OPX provider: GPU (HMEM) support enabled")
  else()
    message(
      STATUS "OPX provider: GPU support disabled (sdma_req_meminfo not found)"
    )
  endif()
endif()

set(OPX_SOURCES
    src/fi_opx_atomic.c
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
    src/fi_opx_ep.c
    src/fi_opx_eq.c
    src/fi_opx_fabric.c
    src/fi_opx_hfi1.c
    src/fi_opx_hfi1_jkr.c
    src/fi_opx_hfi1_sdma.c
    src/fi_opx_hfi1_wfr.c
    src/fi_opx_hfi_select.c
    src/fi_opx_info.c
    src/fi_opx_init.c
    src/fi_opx_mr.c
    src/fi_opx_msg.c
    src/fi_opx_progress.c
    src/fi_opx_reliability.c
    src/fi_opx_rma.c
    src/fi_opx_sep.c
    src/fi_opx_service.c
    src/fi_opx_shm.c
    src/fi_opx_sysfs.c
    src/fi_opx_tagged.c
    src/fi_opx_tid_cache.c
    src/fi_opx_tid_domain.c
    src/opa_proto.c
    src/opa_utils_gen1.c
    src/opx_debug.c
    src/opx_hfi1_rdma_core.c
    src/opx_hfisvc.c
    src/opx_hmem_cache.c
    src/opx_hmem_domain.c
    src/opx_ipc.c
    src/opx_tracer.c
)

set(OPX_LINK_LIBRARIES "")
if(Numa_FOUND)
  list(APPEND OPX_LINK_LIBRARIES Numa::Numa)
endif()
if(UUID_FOUND)
  list(APPEND OPX_LINK_LIBRARIES UUID::UUID)
endif()

# ROCr requires linking to amdhip64
if(HAVE_ROCR AND OPX_HAVE_SDMA_REQ_MEMINFO)
  list(APPEND OPX_LINK_LIBRARIES "-L/opt/rocm/lib" "-lamdhip64")
endif()

libfabric_add_provider(
  NAME opx
  SOURCES ${OPX_SOURCES}
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/include/linux-i386 /usr/include/uapi
  COMPILE_DEFINITIONS ${OPX_COMPILE_DEFINITIONS}
  LINK_LIBRARIES ${OPX_LINK_LIBRARIES}
  DEPENDS Numa UUID
  LINUX_ONLY
)
