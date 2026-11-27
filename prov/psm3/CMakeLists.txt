# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Check basic requirements before proceeding
if(NOT LIBFABRIC_PROVIDER_PSM3)
  message(STATUS "Provider psm3: disabled by option")
  return()
endif()

if(NOT LIBFABRIC_LINUX)
  message(STATUS "Provider psm3: disabled (Linux only)")
  return()
endif()

if(NOT LIBFABRIC_X86_64)
  message(STATUS "Provider psm3: disabled (x86_64 only)")
  return()
endif()

# =============================================================================
# Bundled PSM3 source build
# =============================================================================

message(STATUS "Provider psm3: using bundled source")

# Check required dependencies for bundled build
if(NOT Numa_FOUND)
  message(STATUS "Provider psm3: disabled (libnuma not found)")
  return()
endif()

if(NOT UUID_FOUND)
  message(STATUS "Provider psm3: disabled (libuuid not found)")
  return()
endif()

# At least one HAL must be enabled
set(PSM3_HAL_COUNT 0)
set(PSM3_HALS "")

# Verbs HAL
if(LIBFABRIC_PSM3_VERBS AND NOT IBVerbs_FOUND)
  message(STATUS "Provider psm3: requested Verbs HAL requires libibverbs")
  return()
endif()
if(LIBFABRIC_PSM3_VERBS AND IBVerbs_FOUND)
  set(PSM3_HAVE_VERBS TRUE)
  math(EXPR PSM3_HAL_COUNT "${PSM3_HAL_COUNT} + 1")
  list(APPEND PSM3_HALS "verbs")
else()
  set(PSM3_HAVE_VERBS FALSE)
endif()

# Sockets HAL
if(LIBFABRIC_PSM3_SOCKETS)
  set(PSM3_HAVE_SOCKETS TRUE)
  math(EXPR PSM3_HAL_COUNT "${PSM3_HAL_COUNT} + 1")
  list(APPEND PSM3_HALS "sockets")
else()
  set(PSM3_HAVE_SOCKETS FALSE)
endif()

if(PSM3_HAL_COUNT EQUAL 0)
  message(
    STATUS "Provider psm3: disabled (no HAL available - need verbs or sockets)"
  )
  return()
endif()

# =============================================================================
# Configure HAL-specific sources and definitions (bundled build)
# =============================================================================

set(PSM3_COMPILE_DEFINITIONS "")
set(PSM3_COMPILE_OPTIONS "")
set(PSM3_LINK_LIBRARIES "")

# Common definitions
list(APPEND PSM3_COMPILE_DEFINITIONS HAVE_PSM3_SRC=1)

# PSM3 requires the SSE4.2 CRC intrinsic, not merely an accepted flag.
include(CheckCSourceCompiles)
include(CheckCCompilerFlag)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-msse4.2")
check_c_source_compiles(
  "#include <nmmintrin.h>
   int main(void) { return _mm_crc32_u32(0, 0) != 0; }" PSM3_HAVE_SSE42
)
cmake_pop_check_state()
if(NOT PSM3_HAVE_SSE42)
  message(STATUS "Provider psm3: requires SSE4.2 intrinsics")
  return()
endif()
list(APPEND PSM3_COMPILE_OPTIONS -msse4.2)
check_c_compiler_flag(
  -Waddress-of-packed-member PSM3_HAVE_PACKED_MEMBER_WARNING
)
if(PSM3_HAVE_PACKED_MEMBER_WARNING)
  list(APPEND PSM3_COMPILE_OPTIONS -Wno-address-of-packed-member)
endif()

# Verbs HAL configuration
if(PSM3_HAVE_VERBS)
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM_VERBS)
  list(APPEND PSM3_LINK_LIBRARIES IBVerbs::IBVerbs)

  # RC QP support
  if(LIBFABRIC_PSM3_RC)
    list(APPEND PSM3_COMPILE_DEFINITIONS USE_RC)
  endif()

  # RDMA read support
  if(LIBFABRIC_PSM3_RDMA_READ AND LIBFABRIC_PSM3_RC)
    list(APPEND PSM3_COMPILE_DEFINITIONS USE_RDMA_READ)
  endif()
endif()

# Sockets HAL configuration
if(PSM3_HAVE_SOCKETS)
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM_SOCKETS)

  # UDP support
  if(LIBFABRIC_PSM3_UDP)
    list(APPEND PSM3_COMPILE_DEFINITIONS USE_UDP)
  endif()
endif()

# UMR Cache
if(LIBFABRIC_PSM3_UMR_CACHE AND UFD_FOUND)
  list(APPEND PSM3_COMPILE_DEFINITIONS UMR_CACHE)
endif()

# PSM3 programs DSA directly through the kernel ABI; it does not use libaccel.
if(LIBFABRIC_PSM3_DSA)
  cmake_push_check_state(RESET)
  check_c_source_compiles(
    "#include <stdint.h>
     #include <linux/idxd.h>
     int main(void) {
       struct dsa_hw_desc desc = {0};
       struct dsa_completion_record c = {0};
       desc.opcode = DSA_OPCODE_MEMMOVE;
       desc.flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CC
                    | IDXD_OP_FLAG_BOF;
       desc.xfer_size = 0;
       desc.src_addr = 0;
       desc.dst_addr = 0;
       desc.completion_addr = (uintptr_t)&c;
       return c.status == DSA_COMP_NONE || c.status == DSA_COMP_SUCCESS ||
              (c.status & DSA_COMP_STATUS_MASK) == DSA_COMP_PAGE_FAULT_NOBOF ||
              (c.status & DSA_COMP_STATUS_WRITE);
     }"
    PSM3_HAVE_DSA
  )
  cmake_pop_check_state()
  if(PSM3_HAVE_DSA)
    list(APPEND PSM3_COMPILE_DEFINITIONS PSM_DSA)
  endif()
endif()

# HWLOC support
if(LIBFABRIC_PSM3_HWLOC AND Hwloc_FOUND)
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM_USE_HWLOC)
  list(APPEND PSM3_LINK_LIBRARIES Hwloc::Hwloc)
endif()

# PSM3 loads its GPU entry points itself, matching configure.m4.
if(HAVE_CUDA AND LIBFABRIC_CUDA_DLOPEN)
  set(PSM3_HAVE_CUDA TRUE)
  list(
    APPEND
    PSM3_COMPILE_DEFINITIONS
    PSM_CUDA
    NVIDIA_GPU_DIRECT
    PSM3_CUDA=1
  )
else()
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM3_CUDA=0)
endif()
if(HAVE_ZE AND LIBFABRIC_ZE_DLOPEN)
  set(PSM3_HAVE_ONEAPI TRUE)
  if(NOT LevelZero_DRM_INCLUDE_DIR)
    message(STATUS "PSM3 Level Zero support requires DRM development headers")
    return()
  endif()
  list(
    APPEND
    PSM3_COMPILE_DEFINITIONS
    PSM_ONEAPI
    INTEL_GPU_DIRECT
    PSM3_ONEAPI=1
  )
  if(LevelZero_HAS_DRM)
    list(APPEND PSM3_COMPILE_DEFINITIONS HAVE_DRM)
  else()
    list(APPEND PSM3_COMPILE_DEFINITIONS HAVE_LIBDRM)
  endif()
  cmake_push_check_state()
  check_c_source_compiles(
    "
    #include <sys/syscall.h>
    #include <unistd.h>
    int main(void) { return syscall(__NR_pidfd_getfd, 0, 0, 0); }
    "
    PSM3_HAVE_PIDFD
  )
  if(PSM3_HAVE_PIDFD)
    list(APPEND PSM3_COMPILE_DEFINITIONS PSM_HAVE_PIDFD)
  endif()
  # This backend resolves the entry point through dlsym at runtime.
  set(CMAKE_REQUIRED_INCLUDES ${LevelZero_INCLUDE_DIRS})
  check_c_source_compiles(
    "#include <level_zero/ze_api.h>
     int main(void) { return sizeof(&zeMemPutIpcHandle) == 0; }"
    PSM3_HAVE_ZE_PUT_IPCHANDLE_DECL
  )
  if(PSM3_HAVE_ZE_PUT_IPCHANDLE_DECL)
    list(APPEND PSM3_COMPILE_DEFINITIONS PSM_HAVE_ONEAPI_ZE_PUT_IPCHANDLE)
  endif()
  cmake_pop_check_state()
else()
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM3_ONEAPI=0)
endif()

# AUTO enables RV only for the HAL/GPU combinations supported by Autotools.
if(NOT LIBFABRIC_PSM3_RV STREQUAL "OFF"
   AND (PSM3_HAVE_VERBS
        OR PSM3_HAVE_CUDA
        OR LIBFABRIC_PSM3_RV STREQUAL "ON")
)
  find_path(
    RV_INCLUDE_DIR
    NAMES rdma/rv_user_ioctls.h
    HINTS ${RV_ROOT} ENV RV_ROOT
    PATH_SUFFIXES include
  )
  mark_as_advanced(RV_INCLUDE_DIR)
  if(RV_INCLUDE_DIR)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES ${RV_INCLUDE_DIR})
    check_c_source_compiles(
      "#include <sys/types.h>
       #include <stdint.h>
       #include <rdma/rv_user_ioctls.h>
       int main(void) { struct rv_ring_header ring; ring.overflow_cnt = 0; return 0; }"
      PSM3_HAVE_RV_OVERFLOW_CNT
    )
    cmake_pop_check_state()
    list(APPEND PSM3_COMPILE_DEFINITIONS RNDV_MOD)
    if(NOT PSM3_HAVE_RV_OVERFLOW_CNT)
      list(APPEND PSM3_COMPILE_DEFINITIONS HAVE_NO_PSM3_RV_OVERFLOW_CNT)
    endif()
  elseif(LIBFABRIC_PSM3_RV STREQUAL "ON")
    message(FATAL_ERROR "PSM3 RV requested but rdma/rv_user_ioctls.h not found")
  endif()
endif()

# Add required libraries
list(APPEND PSM3_LINK_LIBRARIES Numa::Numa)
list(APPEND PSM3_LINK_LIBRARIES UUID::UUID)

# =============================================================================
# Source Files
# =============================================================================

# Provider interface sources (psmx3_*)
set(PSM3_PROV_SOURCES
    src/psmx3_am.c
    src/psmx3_atomic.c
    src/psmx3_attr.c
    src/psmx3_av.c
    src/psmx3_cm.c
    src/psmx3_cntr.c
    src/psmx3_cq.c
    src/psmx3_domain.c
    src/psmx3_ep.c
    src/psmx3_fabric.c
    src/psmx3_init.c
    src/psmx3_mr.c
    src/psmx3_msg.c
    src/psmx3_rma.c
    src/psmx3_tagged.c
    src/psmx3_trx_ctxt.c
    src/psmx3_util.c
    src/psmx3_wait.c
)

# PSM3 core library sources
set(PSM3_CORE_SOURCES
    psm3/psm.c
    psm3/psm2_hal.c
    psm3/psm2_hal_loopback.c
    psm3/psm_am.c
    psm3/psm_context.c
    psm3/psm_diags.c
    psm3/psm_ep.c
    psm3/psm_ep_connect.c
    psm3/psm_error.c
    psm3/psm_memcpy.c
    psm3/psm_mock.c
    psm3/psm_mpool.c
    psm3/psm_mq.c
    psm3/psm_mq_recv.c
    psm3/psm_mq_utils.c
    psm3/psm_nic_select.c
    psm3/psm_perf.c
    psm3/psm_rndv_mod.c
    psm3/psm_stats.c
    psm3/psm_sysbuf.c
    psm3/psm_timer.c
    psm3/psm_uffd.c
    psm3/psm_utils.c
    psm3/psm_verbs_mr.c
    psm3/psmi_wrappers.c
)

# Utils sources
set(PSM3_UTILS_SOURCES
    psm3/utils/utils_debug.c
    psm3/utils/utils_dsa.c
    psm3/utils/utils_dwordcpy-x86_64.c
    psm3/utils/utils_env.c
    psm3/utils/utils_mallopt.c
    psm3/utils/utils_sysfs.c
    psm3/utils/utils_syslog.c
    psm3/utils/utils_time.c
)

# PTL AM (Active Messages / shared memory) sources
set(PSM3_PTL_AM_SOURCES psm3/ptl_am/am_reqrep.c psm3/ptl_am/am_reqrep_shmem.c
                        psm3/ptl_am/cmarwu.c psm3/ptl_am/ptl.c
)

# PTL IPS (Inter-Process / network) sources
set(PSM3_PTL_IPS_SOURCES
    psm3/ptl_ips/ips_crc32.c
    psm3/ptl_ips/ips_epstate.c
    psm3/ptl_ips/ips_opp_path_rec.c
    psm3/ptl_ips/ips_path_rec.c
    psm3/ptl_ips/ips_proto.c
    psm3/ptl_ips/ips_proto_am.c
    psm3/ptl_ips/ips_proto_connect.c
    psm3/ptl_ips/ips_proto_dump.c
    psm3/ptl_ips/ips_proto_expected.c
    psm3/ptl_ips/ips_proto_mq.c
    psm3/ptl_ips/ips_proto_recv.c
    psm3/ptl_ips/ips_recvq.c
    psm3/ptl_ips/ips_scb.c
    psm3/ptl_ips/ips_tidflow.c
    psm3/ptl_ips/ptl.c
    psm3/ptl_ips/ptl_rcvthread.c
)

# PTL self sources
set(PSM3_PTL_SELF_SOURCES psm3/ptl_self/ptl.c)

# GPU sources
set(PSM3_GPU_SOURCES psm3/gpu/psm_gpu_hal.c)

if(PSM3_HAVE_CUDA)
  list(APPEND PSM3_GPU_SOURCES psm3/gpu/psm_gpu_cuda.c)
endif()

if(PSM3_HAVE_ONEAPI)
  list(APPEND PSM3_GPU_SOURCES psm3/gpu/psm_gpu_oneapi_ze.c)
endif()

# HAL-specific sources
set(PSM3_HAL_SOURCES "")

if(PSM3_HAVE_VERBS)
  list(
    APPEND
    PSM3_HAL_SOURCES
    psm3/hal_verbs/verbs_ep.c
    psm3/hal_verbs/verbs_gdrcpy.c
    psm3/hal_verbs/verbs_hal.c
    psm3/hal_verbs/verbs_ptl_ips.c
    psm3/hal_verbs/verbs_rcvthread.c
    psm3/hal_verbs/verbs_recvhdrq.c
    psm3/hal_verbs/verbs_service.c
  )
endif()

if(PSM3_HAVE_SOCKETS)
  list(
    APPEND
    PSM3_HAL_SOURCES
    psm3/hal_sockets/sockets_ep.c
    psm3/hal_sockets/sockets_gdrcpy.c
    psm3/hal_sockets/sockets_hal.c
    psm3/hal_sockets/sockets_proto.c
    psm3/hal_sockets/sockets_ptl_ips.c
    psm3/hal_sockets/sockets_rcvthread.c
    psm3/hal_sockets/sockets_recvhdrq.c
    psm3/hal_sockets/sockets_service.c
    # Note: sockets_spio.c is #include'd by sockets_proto.c - not compiled
    # directly
  )
endif()

# Note: psm3/include/psm3_rbtree.c is a template file that gets #include'd into
# other .c files with macros pre-defined - it should NOT be compiled directly

# Combine all sources
set(PSM3_ALL_SOURCES
    ${PSM3_PROV_SOURCES}
    ${PSM3_CORE_SOURCES}
    ${PSM3_UTILS_SOURCES}
    ${PSM3_PTL_AM_SOURCES}
    ${PSM3_PTL_IPS_SOURCES}
    ${PSM3_PTL_SELF_SOURCES}
    ${PSM3_GPU_SOURCES}
    ${PSM3_HAL_SOURCES}
)

# =============================================================================
# Configure Generated Files
# =============================================================================

# Use the bundled provider version and stable metadata for source archives.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PSM3_IEFS_VERSION
     LIMIT_COUNT 1
)
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS VERSION
)
string(REPLACE "_" ";" _version_parts "${PSM3_IEFS_VERSION}")
list(GET _version_parts 0 PSM3_PROV_VER_MAJOR)
list(GET _version_parts 1 PSM3_PROV_VER_MINOR)
list(GET _version_parts 2 PSM3_PROV_VER_MAINT)
list(GET _version_parts 3 PSM3_PROV_VER_PATCH)
set(PSM3_BUILD_TIMESTAMP "<Unknown>")
if(DEFINED ENV{SOURCE_DATE_EPOCH})
  string(TIMESTAMP PSM3_BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S" UTC)
endif()
set(PSM3_SRC_CHECKSUM "<Unknown>")
set(PSM3_GIT_HASH "${LIBFABRIC_BUILD_ID}")

# Generate psm3_revision.c
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/psm3_revision.c.in
  ${CMAKE_CURRENT_BINARY_DIR}/psm3_revision.c @ONLY
)
list(APPEND PSM3_ALL_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/psm3_revision.c)

# The revision template supplies metadata when no Autotools checksum exists.
configure_file(
  "${libfabric_SOURCE_DIR}/cmake/psm3_src_chksum.h.in" psm3_src_chksum.h
  COPYONLY
)

# Determine HAL instance for single-HAL optimization
if(PSM3_HAL_COUNT EQUAL 1)
  if(PSM3_HAVE_VERBS)
    set(PSM3_HAL_INST "verbs")
  else()
    set(PSM3_HAL_INST "sockets")
  endif()
else()
  set(PSM3_HAL_INST "")
endif()

# Generate HAL inline headers PSM3_HAL_CNT is used by the template (matches
# autotools variable name)
set(PSM3_HAL_CNT ${PSM3_HAL_COUNT})
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/psm3/psm2_hal_inlines_d.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/psm2_hal_inlines_d.h @ONLY
)

# Generate psm2_hal_inlines_i.h - the template handles both single and multi-HAL
# cases via preprocessor conditionals. We just need to set PSM3_HAL_INST for the
# single-HAL optimization case.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/psm3/psm2_hal_inlines_i.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/psm2_hal_inlines_i.h @ONLY
)

# =============================================================================
# Include Directories
# =============================================================================

set(PSM3_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/include/linux-i386
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/mpspawn
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/ptl_am
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/ptl_ips
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/ptl_self
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/gpu
    # HAL directories are always included - psm_ep.h includes headers from both
    # regardless of which HAL is actually compiled
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/hal_verbs
    ${CMAKE_CURRENT_SOURCE_DIR}/psm3/hal_sockets
    ${CMAKE_CURRENT_BINARY_DIR}
)
if(RV_INCLUDE_DIR AND NOT LIBFABRIC_PSM3_RV STREQUAL "OFF")
  list(APPEND PSM3_INCLUDE_DIRS ${RV_INCLUDE_DIR})
endif()

# =============================================================================
# Register Provider
# =============================================================================

# Build HAL summary for status message
string(REPLACE ";" "+" PSM3_HALS_STR "${PSM3_HALS}")

libfabric_add_provider(
  NAME psm3
  SOURCES ${PSM3_ALL_SOURCES}
  INCLUDE_DIRS ${PSM3_INCLUDE_DIRS}
  COMPILE_DEFINITIONS ${PSM3_COMPILE_DEFINITIONS}
  COMPILE_OPTIONS ${PSM3_COMPILE_OPTIONS}
  LINK_LIBRARIES ${PSM3_LINK_LIBRARIES}
  LINUX_ONLY X86_64_ONLY
)

if(HAVE_PSM3)
  message(STATUS "  PSM3: bundled source")
  message(STATUS "  PSM3 HALs: ${PSM3_HALS_STR}")
  if(PSM3_HAVE_CUDA)
    message(STATUS "  PSM3 GPU: CUDA")
  endif()
  if(PSM3_HAVE_ONEAPI)
    message(STATUS "  PSM3 GPU: Level Zero")
  endif()
  if(LIBFABRIC_PSM3_DSA AND PSM3_HAVE_DSA)
    message(STATUS "  PSM3 DSA: enabled")
  endif()
endif()
