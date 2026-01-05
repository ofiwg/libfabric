# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Check basic requirements before proceeding
if(NOT LIBFABRIC_PROVIDER_PSM3)
  message(STATUS "Provider psm3: disabled by option")
  set(HAVE_PSM3
      FALSE
      CACHE INTERNAL ""
  )
  return()
endif()

if(NOT LIBFABRIC_LINUX)
  message(STATUS "Provider psm3: disabled (Linux only)")
  set(HAVE_PSM3
      FALSE
      CACHE INTERNAL ""
  )
  return()
endif()

if(NOT LIBFABRIC_X86_64)
  message(STATUS "Provider psm3: disabled (x86_64 only)")
  set(HAVE_PSM3
      FALSE
      CACHE INTERNAL ""
  )
  return()
endif()

if(PSM3_FOUND)
  # Use external libpsm3
  message(STATUS "Provider psm3: using external libpsm3")

  # Provider interface sources only (psmx3_*)
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

  libfabric_add_provider(
    NAME psm3
    SOURCES ${PSM3_PROV_SOURCES}
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    COMPILE_DEFINITIONS HAVE_PSM3_LIB HAVE_PSM3_SRC=0 PSM3_CUDA=0
    LINK_LIBRARIES PSM3::PSM3
    LINUX_ONLY X86_64_ONLY
  )

  if(HAVE_PSM3)
    message(STATUS "  PSM3: external library")
  endif()

  return()
endif()

# =============================================================================
# Bundled PSM3 source build
# =============================================================================

message(STATUS "Provider psm3: using bundled source")

# Check required dependencies for bundled build
if(NOT Numa_FOUND)
  message(STATUS "Provider psm3: disabled (libnuma not found)")
  set(HAVE_PSM3
      FALSE
      CACHE INTERNAL ""
  )
  return()
endif()

if(NOT UUID_FOUND)
  message(STATUS "Provider psm3: disabled (libuuid not found)")
  set(HAVE_PSM3
      FALSE
      CACHE INTERNAL ""
  )
  return()
endif()

# At least one HAL must be enabled
set(PSM3_HAL_COUNT 0)
set(PSM3_HALS "")

# Verbs HAL
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
  set(HAVE_PSM3
      FALSE
      CACHE INTERNAL ""
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
list(APPEND PSM3_COMPILE_DEFINITIONS HAVE_PSM3_LIB HAVE_PSM3_SRC=1)

# PSM3 requires SSE4.2 for CRC instructions
include(CheckCCompilerFlag)
check_c_compiler_flag("-msse4.2" HAVE_SSE42_FLAG)
if(HAVE_SSE42_FLAG)
  list(APPEND PSM3_COMPILE_OPTIONS "-msse4.2")
endif()

# Check for RV (Rendezvous) module support
find_package(RV QUIET)
if(RV_FOUND)
  list(APPEND PSM3_COMPILE_DEFINITIONS RNDV_MOD)
  message(STATUS "Provider psm3: RV module support enabled")
else()
  message(
    STATUS
      "Provider psm3: RV module support disabled (rv_user_ioctls.h not found)"
  )
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
    list(APPEND PSM3_COMPILE_DEFINITIONS PSM_HAVE_RDMA_READ)
  endif()
endif()

# Sockets HAL configuration
if(PSM3_HAVE_SOCKETS)
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM_SOCKETS)

  # UDP support
  if(LIBFABRIC_PSM3_UDP)
    list(APPEND PSM3_COMPILE_DEFINITIONS PSM_UDP)
  endif()
endif()

# UMR Cache
if(LIBFABRIC_PSM3_UMR_CACHE)
  list(APPEND PSM3_COMPILE_DEFINITIONS UMR_CACHE)
endif()

# DSA support
if(LIBFABRIC_PSM3_DSA AND DSA_FOUND)
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM_DSA)
  list(APPEND PSM3_LINK_LIBRARIES DSA::DSA)
endif()

# HWLOC support
if(LIBFABRIC_PSM3_HWLOC AND Hwloc_FOUND)
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM_USE_HWLOC)
  list(APPEND PSM3_LINK_LIBRARIES Hwloc::Hwloc)
endif()

# CUDA GPU support
if(LIBFABRIC_ENABLE_CUDA AND CUDAToolkit_FOUND)
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM_CUDA PSM3_CUDA=1)
  if(LIBFABRIC_CUDA_DLOPEN)
    list(APPEND PSM3_COMPILE_DEFINITIONS PSM_CUDA_DLOPEN)
  else()
    list(APPEND PSM3_LINK_LIBRARIES CUDA::cudart CUDA::cuda_driver)
  endif()
  if(LIBFABRIC_ENABLE_GDRCOPY AND GDRCopy_FOUND)
    list(APPEND PSM3_COMPILE_DEFINITIONS NVIDIA_GPU_DIRECT)
    if(LIBFABRIC_GDRCOPY_DLOPEN)
      list(APPEND PSM3_COMPILE_DEFINITIONS PSM_GDRCOPY_DLOPEN)
    else()
      list(APPEND PSM3_LINK_LIBRARIES GDRCopy::GDRCopy)
    endif()
  endif()
else()
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM3_CUDA=0)
endif()

# Level Zero/OneAPI GPU support
if(LIBFABRIC_ENABLE_ZE AND LevelZero_FOUND)
  list(APPEND PSM3_COMPILE_DEFINITIONS PSM_ONEAPI)
  if(LIBFABRIC_ZE_DLOPEN)
    list(APPEND PSM3_COMPILE_DEFINITIONS PSM_ONEAPI_DLOPEN)
  else()
    list(APPEND PSM3_LINK_LIBRARIES LevelZero::LevelZero)
  endif()
  # DRM header support for dmabuf
  if(LevelZero_HAS_DRM)
    list(APPEND PSM3_COMPILE_DEFINITIONS HAVE_DRM_I915)
  elseif(LevelZero_HAS_LIBDRM)
    list(APPEND PSM3_COMPILE_DEFINITIONS HAVE_LIBDRM_I915)
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

if(LIBFABRIC_ENABLE_CUDA AND CUDAToolkit_FOUND)
  list(APPEND PSM3_GPU_SOURCES psm3/gpu/psm_gpu_cuda.c)
endif()

if(LIBFABRIC_ENABLE_ZE AND LevelZero_FOUND)
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
    psm3/hal_verbs/verbs_spio.c
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

# Version information
set(PSM3_PROV_VER_MAJOR 2)
set(PSM3_PROV_VER_MINOR 0)
set(PSM3_PROV_VER_MAINT 0)
set(PSM3_PROV_VER_PATCH 0)
set(PSM3_IEFS_VERSION "unknown")
string(TIMESTAMP PSM3_BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S")
set(PSM3_SRC_CHECKSUM "cmake-build")

# Get git hash if available
execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE PSM3_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
  RESULT_VARIABLE _git_result
)
if(NOT _git_result EQUAL 0)
  set(PSM3_GIT_HASH "unknown")
endif()

# Generate psm3_revision.c
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/psm3_revision.c.in
  ${CMAKE_CURRENT_BINARY_DIR}/psm3_revision.c @ONLY
)
list(APPEND PSM3_ALL_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/psm3_revision.c)

# Generate psm3_src_chksum.h (empty placeholder for CMake builds)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/psm3_src_chksum.h
     "/* Generated by CMake - source checksum not computed */\n"
     "#ifndef PSM3_SRC_CHKSUM_H\n" "#define PSM3_SRC_CHKSUM_H\n" "#endif\n"
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
  CONDITION "TRUE"
)

if(HAVE_PSM3)
  message(STATUS "  PSM3: bundled source")
  message(STATUS "  PSM3 HALs: ${PSM3_HALS_STR}")
  if(LIBFABRIC_ENABLE_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "  PSM3 GPU: CUDA")
  endif()
  if(LIBFABRIC_ENABLE_ZE AND LevelZero_FOUND)
    message(STATUS "  PSM3 GPU: Level Zero")
  endif()
  if(LIBFABRIC_PSM3_DSA AND DSA_FOUND)
    message(STATUS "  PSM3 DSA: enabled")
  endif()
endif()
