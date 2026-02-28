# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CMakeDependentOption)

# -----------------------------------------------------------------------------
# Core Build Options
# -----------------------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)

option(LIBFABRIC_EMBEDDED "Build for embedded use (disables symbol versioning)"
       OFF
)

option(LIBFABRIC_ENABLE_PROFILE
       "Enable profiling support (enables fi_profile hooks)" OFF
)

option(LIBFABRIC_ENABLE_LTO
       "Enable Link Time Optimization (LTO/IPO) if supported" ON
)

set(LIBFABRIC_BUILD_ID
    ""
    CACHE STRING "Build ID annotation string"
)

set(LIBFABRIC_DIRECT
    ""
    CACHE STRING
          "Build single provider in direct mode (provider name or empty)"
)

# -----------------------------------------------------------------------------
# Compiler Warning and Sanitizer Options
# -----------------------------------------------------------------------------

option(LIBFABRIC_PICKY "Enable extra compiler warnings (pedantic mode)" OFF)

option(LIBFABRIC_ENABLE_ASAN
       "Enable AddressSanitizer for memory error detection" OFF
)

option(LIBFABRIC_ENABLE_TSAN "Enable ThreadSanitizer for data race detection"
       OFF
)

option(LIBFABRIC_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

option(LIBFABRIC_ENABLE_LSAN "Enable LeakSanitizer (standalone, not with ASAN)"
       OFF
)

# -----------------------------------------------------------------------------
# GPU/Accelerator Support
# -----------------------------------------------------------------------------

option(LIBFABRIC_ENABLE_CUDA "Enable CUDA GPU memory support" OFF)

cmake_dependent_option(
  LIBFABRIC_CUDA_DLOPEN "Load CUDA libraries at runtime via dlopen" ON
  "LIBFABRIC_ENABLE_CUDA" OFF
)

option(LIBFABRIC_ENABLE_ZE "Enable Intel Level Zero GPU memory support" OFF)

cmake_dependent_option(
  LIBFABRIC_ZE_DLOPEN "Load Level Zero libraries at runtime via dlopen" ON
  "LIBFABRIC_ENABLE_ZE" OFF
)

option(LIBFABRIC_ENABLE_ROCR "Enable AMD ROCr GPU memory support" OFF)

cmake_dependent_option(
  LIBFABRIC_ROCR_DLOPEN "Load ROCr libraries at runtime via dlopen" ON
  "LIBFABRIC_ENABLE_ROCR" OFF
)

option(LIBFABRIC_ENABLE_NEURON "Enable AWS Neuron accelerator support" OFF)

option(LIBFABRIC_ENABLE_SYNAPSEAI "Enable Habana SynapseAI accelerator support"
       OFF
)

option(LIBFABRIC_ENABLE_GDRCOPY "Enable NVIDIA GDRCopy support" OFF)

cmake_dependent_option(
  LIBFABRIC_GDRCOPY_DLOPEN "Load GDRCopy libraries at runtime via dlopen" ON
  "LIBFABRIC_ENABLE_GDRCOPY" OFF
)

# -----------------------------------------------------------------------------
# Memory Monitor Options
# -----------------------------------------------------------------------------

set(LIBFABRIC_DEFAULT_MONITOR
    ""
    CACHE
      STRING
      "Default memory monitor type (uffd, memhooks, kdreg2, disabled, or empty for auto)"
)
set_property(
  CACHE LIBFABRIC_DEFAULT_MONITOR PROPERTY STRINGS "" "uffd" "memhooks"
                                           "kdreg2" "disabled"
)

option(LIBFABRIC_MEMHOOKS_MONITOR "Enable memhooks memory monitor" ON)

option(LIBFABRIC_UFFD_MONITOR "Enable userfaultfd memory monitor" ON)

option(LIBFABRIC_KDREG2_MONITOR "Enable kdreg2 memory monitor" ON)

# -----------------------------------------------------------------------------
# Optional Features
# -----------------------------------------------------------------------------

option(LIBFABRIC_ENABLE_VALGRIND "Enable Valgrind annotations" OFF)

option(LIBFABRIC_ENABLE_LTTNG "Enable LTTng userspace tracing" OFF)

option(LIBFABRIC_ENABLE_LIBURING "Enable io_uring support (Linux only)" ON)

option(LIBFABRIC_ENABLE_XPMEM "Enable XPMEM cross-process memory support" ON)

option(
  LIBFABRIC_RESTRICTED_DL
  "Only look for dl providers under default location if FI_PROVIDER_PATH is not set"
  OFF
)

# -----------------------------------------------------------------------------
# Provider Enable Options Each provider FOO creates: LIBFABRIC_PROVIDER_FOO -
# Enable the provider (default: ON) LIBFABRIC_FOO_DLOPEN       - Build as
# loadable plugin (default: OFF)
# -----------------------------------------------------------------------------

# Providers with no special dependencies
foreach(
  _prov IN
  ITEMS tcp
        udp
        sockets
        rxm
        rxd
        mrail
        lnx
        sm2
)
  string(TOUPPER ${_prov} _PROV)
  option(LIBFABRIC_PROVIDER_${_PROV} "Enable ${_prov} provider" ON)
  cmake_dependent_option(
    LIBFABRIC_${_PROV}_DLOPEN "Build ${_prov} provider as loadable plugin" OFF
    "LIBFABRIC_PROVIDER_${_PROV};BUILD_SHARED_LIBS" OFF
  )
endforeach()

# Providers with external dependencies (auto-disabled if deps not found)
foreach(
  _prov IN
  ITEMS verbs
        efa
        shm
        psm2
        psm3
        opx
        cxi
        ucx
        usnic
        lpp
)
  string(TOUPPER ${_prov} _PROV)
  option(LIBFABRIC_PROVIDER_${_PROV}
         "Enable ${_prov} provider (requires external dependencies)" ON
  )
  cmake_dependent_option(
    LIBFABRIC_${_PROV}_DLOPEN "Build ${_prov} provider as loadable plugin" OFF
    "LIBFABRIC_PROVIDER_${_PROV};BUILD_SHARED_LIBS" OFF
  )
endforeach()

# Hook providers
foreach(
  _prov IN
  ITEMS perf
        trace
        profile
        monitor
        hook_debug
        hook_hmem
        dmabuf_peer_mem
)
  string(TOUPPER ${_prov} _PROV)
  option(LIBFABRIC_HOOK_${_PROV} "Enable ${_prov} hook provider" ON)
endforeach()

# -----------------------------------------------------------------------------
# PSM3 Provider Options
# -----------------------------------------------------------------------------

cmake_dependent_option(
  LIBFABRIC_PSM3_VERBS "Enable PSM3 Verbs HAL (UD QPs)" ON
  "LIBFABRIC_PROVIDER_PSM3" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_SOCKETS "Enable PSM3 Sockets HAL (TCP)" ON
  "LIBFABRIC_PROVIDER_PSM3" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_UDP "Enable PSM3 UDP support" OFF
  "LIBFABRIC_PROVIDER_PSM3;LIBFABRIC_PSM3_SOCKETS" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_RC "Enable PSM3 User Space RC QPs (experimental)" OFF
  "LIBFABRIC_PROVIDER_PSM3;LIBFABRIC_PSM3_VERBS" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_DSA "Enable PSM3 Intel DSA support" ON
  "LIBFABRIC_PROVIDER_PSM3" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_RDMA_READ "Enable PSM3 RDMA read support" ON
  "LIBFABRIC_PROVIDER_PSM3;LIBFABRIC_PSM3_RC" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_UMR_CACHE "Enable PSM3 UMR caching" ON
  "LIBFABRIC_PROVIDER_PSM3" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_HWLOC "Enable PSM3 hwloc for NIC affinity" ON
  "LIBFABRIC_PROVIDER_PSM3" OFF
)

# -----------------------------------------------------------------------------
# EFA Provider Options
# -----------------------------------------------------------------------------

cmake_dependent_option(
  LIBFABRIC_EFA_POISONING "Enable EFA memory poisoning for debugging" OFF
  "LIBFABRIC_PROVIDER_EFA" OFF
)

cmake_dependent_option(
  LIBFABRIC_EFA_UNIT_TESTS "Enable EFA unit tests (requires CMocka)" OFF
  "LIBFABRIC_PROVIDER_EFA" OFF
)

# -----------------------------------------------------------------------------
# CXI Provider Options
# -----------------------------------------------------------------------------

cmake_dependent_option(
  LIBFABRIC_CXI_CURL_DLOPEN "Load curl library at runtime via dlopen" OFF
  "LIBFABRIC_PROVIDER_CXI" OFF
)

cmake_dependent_option(
  LIBFABRIC_CXI_JSON_DLOPEN "Load json-c library at runtime via dlopen" OFF
  "LIBFABRIC_PROVIDER_CXI" OFF
)

# -----------------------------------------------------------------------------
# Other Provider Options
# -----------------------------------------------------------------------------

cmake_dependent_option(
  LIBFABRIC_LPP_THREAD_SAFE "Enable thread-safe LPP provider" ON
  "LIBFABRIC_PROVIDER_LPP" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM2_MQ_REQ_USER "Enable PSM2 MQ req user support" OFF
  "LIBFABRIC_PROVIDER_PSM2" OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM2_MQ_FP_MSG "Enable PSM2 MQ FP msg support" OFF
  "LIBFABRIC_PROVIDER_PSM2" OFF
)

# -----------------------------------------------------------------------------
# Fabtests Options
# -----------------------------------------------------------------------------

option(LIBFABRIC_BUILD_FABTESTS "Build fabtests test suite" OFF)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_BENCHMARKS "Build fabtests benchmark tests" ON
  "LIBFABRIC_BUILD_FABTESTS" OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_FUNCTIONAL "Build fabtests functional tests" ON
  "LIBFABRIC_BUILD_FABTESTS" OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_UNIT "Build fabtests unit tests" ON
  "LIBFABRIC_BUILD_FABTESTS" OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_MULTINODE "Build fabtests multinode tests" ON
  "LIBFABRIC_BUILD_FABTESTS" OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_UBERTEST "Build fabtests ubertest" ON
  "LIBFABRIC_BUILD_FABTESTS" OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_EFA_GDA
  "Build EFA GPU Direct Async (GDA) test (requires CUDA)" OFF
  "LIBFABRIC_BUILD_FABTESTS;LIBFABRIC_ENABLE_CUDA" OFF
)
