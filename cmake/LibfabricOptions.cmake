# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricOptions
----------------

Declare and validate user-facing build options.

Include after project(). Defines LIBFABRIC_* cache options and the transport
and hook inventories. Primary feature and provider switches remain visible;
low-level discovery paths are marked advanced by their owning modules.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()
include(CMakeDependentOption)

# -----------------------------------------------------------------------------
# Core Build Options
# -----------------------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build a shared libfabric library" ON)
option(LIBFABRIC_INSTALL "Install libfabric and export its CMake package"
       ${PROJECT_IS_TOP_LEVEL}
)
option(LIBFABRIC_BUILD_UTILITIES "Build fi_info and other utilities"
       ${PROJECT_IS_TOP_LEVEL}
)
option(LIBFABRIC_BUILD_TESTING "Enable CMake build integration tests"
       ${PROJECT_IS_TOP_LEVEL}
)

option(LIBFABRIC_ENABLE_PROFILE
       "Enable profiling support (enables fi_profile hooks)" OFF
)

if(LIBFABRIC_ENABLE_LTO)
  message(FATAL_ERROR "Use CMAKE_INTERPROCEDURAL_OPTIMIZATION to enable LTO")
endif()

set(LIBFABRIC_BUILD_ID
    ""
    CACHE STRING "Build ID annotation string"
)

# -----------------------------------------------------------------------------
# Compiler Warning and Sanitizer Options
# -----------------------------------------------------------------------------

option(LIBFABRIC_PICKY "Enable extra compiler warnings (pedantic mode)" OFF)
option(LIBFABRIC_ENABLE_ATOMICS "Use native atomics instead of mutex emulation"
       ON
)
option(LIBFABRIC_ENABLE_COVERAGE
       "Enable gcov-compatible coverage instrumentation" OFF
)
option(LIBFABRIC_THREAD_SAFETY_ANALYSIS "Enable Clang thread safety analysis"
       OFF
)

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
  LIBFABRIC_CUDA_DLOPEN
  "Load CUDA libraries at runtime via dlopen"
  ON
  "LIBFABRIC_ENABLE_CUDA"
  OFF
)

option(LIBFABRIC_ENABLE_ZE "Enable Intel Level Zero GPU memory support" OFF)

cmake_dependent_option(
  LIBFABRIC_ZE_DLOPEN
  "Load Level Zero libraries at runtime via dlopen"
  ON
  "LIBFABRIC_ENABLE_ZE"
  OFF
)

option(LIBFABRIC_ENABLE_ROCR "Enable AMD ROCr GPU memory support" OFF)

cmake_dependent_option(
  LIBFABRIC_ROCR_DLOPEN
  "Load ROCr libraries at runtime via dlopen"
  ON
  "LIBFABRIC_ENABLE_ROCR"
  OFF
)

option(LIBFABRIC_ENABLE_NEURON "Enable AWS Neuron accelerator support" OFF)

option(LIBFABRIC_ENABLE_SYNAPSEAI "Enable Habana SynapseAI accelerator support"
       OFF
)

option(LIBFABRIC_ENABLE_GDRCOPY "Enable NVIDIA GDRCopy support" OFF)

cmake_dependent_option(
  LIBFABRIC_GDRCOPY_DLOPEN
  "Load GDRCopy libraries at runtime via dlopen"
  ON
  "LIBFABRIC_ENABLE_GDRCOPY"
  OFF
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
  CACHE LIBFABRIC_DEFAULT_MONITOR
  PROPERTY STRINGS
           ""
           "uffd"
           "memhooks"
           "kdreg2"
           "disabled"
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
# Enable the provider (default: AUTO) LIBFABRIC_FOO_DLOPEN       - Build as
# loadable plugin (default: OFF)
# -----------------------------------------------------------------------------

set(LIBFABRIC_PROVIDERS
    tcp
    udp
    sockets
    rxm
    rxd
    mrail
    lnx
    sm2
    verbs
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
foreach(_prov IN LISTS LIBFABRIC_PROVIDERS)
  string(TOUPPER "${_prov}" _PROV)
  set(_default AUTO)
  if(_prov STREQUAL "ucx")
    set(_default OFF)
  endif()
  set(LIBFABRIC_PROVIDER_${_PROV}
      ${_default}
      CACHE STRING "Build ${_prov}: AUTO, ON (required), or OFF"
  )
  set_property(CACHE LIBFABRIC_PROVIDER_${_PROV} PROPERTY STRINGS AUTO ON OFF)
  if(NOT LIBFABRIC_PROVIDER_${_PROV} MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "LIBFABRIC_PROVIDER_${_PROV} must be AUTO, ON, or OFF")
  endif()
  option(LIBFABRIC_${_PROV}_DLOPEN "Build ${_prov} as a provider plugin" OFF)
  if(LIBFABRIC_${_PROV}_DLOPEN AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "${_prov}: plugins require BUILD_SHARED_LIBS=ON")
  endif()
endforeach()

set(LIBFABRIC_HOOKS
    perf
    trace
    profile
    monitor
    hook_debug
    hook_hmem
    dmabuf_peer_mem
)
foreach(_prov IN LISTS LIBFABRIC_HOOKS)
  string(TOUPPER "${_prov}" _PROV)
  option(LIBFABRIC_HOOK_${_PROV} "Build ${_prov} hook" ON)
  option(LIBFABRIC_${_PROV}_DLOPEN "Build ${_prov} as a hook plugin" OFF)
  if(LIBFABRIC_${_PROV}_DLOPEN AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "${_prov}: plugins require BUILD_SHARED_LIBS=ON")
  endif()
endforeach()

# -----------------------------------------------------------------------------
# PSM3 Provider Options
# -----------------------------------------------------------------------------

if(LIBFABRIC_PERF_DLOPEN)
  message(
    FATAL_ERROR
      "The perf hook requires private core state; build it into libfabric"
  )
endif()

cmake_dependent_option(
  LIBFABRIC_PSM3_VERBS
  "Enable PSM3 Verbs HAL (UD QPs)"
  ON
  "LIBFABRIC_PROVIDER_PSM3"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_SOCKETS
  "Enable PSM3 Sockets HAL (TCP)"
  ON
  "LIBFABRIC_PROVIDER_PSM3"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_UDP
  "Enable PSM3 UDP support"
  OFF
  "LIBFABRIC_PROVIDER_PSM3;LIBFABRIC_PSM3_SOCKETS"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_RC
  "Enable PSM3 User Space RC QPs"
  ON
  "LIBFABRIC_PROVIDER_PSM3;LIBFABRIC_PSM3_VERBS"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_DSA
  "Enable PSM3 Intel DSA support"
  ON
  "LIBFABRIC_PROVIDER_PSM3"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_RDMA_READ
  "Enable PSM3 RDMA read support"
  ON
  "LIBFABRIC_PROVIDER_PSM3;LIBFABRIC_PSM3_RC"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_UMR_CACHE
  "Enable PSM3 UMR caching"
  ON
  "LIBFABRIC_PROVIDER_PSM3"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM3_HWLOC
  "Enable PSM3 hwloc for NIC affinity"
  ON
  "LIBFABRIC_PROVIDER_PSM3"
  OFF
)

# -----------------------------------------------------------------------------
# EFA Provider Options
# -----------------------------------------------------------------------------

cmake_dependent_option(
  LIBFABRIC_EFA_POISONING
  "Enable EFA memory poisoning for debugging"
  OFF
  "LIBFABRIC_PROVIDER_EFA"
  OFF
)

# -----------------------------------------------------------------------------
# CXI Provider Options
# -----------------------------------------------------------------------------

cmake_dependent_option(
  LIBFABRIC_CXI_CURL_DLOPEN
  "Load curl library at runtime via dlopen"
  OFF
  "LIBFABRIC_PROVIDER_CXI"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_CXI_JSON_DLOPEN
  "Load json-c library at runtime via dlopen"
  OFF
  "LIBFABRIC_PROVIDER_CXI"
  OFF
)

# -----------------------------------------------------------------------------
# Other Provider Options
# -----------------------------------------------------------------------------

cmake_dependent_option(
  LIBFABRIC_LPP_THREAD_SAFE
  "Enable thread-safe LPP provider"
  ON
  "LIBFABRIC_PROVIDER_LPP"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM2_MQ_REQ_USER
  "Enable PSM2 MQ req user support"
  OFF
  "LIBFABRIC_PROVIDER_PSM2"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_PSM2_MQ_FP_MSG
  "Enable PSM2 MQ FP msg support"
  OFF
  "LIBFABRIC_PROVIDER_PSM2"
  OFF
)

foreach(_feature SHM_DSA PSM3_RV OPX_TARGET_CLONES OPX_HFISVC)
  set(LIBFABRIC_${_feature}
      AUTO
      CACHE STRING "Enable ${_feature}: AUTO, ON, or OFF"
  )
  set_property(CACHE LIBFABRIC_${_feature} PROPERTY STRINGS AUTO ON OFF)
  if(NOT LIBFABRIC_${_feature} MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "LIBFABRIC_${_feature} must be AUTO, ON, or OFF")
  endif()
endforeach()
set(LIBFABRIC_OPX_TRACER
    ""
    CACHE STRING "OPX trace categories (comma-separated, all, or ^ exclusions)"
)

# -----------------------------------------------------------------------------
# Fabtests Options
# -----------------------------------------------------------------------------

option(LIBFABRIC_BUILD_FABTESTS "Build fabtests test suite" OFF)
foreach(_provider EFA LPP VERBS)
  cmake_dependent_option(
    LIBFABRIC_FABTESTS_${_provider}
    "Build ${_provider} provider fabtests"
    ON
    "LIBFABRIC_BUILD_FABTESTS"
    OFF
  )
endforeach()

cmake_dependent_option(
  LIBFABRIC_FABTESTS_BENCHMARKS
  "Build fabtests benchmark tests"
  ON
  "LIBFABRIC_BUILD_FABTESTS"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_FUNCTIONAL
  "Build fabtests functional tests"
  ON
  "LIBFABRIC_BUILD_FABTESTS"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_UNIT
  "Build fabtests unit tests"
  ON
  "LIBFABRIC_BUILD_FABTESTS"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_MULTINODE
  "Build fabtests multinode tests"
  ON
  "LIBFABRIC_BUILD_FABTESTS"
  OFF
)

cmake_dependent_option(
  LIBFABRIC_FABTESTS_UBERTEST
  "Build fabtests ubertest"
  ON
  "LIBFABRIC_BUILD_FABTESTS"
  OFF
)

# Do not silently accept unsupported switches from the earlier CMake draft.
foreach(_unsupported LIBFABRIC_EMBEDDED LIBFABRIC_DIRECT
                     LIBFABRIC_EFA_UNIT_TESTS LIBFABRIC_FABTESTS_EFA_GDA
)
  if(${_unsupported})
    message(FATAL_ERROR "${_unsupported} is not supported by the CMake build")
  endif()
endforeach()
