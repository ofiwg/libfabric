# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FeatureSummary)

# Register custom package types for organized output

set_property(GLOBAL APPEND PROPERTY FeatureSummary_PKG_TYPES CORE)
set_property(GLOBAL APPEND PROPERTY FeatureSummary_PKG_TYPES PROVIDER_DEPS)
set_property(GLOBAL APPEND PROPERTY FeatureSummary_PKG_TYPES GPU)
set_property(GLOBAL APPEND PROPERTY FeatureSummary_PKG_TYPES MEMORY)
set_property(GLOBAL APPEND PROPERTY FeatureSummary_PKG_TYPES DEBUG)

# =============================================================================
# Package Property Registration
# =============================================================================

# -----------------------------------------------------------------------------
# libfabric_set_package_info() Register package properties for all dependencies
# Call this after find_package() calls complete
# -----------------------------------------------------------------------------
function(libfabric_set_package_info)
  # -------------------------------------------------------------------------
  # Core Dependencies
  # -------------------------------------------------------------------------
  set_package_properties(
    Threads PROPERTIES
    TYPE REQUIRED
    DESCRIPTION "POSIX threads"
    PURPOSE "Thread support for concurrent operations"
  )

  set_package_properties(
    Atomics PROPERTIES
    TYPE REQUIRED
    DESCRIPTION "Atomic operations"
    PURPOSE "Lock-free data structures and synchronization"
  )

  set_package_properties(
    DL PROPERTIES
    TYPE CORE
    DESCRIPTION "Dynamic linker"
    PURPOSE "Runtime loading of provider plugins"
  )

  set_package_properties(
    RT PROPERTIES
    TYPE CORE
    DESCRIPTION "POSIX realtime extensions"
    PURPOSE "High-resolution timers (clock_gettime)"
  )

  set_package_properties(
    Epoll PROPERTIES
    TYPE CORE
    DESCRIPTION "Linux epoll"
    PURPOSE "Efficient I/O event notification"
  )

  set_package_properties(
    Kqueue PROPERTIES
    TYPE CORE
    DESCRIPTION "BSD kqueue"
    PURPOSE "Efficient I/O event notification on BSD/macOS"
  )

  set_package_properties(
    LibUring PROPERTIES
    TYPE CORE
    DESCRIPTION "io_uring library"
    URL "https://github.com/axboe/liburing"
    PURPOSE "Async I/O via Linux io_uring"
  )

  # -------------------------------------------------------------------------
  # Provider Dependencies
  # -------------------------------------------------------------------------
  set_package_properties(
    IBVerbs PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "libibverbs (rdma-core)"
    URL "https://github.com/linux-rdma/rdma-core"
    PURPOSE "InfiniBand/RoCE verbs interface (verbs, efa providers)"
  )

  set_package_properties(
    RdmaCM PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "librdmacm (rdma-core)"
    URL "https://github.com/linux-rdma/rdma-core"
    PURPOSE "RDMA connection management (verbs provider)"
  )

  set_package_properties(
    Efadv PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "EFA direct verbs"
    URL "https://github.com/amzn/amzn-drivers"
    PURPOSE "AWS EFA provider"
  )

  set_package_properties(
    PSM2 PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "Intel PSM2 library"
    PURPOSE "Intel Omni-Path fabric (psm2 provider)"
  )

  set_package_properties(
    PSM3 PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "Intel PSM3 library"
    PURPOSE "Intel Ethernet with PSM3 (psm3 provider)"
  )

  set_package_properties(
    UCX PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "Unified Communication X"
    URL "https://github.com/openucx/ucx"
    PURPOSE "UCX unified communication (ucx provider)"
  )

  set_package_properties(
    LibCxi PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "HPE Cassini CXI library"
    PURPOSE "HPE Slingshot fabric (cxi provider)"
  )

  set_package_properties(
    CXIDriverHeaders PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "CXI driver UAPI headers"
    PURPOSE "CXI provider kernel interface"
  )

  set_package_properties(
    LibNL PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "Netlink library"
    URL "https://www.infradead.org/~tgr/libnl/"
    PURPOSE "Route resolution (usnic provider)"
  )

  set_package_properties(
    Numa PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "NUMA policy library"
    PURPOSE "NUMA-aware memory allocation"
  )

  set_package_properties(
    Hwloc PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "Hardware locality library"
    URL "https://www.open-mpi.org/projects/hwloc/"
    PURPOSE "Hardware topology discovery (psm3 provider)"
  )

  set_package_properties(
    UUID PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "UUID generation library"
    PURPOSE "Unique identifier generation"
  )

  set_package_properties(
    DSA PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "Intel Data Streaming Accelerator"
    PURPOSE "Hardware-accelerated memory operations (shm, psm3 providers)"
  )

  set_package_properties(
    JsonC PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "JSON-C library"
    URL "https://github.com/json-c/json-c"
    PURPOSE "Configuration parsing (cxi provider)"
  )

  set_package_properties(
    LibCurl PROPERTIES
    TYPE PROVIDER_DEPS
    DESCRIPTION "libcurl"
    URL "https://curl.se/libcurl/"
    PURPOSE "HTTP client for telemetry (cxi provider)"
  )

  # -------------------------------------------------------------------------
  # GPU/Accelerator Support
  # -------------------------------------------------------------------------
  set_package_properties(
    CUDAToolkit PROPERTIES
    TYPE GPU
    DESCRIPTION "NVIDIA CUDA Toolkit"
    URL "https://developer.nvidia.com/cuda-toolkit"
    PURPOSE "NVIDIA GPU memory registration and transfers"
  )

  set_package_properties(
    GDRCopy PROPERTIES
    TYPE GPU
    DESCRIPTION "NVIDIA GDRCopy"
    URL "https://github.com/NVIDIA/gdrcopy"
    PURPOSE "GPU-direct RDMA copy optimization"
  )

  set_package_properties(
    LevelZero PROPERTIES
    TYPE GPU
    DESCRIPTION "Intel oneAPI Level Zero"
    URL "https://github.com/oneapi-src/level-zero"
    PURPOSE "Intel GPU memory registration and transfers"
  )

  set_package_properties(
    ROCm PROPERTIES
    TYPE GPU
    DESCRIPTION "AMD ROCm runtime"
    URL "https://rocm.docs.amd.com/"
    PURPOSE "AMD GPU memory registration and transfers"
  )

  set_package_properties(
    Neuron PROPERTIES
    TYPE GPU
    DESCRIPTION "AWS Neuron SDK"
    URL "https://aws.amazon.com/machine-learning/neuron/"
    PURPOSE "AWS Inferentia/Trainium accelerator support"
  )

  set_package_properties(
    SynapseAI PROPERTIES
    TYPE GPU
    DESCRIPTION "Habana SynapseAI"
    URL "https://habana.ai/"
    PURPOSE "Habana Gaudi accelerator support"
  )

  # -------------------------------------------------------------------------
  # Memory Monitors
  # -------------------------------------------------------------------------
  set_package_properties(
    UFD PROPERTIES
    TYPE MEMORY
    DESCRIPTION "Userfaultfd"
    PURPOSE "Memory page fault monitoring for MR cache invalidation"
  )

  set_package_properties(
    XPMEM PROPERTIES
    TYPE MEMORY
    DESCRIPTION "Cross-process memory"
    PURPOSE "Zero-copy cross-process memory sharing"
  )

  set_package_properties(
    Kdreg2 PROPERTIES
    TYPE MEMORY
    DESCRIPTION "Kdreg2 kernel module"
    PURPOSE "Kernel-assisted memory registration monitoring"
  )

  set_package_properties(
    CMA PROPERTIES
    TYPE MEMORY
    DESCRIPTION "Cross Memory Attach"
    PURPOSE "Process-to-process memory transfers (shm provider)"
  )

  # -------------------------------------------------------------------------
  # Debug/Development
  # -------------------------------------------------------------------------
  set_package_properties(
    LTTng PROPERTIES
    TYPE DEBUG
    DESCRIPTION "LTTng userspace tracer"
    URL "https://lttng.org/"
    PURPOSE "Low-overhead userspace tracing"
  )

  set_package_properties(
    Valgrind PROPERTIES
    TYPE DEBUG
    DESCRIPTION "Valgrind memory debugger"
    URL "https://valgrind.org/"
    PURPOSE "Memory error detection annotations"
  )

  set_package_properties(
    Ethtool PROPERTIES
    TYPE DEBUG
    DESCRIPTION "Linux ethtool interface"
    PURPOSE "Network interface speed detection"
  )
endfunction()

# -----------------------------------------------------------------------------
# libfabric_register_features() Register feature info for all libfabric features
# -----------------------------------------------------------------------------
function(libfabric_register_features)
  # Core compiler/platform features
  add_feature_info("C11 Atomics" HAVE_ATOMICS "C11 stdatomic.h support")
  add_feature_info(
    "128-bit Atomics" HAVE_BUILTIN_MM_INT128_ATOMICS
    "Lock-free 128-bit atomic operations"
  )
  add_feature_info(
    "Symbol Versioning" HAVE_SYMVER_SUPPORT "ELF .symver for ABI compatibility"
  )
  add_feature_info("Pthread Spinlock" PT_LOCK_SPIN "pthread_spin_* functions")

  # I/O features
  add_feature_info("epoll" HAVE_EPOLL "Linux epoll I/O multiplexing")
  add_feature_info("kqueue" HAVE_KQUEUE "BSD/macOS kqueue I/O multiplexing")
  add_feature_info("io_uring" HAVE_LIBURING "Linux io_uring async I/O")

  # Memory features
  add_feature_info(
    "Memhooks Monitor" HAVE_MEMHOOKS_MONITOR
    "Memory hooks for MR cache invalidation"
  )
  add_feature_info(
    "UFFD Monitor" HAVE_UFFD_MONITOR "Userfaultfd for MR cache invalidation"
  )
  add_feature_info(
    "Kdreg2 Monitor" HAVE_KDREG2_MONITOR "Kdreg2 kernel module for MR cache"
  )
  add_feature_info("XPMEM" HAVE_XPMEM "Cross-process memory sharing")

  # GPU features
  add_feature_info("CUDA" HAVE_CUDA "NVIDIA CUDA GPU memory support")
  add_feature_info(
    "CUDA dmabuf" HAVE_CUDA_DMABUF "CUDA dmabuf for GPU direct RDMA"
  )
  add_feature_info("Level Zero" HAVE_ZE "Intel GPU memory support")
  add_feature_info("ROCm" HAVE_ROCR "AMD ROCm GPU memory support")
  add_feature_info(
    "ROCm dmabuf" HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF
    "ROCm dmabuf for GPU direct RDMA"
  )
  add_feature_info("GDRCopy" HAVE_GDRCOPY "NVIDIA GDRCopy optimization")
  add_feature_info("Neuron" HAVE_NEURON "AWS Inferentia/Trainium accelerator")
  add_feature_info("SynapseAI" HAVE_SYNAPSEAI "Habana Gaudi accelerator")

  # Debug features
  add_feature_info("LTTng Tracing" HAVE_LTTNG "LTTng userspace tracepoints")
  add_feature_info("Valgrind" INCLUDE_VALGRIND "Valgrind memory annotations")
  add_feature_info(
    "Linux rdpmc" HAVE_LINUX_PERF_RDPMC "Hardware performance counters"
  )

  # Build configuration
  add_feature_info(
    "Debug Build" ENABLE_DEBUG "Debug assertions and checks enabled"
  )
  add_feature_info("Profiling" HAVE_FABRIC_PROFILE "fi_profile hook support")
endfunction()

# -----------------------------------------------------------------------------
# libfabric_print_feature_summary() Print organized feature summary with custom
# categories
# -----------------------------------------------------------------------------
function(libfabric_print_feature_summary)
  message(STATUS "")
  message(
    STATUS
      "==============================================================================="
  )
  message(STATUS "                        Libfabric Configuration Summary")
  message(
    STATUS
      "==============================================================================="
  )

  # Core dependencies
  feature_summary(
    WHAT CORE_PACKAGES_FOUND
    DESCRIPTION "Core dependencies found:"
    QUIET_ON_EMPTY
  )

  # Provider dependencies
  feature_summary(
    WHAT PROVIDER_DEPS_PACKAGES_FOUND
    DESCRIPTION "Provider dependencies found:"
    QUIET_ON_EMPTY
  )
  feature_summary(
    WHAT PROVIDER_DEPS_PACKAGES_NOT_FOUND
    DESCRIPTION
      "Provider dependencies not found (some providers may be disabled):"
    QUIET_ON_EMPTY
  )

  # GPU/Accelerator packages
  feature_summary(
    WHAT GPU_PACKAGES_FOUND
    DESCRIPTION "GPU/Accelerator support:"
    QUIET_ON_EMPTY
  )

  # Memory subsystem packages
  feature_summary(
    WHAT MEMORY_PACKAGES_FOUND
    DESCRIPTION "Memory subsystem support:"
    QUIET_ON_EMPTY
  )

  # Debug packages
  feature_summary(
    WHAT DEBUG_PACKAGES_FOUND
    DESCRIPTION "Debug/Development tools:"
    QUIET_ON_EMPTY
  )

  # All enabled features
  message(STATUS "")
  feature_summary(WHAT ENABLED_FEATURES DESCRIPTION "Enabled features:")

  # Print provider summary (uses our custom function)
  libfabric_print_provider_summary()

  message(
    STATUS
      "==============================================================================="
  )
endfunction()
