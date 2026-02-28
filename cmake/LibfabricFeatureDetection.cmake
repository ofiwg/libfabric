# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)
include(CheckCSourceCompiles)

# -----------------------------------------------------------------------------
# libfabric_detect_features() Main function to detect all features. Call this
# after find_package() calls. Sets HAVE_* variables in parent scope for config.h
# generation.
# -----------------------------------------------------------------------------
function(libfabric_detect_features)
  # -------------------------------------------------------------------------
  # Atomics Detection
  # -------------------------------------------------------------------------
  set(HAVE_ATOMICS
      ${Atomics_C11_FOUND}
      PARENT_SCOPE
  )
  set(HAVE_BUILTIN_ATOMICS
      ${Atomics_BUILTIN_FOUND}
      PARENT_SCOPE
  )
  set(HAVE_BUILTIN_MM_ATOMICS
      ${Atomics_BUILTIN_MM_FOUND}
      PARENT_SCOPE
  )

  # Check for atomic least types (int_least32_t, int_least64_t)
  if(Atomics_C11_FOUND)
    libfabric_check_c11_atomic_least_types(_have_least_types)
    set(HAVE_ATOMICS_LEAST_TYPES
        ${_have_least_types}
        PARENT_SCOPE
    )
  endif()

  # Check for 128-bit integer and atomics
  check_type_size("__int128" SIZEOF_INT128)
  if(SIZEOF_INT128)
    set(HAVE___INT128
        TRUE
        PARENT_SCOPE
    )
    if(Atomics_BUILTIN_MM_FOUND)
      libfabric_check_int128_atomics(_have_int128_atomics)
      set(HAVE_BUILTIN_MM_INT128_ATOMICS
          ${_have_int128_atomics}
          PARENT_SCOPE
      )
    endif()
  else()
    set(HAVE___INT128
        FALSE
        PARENT_SCOPE
    )
  endif()

  # -------------------------------------------------------------------------
  # CPU/Compiler Feature Detection
  # -------------------------------------------------------------------------
  libfabric_check_cpuid(_have_cpuid)
  set(HAVE_CPUID
      ${_have_cpuid}
      PARENT_SCOPE
  )

  libfabric_check_pthread_spin(_have_spinlock)
  set(PT_LOCK_SPIN
      ${_have_spinlock}
      PARENT_SCOPE
  )

  libfabric_check_getifaddrs(_have_getifaddrs)
  set(HAVE_GETIFADDRS
      ${_have_getifaddrs}
      PARENT_SCOPE
  )

  # Symbol versioning (disabled for embedded builds)
  if(NOT LIBFABRIC_EMBEDDED)
    libfabric_check_symver_support(_have_symver)
    set(HAVE_SYMVER_SUPPORT
        ${_have_symver}
        PARENT_SCOPE
    )
  else()
    set(HAVE_SYMVER_SUPPORT
        FALSE
        PARENT_SCOPE
    )
  endif()

  libfabric_check_alias_attribute(_have_alias)
  set(HAVE_ALIAS_ATTRIBUTE
      ${_have_alias}
      PARENT_SCOPE
  )

  # -------------------------------------------------------------------------
  # I/O Multiplexing
  # -------------------------------------------------------------------------
  set(HAVE_EPOLL
      ${Epoll_FOUND}
      PARENT_SCOPE
  )
  set(HAVE_KQUEUE
      ${Kqueue_FOUND}
      PARENT_SCOPE
  )

  # -------------------------------------------------------------------------
  # Ethtool (Linux)
  # -------------------------------------------------------------------------
  set(HAVE_ETHTOOL
      ${Ethtool_FOUND}
      PARENT_SCOPE
  )
  if(Ethtool_FOUND)
    set(HAVE_DECL_ETHTOOL_CMD_SPEED
        ${Ethtool_HAS_CMD_SPEED}
        PARENT_SCOPE
    )
    set(HAVE_DECL_SPEED_UNKNOWN
        ${Ethtool_HAS_SPEED_UNKNOWN}
        PARENT_SCOPE
    )
  else()
    set(HAVE_DECL_ETHTOOL_CMD_SPEED
        FALSE
        PARENT_SCOPE
    )
    set(HAVE_DECL_SPEED_UNKNOWN
        FALSE
        PARENT_SCOPE
    )
  endif()

  # -------------------------------------------------------------------------
  # Linux Performance Counter (rdpmc)
  # -------------------------------------------------------------------------
  if(LIBFABRIC_LINUX AND LIBFABRIC_X86_64)
    libfabric_check_linux_perf_rdpmc(_have_rdpmc)
    set(HAVE_LINUX_PERF_RDPMC
        ${_have_rdpmc}
        PARENT_SCOPE
    )
  else()
    set(HAVE_LINUX_PERF_RDPMC
        FALSE
        PARENT_SCOPE
    )
  endif()

  # -------------------------------------------------------------------------
  # Memory Subsystem Features
  # -------------------------------------------------------------------------

  # Userfaultfd
  if(UFD_FOUND)
    set(HAVE_UFFD_UNMAP
        ${UFD_HAS_UNMAP}
        PARENT_SCOPE
    )
    set(HAVE_UFFD_THREAD_ID
        ${UFD_HAS_THREAD_ID}
        PARENT_SCOPE
    )
  else()
    set(HAVE_UFFD_UNMAP
        FALSE
        PARENT_SCOPE
    )
    set(HAVE_UFFD_THREAD_ID
        FALSE
        PARENT_SCOPE
    )
  endif()

  # XPMEM
  set(HAVE_XPMEM
      ${XPMEM_FOUND}
      PARENT_SCOPE
  )

  # kdreg2
  set(HAVE_KDREG2
      ${Kdreg2_FOUND}
      PARENT_SCOPE
  )
  set(HAVE_KDREG2_MONITOR
      ${Kdreg2_FOUND}
      PARENT_SCOPE
  )
  if(Kdreg2_FOUND AND NOT Kdreg2_IN_LINUX)
    set(HAVE_KDREG2_INCLUDE_PATH
        TRUE
        PARENT_SCOPE
    )
  else()
    set(HAVE_KDREG2_INCLUDE_PATH
        FALSE
        PARENT_SCOPE
    )
  endif()

  # Memory monitors - compute availability from platform and options
  if(HAVE_MEMHOOKS_SUPPORT AND LIBFABRIC_MEMHOOKS_MONITOR)
    set(HAVE_MEMHOOKS_MONITOR
        TRUE
        PARENT_SCOPE
    )
  else()
    set(HAVE_MEMHOOKS_MONITOR
        FALSE
        PARENT_SCOPE
    )
  endif()

  if(UFD_FOUND
     AND UFD_HAS_UNMAP
     AND LIBFABRIC_UFFD_MONITOR
  )
    set(HAVE_UFFD_MONITOR
        TRUE
        PARENT_SCOPE
    )
  else()
    set(HAVE_UFFD_MONITOR
        FALSE
        PARENT_SCOPE
    )
  endif()

  # Default memory monitor
  if(LIBFABRIC_DEFAULT_MONITOR STREQUAL "")
    unset(HAVE_MR_CACHE_MONITOR_DEFAULT PARENT_SCOPE)
  else()
    set(HAVE_MR_CACHE_MONITOR_DEFAULT
        "${LIBFABRIC_DEFAULT_MONITOR}"
        PARENT_SCOPE
    )
  endif()

  # -------------------------------------------------------------------------
  # io_uring
  # -------------------------------------------------------------------------
  set(HAVE_LIBURING
      ${LibUring_FOUND}
      PARENT_SCOPE
  )

  # -------------------------------------------------------------------------
  # Tracing and Debugging
  # -------------------------------------------------------------------------
  set(HAVE_LTTNG
      ${LTTng_FOUND}
      PARENT_SCOPE
  )
  set(INCLUDE_VALGRIND
      ${Valgrind_FOUND}
      PARENT_SCOPE
  )

  # -------------------------------------------------------------------------
  # GPU/Accelerator Support
  # -------------------------------------------------------------------------

  # CUDA
  if(CUDAToolkit_FOUND)
    set(HAVE_CUDA
        TRUE
        PARENT_SCOPE
    )
    set(HAVE_CUDA_RUNTIME_H
        TRUE
        PARENT_SCOPE
    )
    set(ENABLE_CUDA_DLOPEN
        ${LIBFABRIC_CUDA_DLOPEN}
        PARENT_SCOPE
    )
    # CUDA dmabuf detection is done via CheckCUDADmabuf module
    include(CheckCUDADmabuf)
    set(HAVE_CUDA_DMABUF
        ${CUDAToolkit_HAS_DMABUF}
        PARENT_SCOPE
    )
    set(HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE
        ${CUDAToolkit_HAS_DMABUF_PCIE}
        PARENT_SCOPE
    )
  else()
    set(HAVE_CUDA
        FALSE
        PARENT_SCOPE
    )
    set(ENABLE_CUDA_DLOPEN
        FALSE
        PARENT_SCOPE
    )
    set(HAVE_CUDA_DMABUF
        FALSE
        PARENT_SCOPE
    )
    set(HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE
        FALSE
        PARENT_SCOPE
    )
  endif()

  # Level Zero (Intel GPU)
  set(HAVE_ZE
      ${LevelZero_FOUND}
      PARENT_SCOPE
  )
  set(ENABLE_ZE_DLOPEN
      ${LIBFABRIC_ZE_DLOPEN}
      PARENT_SCOPE
  )
  if(LevelZero_FOUND)
    set(HAVE_DRM
        ${LevelZero_HAS_DRM}
        PARENT_SCOPE
    )
    set(HAVE_LIBDRM
        ${LevelZero_HAS_LIBDRM}
        PARENT_SCOPE
    )
  else()
    set(HAVE_DRM
        FALSE
        PARENT_SCOPE
    )
    set(HAVE_LIBDRM
        FALSE
        PARENT_SCOPE
    )
  endif()

  # ROCm (AMD GPU)
  set(HAVE_ROCR
      ${ROCm_FOUND}
      PARENT_SCOPE
  )
  set(ENABLE_ROCR_DLOPEN
      ${LIBFABRIC_ROCR_DLOPEN}
      PARENT_SCOPE
  )
  if(ROCm_FOUND)
    set(HAVE_ROCR_RUNTIME_H
        TRUE
        PARENT_SCOPE
    )
    set(HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF
        ${ROCm_HAS_DMABUF}
        PARENT_SCOPE
    )
  else()
    set(HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF
        FALSE
        PARENT_SCOPE
    )
  endif()

  # GDRCopy
  set(HAVE_GDRCOPY
      ${GDRCopy_FOUND}
      PARENT_SCOPE
  )
  set(ENABLE_GDRCOPY_DLOPEN
      ${LIBFABRIC_GDRCOPY_DLOPEN}
      PARENT_SCOPE
  )

  # Neuron
  set(HAVE_NEURON
      ${Neuron_FOUND}
      PARENT_SCOPE
  )

  # SynapseAI
  set(HAVE_SYNAPSEAI
      ${SynapseAI_FOUND}
      PARENT_SCOPE
  )

  # -------------------------------------------------------------------------
  # Miscellaneous
  # -------------------------------------------------------------------------
  set(HAVE_RESTRICTED_DL
      ${LIBFABRIC_RESTRICTED_DL}
      PARENT_SCOPE
  )
  set(HAVE_CLOCK_GETTIME
      TRUE
      PARENT_SCOPE
  )

  # -------------------------------------------------------------------------
  # Header Checks
  # -------------------------------------------------------------------------
  check_include_file("elf.h" _have_elf_h)
  check_include_file("sys/auxv.h" _have_sys_auxv_h)
  check_include_file("linux/mman.h" _have_linux_mman_h)
  check_include_file("sys/syscall.h" _have_sys_syscall_h)

  set(HAVE_ELF_H
      ${_have_elf_h}
      PARENT_SCOPE
  )
  set(HAVE_SYS_AUXV_H
      ${_have_sys_auxv_h}
      PARENT_SCOPE
  )
  set(HAVE_LINUX_MMAN_H
      ${_have_linux_mman_h}
      PARENT_SCOPE
  )
  set(HAVE_SYS_SYSCALL_H
      ${_have_sys_syscall_h}
      PARENT_SCOPE
  )

  # -------------------------------------------------------------------------
  # Memhooks Symbol Checks
  # -------------------------------------------------------------------------
  check_symbol_exists(__curbrk "unistd.h" _have___curbrk)
  check_symbol_exists(__clear_cache "stdlib.h" _have___clear_cache)
  set(HAVE___CURBRK
      ${_have___curbrk}
      PARENT_SCOPE
  )
  set(HAVE___CLEAR_CACHE
      ${_have___clear_cache}
      PARENT_SCOPE
  )

  if(_have_sys_syscall_h)
    check_symbol_exists(__syscall "sys/syscall.h" _have___syscall)
    set(HAVE___SYSCALL
        ${_have___syscall}
        PARENT_SCOPE
    )

    check_c_source_compiles(
      "
      #include <sys/syscall.h>
      int main() {
        #ifndef __syscall
        #error __syscall not declared
        #endif
        return 0;
      }
    "
      _have_decl___syscall
    )
    set(HAVE_DECL___SYSCALL
        ${_have_decl___syscall}
        PARENT_SCOPE
    )
  endif()

  # -------------------------------------------------------------------------
  # UCX Feature Detection
  # -------------------------------------------------------------------------
  if(UCX_FOUND)
    if(UCX_HAVE_WORKER_FLAG_IGNORE_REQUEST_LEAK)
      set(HAVE_DECL_UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK
          1
          PARENT_SCOPE
      )
    else()
      set(HAVE_DECL_UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK
          0
          PARENT_SCOPE
      )
    endif()
  else()
    set(HAVE_DECL_UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK
        0
        PARENT_SCOPE
    )
  endif()

  # -------------------------------------------------------------------------
  # Build Configuration
  # -------------------------------------------------------------------------
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ENABLE_DEBUG
        1
        PARENT_SCOPE
    )
  else()
    set(ENABLE_DEBUG
        0
        PARENT_SCOPE
    )
  endif()

  set(HAVE_FABRIC_PROFILE
      ${LIBFABRIC_ENABLE_PROFILE}
      PARENT_SCOPE
  )

  if(LIBFABRIC_DIRECT)
    set(FABRIC_DIRECT_ENABLED
        TRUE
        PARENT_SCOPE
    )
  endif()
endfunction()
