# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricFeatureDetection
-------------------------

Translate dependency and compiler results into config.h feature values.

Requires LibfabricDependencies and LibfabricPlatform. Sets HAVE_* and other
configuration variables; target creation and config.h generation follow.
Compiler checks use CMake's internal cache, not user-facing option entries.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)
include(CheckCSourceCompiles)
include(CMakePushCheckState)

set(HAVE_LIBDL ${LIBFABRIC_HAVE_DLOPEN})
# Atomics Detection
set(HAVE_ATOMICS ${Atomics_C11_FOUND})
set(HAVE_BUILTIN_ATOMICS ${Atomics_BUILTIN_FOUND})
set(HAVE_BUILTIN_MM_ATOMICS ${Atomics_BUILTIN_MM_FOUND})
set(HAVE_ATOMICS_LEAST_TYPES FALSE)
set(HAVE_BUILTIN_MM_INT128_ATOMICS FALSE)
cmake_push_check_state()
list(APPEND CMAKE_REQUIRED_LIBRARIES ${Atomics_LIBRARIES})

# Check for atomic least types (int_least32_t, int_least64_t)
if(Atomics_C11_FOUND)
  check_c_source_compiles(
    "
    #include <stdatomic.h>
    int main() {
      atomic_int_least32_t a;
      atomic_int_least64_t b;
      return 0;
    }
  "
    _have_least_types
  )
  set(HAVE_ATOMICS_LEAST_TYPES ${_have_least_types})
endif()

# Check for 128-bit integer and atomics
check_type_size("__int128" SIZEOF_INT128)
if(SIZEOF_INT128)
  set(HAVE___INT128 TRUE)
  if(Atomics_BUILTIN_MM_FOUND)
    check_c_source_compiles(
      "
      #include <stdint.h>
      int main() {
        __int128 d = 0;
        __int128 s = 1;
        __int128 c = 0;
        __int128 r;
        r = __atomic_fetch_add(&d, s, __ATOMIC_SEQ_CST);
        __atomic_load(&d, &r, __ATOMIC_SEQ_CST);
        __atomic_exchange(&d, &s, &r, __ATOMIC_SEQ_CST);
        __atomic_compare_exchange(&d, &c, &s, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        return !__atomic_is_lock_free(sizeof(d), 0);
      }
    "
      _have_int128_atomics
    )
    set(HAVE_BUILTIN_MM_INT128_ATOMICS ${_have_int128_atomics})
  endif()
else()
  set(HAVE___INT128 FALSE)
endif()

# CPU/Compiler Feature Detection
cmake_pop_check_state()
check_c_source_compiles(
  "
  #include <stddef.h>
  #include <cpuid.h>
  int main() {
    int a, b, c, d;
    __cpuid_count(0, 0, a, b, c, d);
    return 0;
  }
"
  _have_cpuid
)
set(HAVE_CPUID ${_have_cpuid})

cmake_push_check_state()
set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
check_symbol_exists(pthread_spin_init "pthread.h" _have_spinlock)
cmake_pop_check_state()
set(PT_LOCK_SPIN ${_have_spinlock})

check_symbol_exists(getifaddrs "sys/types.h;ifaddrs.h" _have_getifaddrs)
if(WIN32)
  # src/windows/osd.c supplies this compatibility function.
  set(HAVE_GETIFADDRS TRUE)
else()
  set(HAVE_GETIFADDRS ${_have_getifaddrs})
endif()

check_c_source_compiles(
  "
  __asm__(\".symver main_, main@ABIVER_1.0\");
  int main() { return 0; }
"
  _have_symver
)
set(HAVE_SYMVER_SUPPORT ${_have_symver})

check_c_source_compiles(
  "
  int foo(int arg);
  int foo(int arg) { return arg + 3; }
  int foo2(int arg) __attribute__ (( __alias__(\"foo\")));
  int main() { foo2(1); return 0; }
"
  _have_alias
)
set(HAVE_ALIAS_ATTRIBUTE ${_have_alias})

# I/O Multiplexing
set(HAVE_EPOLL ${Epoll_FOUND})
set(HAVE_KQUEUE ${Kqueue_FOUND})

# Ethtool (Linux)
set(HAVE_ETHTOOL ${Ethtool_FOUND})
if(Ethtool_FOUND)
  set(HAVE_DECL_ETHTOOL_CMD_SPEED ${Ethtool_HAS_CMD_SPEED})
  set(HAVE_DECL_SPEED_UNKNOWN ${Ethtool_HAS_SPEED_UNKNOWN})
else()
  set(HAVE_DECL_ETHTOOL_CMD_SPEED FALSE)
  set(HAVE_DECL_SPEED_UNKNOWN FALSE)
endif()

# Linux Performance Counter (rdpmc)
if(LIBFABRIC_LINUX AND LIBFABRIC_X86_64)
  check_c_source_compiles(
    "
    #include <linux/perf_event.h>
    int main() {
      __builtin_ia32_rdpmc(0);
      return 0;
    }
  "
    _have_rdpmc
  )
  set(HAVE_LINUX_PERF_RDPMC ${_have_rdpmc})
else()
  set(HAVE_LINUX_PERF_RDPMC FALSE)
endif()

# Memory Subsystem Features

# Userfaultfd
if(UFD_FOUND)
  set(HAVE_UFFD_UNMAP ${UFD_HAS_UNMAP})
  set(HAVE_UFFD_THREAD_ID ${UFD_HAS_THREAD_ID})
else()
  set(HAVE_UFFD_UNMAP FALSE)
  set(HAVE_UFFD_THREAD_ID FALSE)
endif()

# XPMEM
set(HAVE_XPMEM ${XPMEM_FOUND})

# kdreg2
set(HAVE_KDREG2 ${Kdreg2_FOUND})
set(HAVE_KDREG2_MONITOR ${Kdreg2_FOUND})
if(Kdreg2_FOUND AND NOT Kdreg2_IN_LINUX)
  set(HAVE_KDREG2_INCLUDE_PATH TRUE)
else()
  set(HAVE_KDREG2_INCLUDE_PATH FALSE)
endif()

# Memory monitors - compute availability from platform and options
if(HAVE_MEMHOOKS_SUPPORT AND LIBFABRIC_MEMHOOKS_MONITOR)
  set(ENABLE_MEMHOOKS_MONITOR TRUE)
else()
  set(ENABLE_MEMHOOKS_MONITOR FALSE)
endif()

if(UFD_FOUND
   AND UFD_HAS_UNMAP
   AND LIBFABRIC_UFFD_MONITOR
)
  set(ENABLE_UFFD_MONITOR TRUE)
else()
  set(ENABLE_UFFD_MONITOR FALSE)
endif()

if(NOT LIBFABRIC_DEFAULT_MONITOR MATCHES "^(|uffd|memhooks|kdreg2|disabled)$")
  message(FATAL_ERROR "Unknown memory monitor: ${LIBFABRIC_DEFAULT_MONITOR}")
endif()
if((LIBFABRIC_DEFAULT_MONITOR STREQUAL "uffd" AND NOT UFD_FOUND)
   OR (LIBFABRIC_DEFAULT_MONITOR STREQUAL "kdreg2" AND NOT Kdreg2_FOUND)
   OR (LIBFABRIC_DEFAULT_MONITOR STREQUAL "memhooks"
       AND (NOT HAVE_MEMHOOKS_SUPPORT
            OR NOT LIBFABRIC_MEMHOOKS_MONITOR
            OR LIBFABRIC_ENABLE_ASAN)
      )
)
  message(
    FATAL_ERROR
      "Requested memory monitor is unavailable: ${LIBFABRIC_DEFAULT_MONITOR}"
  )
endif()
# Default memory monitor
if(LIBFABRIC_DEFAULT_MONITOR STREQUAL "")
  unset(HAVE_MR_CACHE_MONITOR_DEFAULT)
else()
  set(HAVE_MR_CACHE_MONITOR_DEFAULT "${LIBFABRIC_DEFAULT_MONITOR}")
endif()

# io_uring
set(HAVE_LIBURING ${LibUring_FOUND})

# Tracing and Debugging
set(HAVE_LTTNG ${LTTng_FOUND})
set(INCLUDE_VALGRIND ${Valgrind_FOUND})

# GPU/Accelerator Support

# CUDA
if(CUDAToolkit_FOUND)
  set(HAVE_CUDA TRUE)
  set(HAVE_CUDA_RUNTIME_H TRUE)
  set(ENABLE_CUDA_DLOPEN ${LIBFABRIC_CUDA_DLOPEN})
  # CUDA dmabuf detection is done via CheckCUDADmabuf module
  include(CheckCUDADmabuf)
  set(HAVE_CUDA_DMABUF ${CUDAToolkit_HAS_DMABUF})
  set(HAVE_CUDA_CTX_SYNC_MEMOPS ${CUDAToolkit_HAS_CTX_SYNC_MEMOPS})
else()
  set(HAVE_CUDA FALSE)
  set(ENABLE_CUDA_DLOPEN FALSE)
  set(HAVE_CUDA_DMABUF FALSE)
  set(HAVE_CUDA_CTX_SYNC_MEMOPS FALSE)
endif()

# Level Zero (Intel GPU)
set(HAVE_ZE ${LevelZero_FOUND})
set(ENABLE_ZE_DLOPEN ${LIBFABRIC_ZE_DLOPEN})
if(LevelZero_FOUND)
  set(HAVE_DRM ${LevelZero_HAS_DRM})
  set(HAVE_LIBDRM ${LevelZero_HAS_LIBDRM})
else()
  set(HAVE_DRM FALSE)
  set(HAVE_LIBDRM FALSE)
endif()

# ROCm (AMD GPU)
set(HAVE_ROCR ${ROCm_FOUND})
set(ENABLE_ROCR_DLOPEN ${LIBFABRIC_ROCR_DLOPEN})
if(ROCm_FOUND)
  set(HAVE_ROCR_RUNTIME_H TRUE)
  set(HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF ${ROCm_HAS_DMABUF})
else()
  set(HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF FALSE)
endif()

# GDRCopy
set(HAVE_GDRCOPY ${GDRCopy_FOUND})
set(ENABLE_GDRCOPY_DLOPEN ${LIBFABRIC_GDRCOPY_DLOPEN})

# Neuron
set(HAVE_NEURON ${Neuron_FOUND})

# SynapseAI
set(HAVE_SYNAPSEAI ${SynapseAI_FOUND})

# Miscellaneous
set(HAVE_RESTRICTED_DL ${LIBFABRIC_RESTRICTED_DL})
if(LIBFABRIC_HAVE_CLOCK_GETTIME OR LIBFABRIC_HAVE_CLOCK_GETTIME_RT)
  set(HAVE_CLOCK_GETTIME 1)
else()
  set(HAVE_CLOCK_GETTIME 0)
endif()

# Header Checks
check_include_file("elf.h" _have_elf_h)
check_include_file("sys/auxv.h" _have_sys_auxv_h)
check_include_file("linux/mman.h" _have_linux_mman_h)
check_include_file("sys/syscall.h" _have_sys_syscall_h)

set(HAVE_ELF_H ${_have_elf_h})
set(HAVE_SYS_AUXV_H ${_have_sys_auxv_h})
set(HAVE_LINUX_MMAN_H ${_have_linux_mman_h})
set(HAVE_SYS_SYSCALL_H ${_have_sys_syscall_h})

# These libc/compiler-runtime symbols have no public header declarations.
check_c_source_compiles(
  "
  extern void *__curbrk;
  int main(void) { return __curbrk == 0; }" LIBFABRIC_HAVE_CURBRK
)
check_c_source_compiles(
  "
  void __clear_cache(void *, void *);
  int main(void) { char code[1]; __clear_cache(code, code + 1); return 0; }"
  LIBFABRIC_HAVE_CLEAR_CACHE
)
set(HAVE___CURBRK ${LIBFABRIC_HAVE_CURBRK})
set(HAVE___CLEAR_CACHE ${LIBFABRIC_HAVE_CLEAR_CACHE})

if(_have_sys_syscall_h)
  check_symbol_exists(__syscall "sys/syscall.h" _have___syscall)
  set(HAVE___SYSCALL ${_have___syscall})

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
  set(HAVE_DECL___SYSCALL ${_have_decl___syscall})
endif()

# UCX Feature Detection
if(UCX_FOUND)
  if(UCX_HAVE_WORKER_FLAG_IGNORE_REQUEST_LEAK)
    set(HAVE_DECL_UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK 1)
  else()
    set(HAVE_DECL_UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK 0)
  endif()
else()
  set(HAVE_DECL_UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK 0)
endif()

# Build Configuration

set(HAVE_FABRIC_PROFILE ${LIBFABRIC_ENABLE_PROFILE})
