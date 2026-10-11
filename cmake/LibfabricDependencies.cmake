# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricDependencies
---------------------

Dispatch dependency discovery for enabled features and providers.

Requires LibfabricOptions, LibfabricPlatform, and an enabled C compiler.
Find modules locate dependencies; Check modules validate provider-specific
APIs. Results feed LibfabricFeatureDetection and provider configuration.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()
include(CheckCSourceCompiles)
include(CheckSymbolExists)
include(CMakePushCheckState)

# Do not inherit availability for backends the caller has disabled.
foreach(
  _backend
  XPMEM
  Kdreg2
  UFD
  LibUring
  LTTng
  Valgrind
)
  set(${_backend}_FOUND FALSE)
endforeach()

# Required
find_package(Threads REQUIRED)
set(Atomics_C11_FOUND FALSE)
set(Atomics_BUILTIN_FOUND FALSE)
set(Atomics_BUILTIN_MM_FOUND FALSE)
set(Atomics_LIBRARIES "")
if(LIBFABRIC_ENABLE_ATOMICS)
  # The core can emulate atomics with locks; providers check their own needs.
  find_package(Atomics)
elseif(WIN32)
  message(FATAL_ERROR "Windows requires native atomics")
endif()

# Platform facilities are compiler/linker checks, not external packages.
if(NOT WIN32)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS})
  check_symbol_exists(dlopen "dlfcn.h" LIBFABRIC_HAVE_DLOPEN)
  if(NOT LIBFABRIC_HAVE_DLOPEN)
    message(FATAL_ERROR "libfabric requires dlopen on this platform")
  endif()
  set(CMAKE_REQUIRED_LIBRARIES "")
  check_symbol_exists(clock_gettime "time.h" LIBFABRIC_HAVE_CLOCK_GETTIME)
  if(NOT LIBFABRIC_HAVE_CLOCK_GETTIME)
    set(CMAKE_REQUIRED_LIBRARIES rt)
    check_symbol_exists(clock_gettime "time.h" LIBFABRIC_HAVE_CLOCK_GETTIME_RT)
    if(LIBFABRIC_HAVE_CLOCK_GETTIME_RT)
      set(LIBFABRIC_RT_LIBRARIES rt)
    endif()
  endif()
  # shm_open stayed in librt after clock_gettime moved to libc on glibc.
  if(NOT APPLE)
    set(CMAKE_REQUIRED_LIBRARIES "")
    check_symbol_exists(shm_open "sys/mman.h" LIBFABRIC_HAVE_SHM_OPEN)
    if(NOT LIBFABRIC_HAVE_SHM_OPEN)
      set(CMAKE_REQUIRED_LIBRARIES rt)
      check_symbol_exists(shm_open "sys/mman.h" LIBFABRIC_HAVE_SHM_OPEN_RT)
      if(NOT LIBFABRIC_HAVE_SHM_OPEN_RT)
        message(FATAL_ERROR "libfabric requires shm_open on this platform")
      endif()
      set(LIBFABRIC_RT_LIBRARIES rt)
    endif()
  endif()
  cmake_pop_check_state()
endif()

# Optional features (auto-detected)
find_package(Epoll)
find_package(Kqueue)
find_package(Ethtool)
find_package(CMA)

# Userfaultfd
if(LIBFABRIC_UFFD_MONITOR)
  find_package(UFD)
endif()

# XPMEM
if(LIBFABRIC_ENABLE_XPMEM)
  find_package(XPMEM)
endif()

# kdreg2
if(LIBFABRIC_KDREG2_MONITOR)
  find_package(Kdreg2)
endif()

# io_uring
if(LIBFABRIC_ENABLE_LIBURING AND LIBFABRIC_LINUX)
  find_package(LibUring)
  if(LibUring_FOUND)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES LibUring::LibUring)
    check_c_source_compiles(
      "
      #include <liburing.h>
      int main(void) {
        struct io_uring_sqe sqe = {0};
        io_uring_prep_poll_multishot(&sqe, 0, 0);
        return IORING_CQE_F_MORE;
      }"
      LIBFABRIC_LIBURING_MULTISHOT
    )
    cmake_pop_check_state()
    if(NOT LIBFABRIC_LIBURING_MULTISHOT)
      set(LibUring_FOUND FALSE)
    endif()
  endif()
endif()

# LTTng
if(LIBFABRIC_ENABLE_LTTNG)
  find_package(LTTng REQUIRED)
endif()

# Valgrind
if(LIBFABRIC_ENABLE_VALGRIND)
  find_package(Valgrind REQUIRED)
endif()

include(LibfabricHmemDependencies)

# Search only for dependencies used by enabled provider candidates.
if(LIBFABRIC_PROVIDER_VERBS
   OR LIBFABRIC_PROVIDER_EFA
   OR LIBFABRIC_PROVIDER_USNIC
   OR (LIBFABRIC_PROVIDER_PSM3 AND LIBFABRIC_PSM3_VERBS)
)
  find_package(IBVerbs)
  include(CheckIBVerbs)
endif()
if(LIBFABRIC_PROVIDER_VERBS)
  find_package(RdmaCM)
  include(CheckRdmaCM)
endif()
if(LIBFABRIC_PROVIDER_EFA)
  find_package(Efadv)
  include(CheckEfadv)
endif()
if(LIBFABRIC_PROVIDER_OPX
   OR LIBFABRIC_PROVIDER_PSM3
   OR LIBFABRIC_PROVIDER_SHM
)
  find_package(Numa)
  find_package(UUID)
  find_package(DSA)
endif()
if(LIBFABRIC_PROVIDER_VERBS OR (LIBFABRIC_PROVIDER_PSM3 AND LIBFABRIC_PSM3_HWLOC
                               )
)
  find_package(Hwloc)
endif()
if(LIBFABRIC_PROVIDER_PSM2)
  find_package(PSM2)
endif()
if(LIBFABRIC_PROVIDER_UCX)
  find_package(UCX)
  include(CheckUCX)
endif()
if(LIBFABRIC_PROVIDER_CXI)
  find_package(LibCxi)
  find_package(CXIDriverHeaders)
  include(CheckLibCxi)
  find_package(JsonC)
  find_package(CURL)
endif()
if(LIBFABRIC_PROVIDER_USNIC)
  find_package(LibNL)
endif()
