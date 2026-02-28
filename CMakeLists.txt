# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
cmake_minimum_required(VERSION 3.16)

# Version from configure.ac: AC_INIT([libfabric], [2.5.0a1], ...) CMake's
# VERSION only supports numeric components (major.minor.patch.tweak) so we store
# the suffix separately for display purposes
project(
  libfabric
  VERSION 2.5.0
  DESCRIPTION "User-space RDMA Fabric Interfaces"
  HOMEPAGE_URL "https://github.com/ofiwg/libfabric"
  LANGUAGES C
)

# Version suffix (e.g., "a1" for alpha, "rc1" for release candidate, "" for
# release)
set(LIBFABRIC_VERSION_SUFFIX "a1")

# Full version string including suffix
if(LIBFABRIC_VERSION_SUFFIX)
  set(LIBFABRIC_VERSION_FULL "${PROJECT_VERSION}${LIBFABRIC_VERSION_SUFFIX}")
else()
  set(LIBFABRIC_VERSION_FULL "${PROJECT_VERSION}")
endif()

# Enable C++ on Windows for getopt implementation in utilities
if(WIN32)
  enable_language(CXX)
endif()

# C11 required (for static_assert at file scope, used by OPX and other
# providers)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON) # For GNU extensions

# Enable position-independent code for all targets Required for shared library
# builds and recommended for static libraries that may be linked into shared
# libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Link Time Optimization (LTO/IPO) - enabled by default if supported
if(LIBFABRIC_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _lto_supported OUTPUT _lto_error)
  if(_lto_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "LTO: enabled")
  else()
    message(STATUS "LTO: not supported by toolchain")
  endif()
else()
  message(STATUS "LTO: disabled")
endif()

# RPATH settings for executables to find shared libraries Build tree: find
# libraries in the build output directory
set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")
# Install tree: macOS needs RPATH, Linux relies on ldconfig or LD_LIBRARY_PATH
# Developers can pass -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' for relocatable
# installs
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Standard includes
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FeatureSummary)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)

# Our modules
include(CMakeOptions) # All user options (must be first)
include(LibfabricPlatform) # Platform detection
include(LibfabricHelpers) # Helper functions
include(LibfabricProvider) # Provider macros
include(LibfabricFeatureDetection) # Feature detection
include(LibfabricFeatureSummary) # Feature summary utilities

# Configure sanitizers and warnings (must be before target definitions)
libfabric_configure_sanitizers()
libfabric_configure_picky_warnings()

# Required
find_package(Threads REQUIRED)
find_package(Atomics REQUIRED)

# Platform-specific required
if(LIBFABRIC_LINUX)
  find_package(DL REQUIRED)
  find_package(RT)
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
endif()

# LTTng
if(LIBFABRIC_ENABLE_LTTNG)
  find_package(LTTng REQUIRED)
endif()

# Valgrind
if(LIBFABRIC_ENABLE_VALGRIND)
  find_package(Valgrind REQUIRED)
endif()

# GPU/Accelerator support
if(LIBFABRIC_ENABLE_CUDA)
  find_package(CUDAToolkit)
  set_package_properties(
    CUDAToolkit PROPERTIES
    TYPE OPTIONAL
    PURPOSE "CUDA GPU memory support"
  )
  if(LIBFABRIC_ENABLE_GDRCOPY AND CUDAToolkit_FOUND)
    find_package(GDRCopy)
  endif()
endif()

if(LIBFABRIC_ENABLE_ZE)
  find_package(LevelZero)
  set_package_properties(
    LevelZero PROPERTIES
    TYPE OPTIONAL
    PURPOSE "Intel GPU memory support"
  )
endif()

if(LIBFABRIC_ENABLE_ROCR)
  find_package(ROCm)
  set_package_properties(
    ROCm PROPERTIES
    TYPE OPTIONAL
    PURPOSE "AMD GPU memory support"
  )
endif()

if(LIBFABRIC_ENABLE_NEURON)
  find_package(Neuron)
endif()

if(LIBFABRIC_ENABLE_SYNAPSEAI)
  find_package(SynapseAI)
endif()

# Provider dependencies (checked when providers are configured)
find_package(IBVerbs)
find_package(RdmaCM)
find_package(Efadv)
find_package(Numa)
find_package(UUID)
find_package(Hwloc)
find_package(PSM2)
find_package(PSM3)
find_package(UCX)
find_package(LibCxi)
find_package(CXIDriverHeaders)
find_package(JsonC)
find_package(LibCurl)
find_package(DSA)
find_package(LibNL)

# =============================================================================
# Feature Detection
# =============================================================================
# Sets HAVE_* variables for config.h generation based on find_package() results.
# See cmake/LibfabricFeatureDetection.cmake for implementation details.

libfabric_detect_features()

# =============================================================================
# Core Library Sources
# =============================================================================

set(FABRIC_CORE_SOURCES
    src/fabric.c
    src/fi_tostr.c
    src/perf.c
    src/log.c
    src/var.c
    src/abi_1_0.c
    src/hmem.c
    src/hmem_rocr.c
    src/hmem_cuda.c
    src/hmem_cuda_gdrcopy.c
    src/hmem_ze.c
    src/hmem_neuron.c
    src/hmem_synapseai.c
    src/hmem_ipc_cache.c
    src/xpmem.c
    src/xpmem_cache.c
    src/common.c
    src/enosys.c
    src/rbtree.c
    src/tree.c
    src/fasthash.c
    src/indexer.c
    src/mem.c
    src/iov.c
    src/ofi_str.c
)

# Add io_uring source if available
if(HAVE_LIBURING)
  list(APPEND FABRIC_CORE_SOURCES src/iouring.c)
endif()

# Add rdpmc source if available
if(HAVE_LINUX_PERF_RDPMC)
  list(APPEND FABRIC_CORE_SOURCES src/linux/rdpmc.c)
endif()

# Utility provider sources (always included)
set(FABRIC_UTIL_SOURCES
    prov/util/src/util_atomic.c
    prov/util/src/util_attr.c
    prov/util/src/util_av.c
    prov/util/src/rxm_av.c
    prov/util/src/util_cq.c
    prov/util/src/util_cntr.c
    prov/util/src/util_domain.c
    prov/util/src/util_ep.c
    prov/util/src/util_pep.c
    prov/util/src/util_eq.c
    prov/util/src/util_fabric.c
    prov/util/src/util_main.c
    prov/util/src/util_poll.c
    prov/util/src/util_wait.c
    prov/util/src/util_buf.c
    prov/util/src/util_mr_map.c
    prov/util/src/util_ns.c
    prov/util/src/util_srx.c
    prov/util/src/util_mem_monitor.c
    prov/util/src/util_mem_hooks.c
    prov/util/src/util_mr_cache.c
    prov/util/src/cuda_mem_monitor.c
    prov/util/src/rocr_mem_monitor.c
    prov/util/src/ze_mem_monitor.c
    prov/util/src/cuda_ipc_monitor.c
    prov/util/src/rocr_ipc_monitor.c
    prov/util/src/ze_ipc_monitor.c
    prov/util/src/xpmem_monitor.c
    prov/util/src/kdreg2_mem_monitor.c
    prov/util/src/uffd_mem_monitor.c
    prov/util/src/import_mem_monitor.c
    prov/util/src/util_profile.c
)

# Hook infrastructure sources (always included)
set(FABRIC_HOOK_SOURCES
    prov/hook/src/hook.c
    prov/hook/src/hook_av.c
    prov/hook/src/hook_cm.c
    prov/hook/src/hook_cntr.c
    prov/hook/src/hook_cq.c
    prov/hook/src/hook_domain.c
    prov/hook/src/hook_ep.c
    prov/hook/src/hook_eq.c
    prov/hook/src/hook_wait.c
    prov/hook/src/hook_xfer.c
)

# Collective infrastructure sources (always included)
set(FABRIC_COLL_SOURCES
    prov/coll/src/coll_attr.c
    prov/coll/src/coll_av.c
    prov/coll/src/coll_av_set.c
    prov/coll/src/coll_coll.c
    prov/coll/src/coll_cq.c
    prov/coll/src/coll_domain.c
    prov/coll/src/coll_ep.c
    prov/coll/src/coll_eq.c
    prov/coll/src/coll_fabric.c
    prov/coll/src/coll_init.c
)

# =============================================================================
# Configure Providers
# =============================================================================

# Interface library for hook provider includes needed by core hook code Hook
# providers add their include directories to this target
add_library(libfabric_hook_includes INTERFACE)

add_subdirectory(prov)

# Get accumulated provider information
libfabric_get_provider_object_libs(PROVIDER_OBJECT_LIBS)
libfabric_get_provider_link_libraries(PROVIDER_LINK_LIBRARIES)

# =============================================================================
# Generate Configuration Header
# =============================================================================
# NOTE: This must come after providers are configured so that HAVE_* variables
# are set correctly for the config.h generation.

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/config.h.in" "${CMAKE_BINARY_DIR}/config.h" @ONLY
)

# =============================================================================
# Build Core Library
# =============================================================================

# Combine all core sources (providers are now object libraries)
set(FABRIC_CORE_ALL_SOURCES
    ${FABRIC_CORE_SOURCES} ${FABRIC_UTIL_SOURCES} ${FABRIC_HOOK_SOURCES}
    ${FABRIC_COLL_SOURCES} ${LIBFABRIC_PLATFORM_SOURCES}
)

# Build list of $<TARGET_OBJECTS:...> generator expressions for provider object
# libraries This avoids export issues - OBJECT libraries are build-time only and
# should not be exported
set(PROVIDER_OBJECT_SOURCES "")
if(PROVIDER_OBJECT_LIBS)
  foreach(_objlib IN LISTS PROVIDER_OBJECT_LIBS)
    list(APPEND PROVIDER_OBJECT_SOURCES $<TARGET_OBJECTS:${_objlib}>)
  endforeach()
endif()

# Create shared library
if(BUILD_SHARED_LIBS)
  add_library(
    fabric SHARED ${FABRIC_CORE_ALL_SOURCES} ${PROVIDER_OBJECT_SOURCES}
  )
  set_target_properties(
    fabric
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION 1
               OUTPUT_NAME fabric
               EXPORT_NAME fabric
  )

  # Symbol versioning
  if(HAVE_SYMVER_SUPPORT AND NOT LIBFABRIC_EMBEDDED)
    # Generate version script from template
    set(FI_DIRECT_PROVIDER_API_10 "")
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/libfabric.map.in
      ${CMAKE_CURRENT_BINARY_DIR}/libfabric.map @ONLY
    )
    target_link_options(
      fabric PRIVATE
      "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/libfabric.map"
    )
    set_target_properties(
      fabric PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libfabric.map
    )
  endif()

  add_library(Libfabric::fabric ALIAS fabric)
endif()

# Create static library
add_library(
  fabric_static STATIC ${FABRIC_CORE_ALL_SOURCES} ${PROVIDER_OBJECT_SOURCES}
)
set_target_properties(
  fabric_static PROPERTIES OUTPUT_NAME fabric EXPORT_NAME fabric_static
)
add_library(Libfabric::fabric_static ALIAS fabric_static)

# Configure targets
foreach(_target fabric fabric_static)
  if(NOT TARGET ${_target})
    continue()
  endif()

  target_include_directories(
    ${_target}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${LIBFABRIC_PLATFORM_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/prov/util/src
            ${CMAKE_CURRENT_SOURCE_DIR}/prov/coll/src
            ${CMAKE_CURRENT_SOURCE_DIR}/prov/hook/src
            ${CMAKE_CURRENT_SOURCE_DIR}/prov/hook/include
  )

  # Link interface library for hook provider includes needed by core hook code
  target_link_libraries(${_target} PRIVATE libfabric_hook_includes)

  target_compile_definitions(
    ${_target}
    PRIVATE ${LIBFABRIC_PLATFORM_DEFINITIONS}
            SYSCONFDIR="${CMAKE_INSTALL_FULL_SYSCONFDIR}"
            RDMADIR="${CMAKE_INSTALL_FULL_SYSCONFDIR}/rdma"
            PROVDLDIR="${CMAKE_INSTALL_FULL_LIBDIR}/libfabric"
  )

  target_link_libraries(
    ${_target}
    PRIVATE Threads::Threads
            $<$<BOOL:${DL_FOUND}>:DL::DL>
            $<$<BOOL:${RT_FOUND}>:RT::RT>
            $<$<BOOL:${HAVE_LIBURING}>:LibUring::LibUring>
            $<$<BOOL:${HAVE_LTTNG}>:LTTng::LTTng>
            $<$<BOOL:${HAVE_XPMEM}>:XPMEM::XPMEM>
            $<$<BOOL:${LIBFABRIC_WINDOWS}>:ws2_32>
            $<$<BOOL:${LIBFABRIC_WINDOWS}>:iphlpapi>
            $<$<BOOL:${LIBFABRIC_WINDOWS}>:synchronization>
            ${PROVIDER_LINK_LIBRARIES}
  )

  libfabric_set_visibility(${_target})
endforeach()

# =============================================================================
# Utilities
# =============================================================================

add_subdirectory(util)

# =============================================================================
# Fabtests
# =============================================================================

if(LIBFABRIC_BUILD_FABTESTS)
  # EFA GDA support for fabtests (requires CUDA)
  if(LIBFABRIC_FABTESTS_EFA_GDA AND CUDAToolkit_FOUND)
    set(HAVE_EFAGDA TRUE)
  endif()

  add_subdirectory(fabtests)
endif()

# =============================================================================
# Installation
# =============================================================================

# Determine which library target to install
if(BUILD_SHARED_LIBS)
  set(INSTALL_TARGETS fabric fabric_static)
else()
  set(INSTALL_TARGETS fabric_static)
endif()

# Libraries Note: libfabric_hook_includes is an internal interface library used
# during build only. It must be in the export set because fabric targets link to
# it, but it has no INSTALL_INTERFACE includes so it's effectively empty at
# install time.
install(
  TARGETS ${INSTALL_TARGETS} libfabric_hook_includes
  EXPORT libfabric-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Headers
install(
  DIRECTORY include/rdma
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rdma
  RENAME fabric_config.h
)

# CMake package config
configure_package_config_file(
  cmake/LibfabricConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/LibfabricConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libfabric
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/LibfabricConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  EXPORT libfabric-targets
  FILE LibfabricTargets.cmake
  NAMESPACE Libfabric::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libfabric
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LibfabricConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/LibfabricConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libfabric
)

# pkg-config
libfabric_add_pkgconfig()

# =============================================================================
# Man Pages
# =============================================================================

libfabric_install_man_pages()

# =============================================================================
# Summary
# =============================================================================

# Register package properties and features
libfabric_set_package_info()
libfabric_register_features()

# Print organized feature summary
libfabric_print_feature_summary()
