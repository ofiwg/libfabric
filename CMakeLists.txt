# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/LibfabricVersion.cmake")
project(
  libfabric
  VERSION ${LIBFABRIC_VERSION_NUMBER}
  DESCRIPTION "User-space RDMA Fabric Interfaces"
  LANGUAGES C
)
if(WIN32)
  enable_language(CXX)
  # Executables and their DLL must share a directory in the build tree.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${libfabric_BINARY_DIR}/bin/$<CONFIG>")
endif()
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CMakePushCheckState)
# A parent project's temporary probe settings are not libfabric requirements.
cmake_push_check_state(RESET)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(LibfabricOptions)
include(LibfabricPlatform)
include(LibfabricTargetSettings)
set(_libfabric_ipo_requested "${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES CMAKE_BUILD_TYPE)
  string(TOUPPER "${_config}" _config)
  if(CMAKE_INTERPROCEDURAL_OPTIMIZATION_${_config})
    set(_libfabric_ipo_requested TRUE)
  endif()
endforeach()
if(_libfabric_ipo_requested)
  include(CheckIPOSupported)
  check_ipo_supported(
    RESULT _ipo_supported
    OUTPUT _ipo_error
    LANGUAGES C
  )
  if(NOT _ipo_supported)
    message(FATAL_ERROR "LTO requested but unavailable: ${_ipo_error}")
  endif()
endif()

# Probe with the same language and feature-test macros as the library. These
# defaults are local to this directory, including in a parent project.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
include(LibfabricDependencies)
include(LibfabricFeatureDetection)

# Internal usage requirements are never part of the installed public API.
add_library(libfabric_build_settings INTERFACE)
target_compile_features(libfabric_build_settings INTERFACE c_std_11)
target_compile_definitions(
  libfabric_build_settings
  INTERFACE
    HAVE_CONFIG_H=1
    ENABLE_DEBUG=$<IF:$<CONFIG:Debug>,1,0>
    ${LIBFABRIC_PLATFORM_DEFINITIONS}
    SYSCONFDIR="${CMAKE_INSTALL_FULL_SYSCONFDIR}"
    RDMADIR="${CMAKE_INSTALL_FULL_SYSCONFDIR}/rdma"
    PROVDLDIR="${CMAKE_INSTALL_FULL_LIBDIR}/libfabric"
)
target_include_directories(
  libfabric_build_settings
  INTERFACE
    "${libfabric_BINARY_DIR}"
    "${libfabric_SOURCE_DIR}/include"
    ${LIBFABRIC_PLATFORM_INCLUDE_DIRS}
    "${libfabric_SOURCE_DIR}/src"
    "${libfabric_SOURCE_DIR}/prov/util/src"
    "${libfabric_SOURCE_DIR}/prov/coll/src"
    "${libfabric_SOURCE_DIR}/prov/hook/include"
    "${libfabric_SOURCE_DIR}/prov/hook/src"
)
libfabric_configure_instrumentation(libfabric_build_settings)
get_target_property(
  LIBFABRIC_INSTRUMENTATION_LINK_OPTIONS libfabric_build_settings
  INTERFACE_LINK_OPTIONS
)
include(LibfabricHmem)
foreach(_package Kdreg2 Valgrind)
  if(${_package}_FOUND)
    target_include_directories(
      libfabric_build_settings SYSTEM INTERFACE ${${_package}_INCLUDE_DIRS}
    )
  endif()
endforeach()

include(LibfabricSources)
# The common objects are also needed by plugins: their internal symbols are
# deliberately hidden from libfabric's public ABI.
add_library(libfabric_common OBJECT ${FABRIC_COMMON_SOURCES})
libfabric_configure_target(libfabric_common)
# Keep .symver assembly and its API definitions in the same translation unit.
# LTO can repartition them into separate units even with externally_visible.
add_library(libfabric_api OBJECT ${FABRIC_PUBLIC_SOURCES})
libfabric_configure_target(libfabric_api)
set_property(TARGET libfabric_api PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES CMAKE_BUILD_TYPE)
  string(TOUPPER "${_config}" _config)
  set_property(
    TARGET libfabric_api PROPERTY INTERPROCEDURAL_OPTIMIZATION_${_config} OFF
  )
endforeach()
add_library(libfabric_hooks OBJECT ${FABRIC_HOOK_SOURCES})
libfabric_configure_target(libfabric_hooks)
add_library(
  fabric $<TARGET_OBJECTS:libfabric_hooks> $<TARGET_OBJECTS:libfabric_api>
         $<TARGET_OBJECTS:libfabric_common>
)
add_library(Libfabric::fabric ALIAS fabric)
if(WIN32)
  target_include_directories(
    fabric PUBLIC $<BUILD_INTERFACE:${libfabric_SOURCE_DIR}/include/windows>
  )
endif()
if(LIBFABRIC_INSTRUMENTATION_LINK_OPTIONS)
  target_link_options(
    fabric INTERFACE ${LIBFABRIC_INSTRUMENTATION_LINK_OPTIONS}
  )
endif()
libfabric_configure_target(fabric)
if(WIN32 AND BUILD_SHARED_LIBS)
  target_sources(fabric PRIVATE libfabric.def)
endif()
target_include_directories(
  fabric
  PUBLIC $<BUILD_INTERFACE:${libfabric_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${libfabric_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(
  fabric PROPERTIES VERSION "${LIBFABRIC_ABI_VERSION}" SOVERSION
                                                       "${LIBFABRIC_SOVERSION}"
)

set(LIBFABRIC_CORE_LIBRARIES
    Threads::Threads
    ${CMAKE_DL_LIBS}
    ${LIBFABRIC_RT_LIBRARIES}
    ${LIBFABRIC_HMEM_LIBRARIES}
    ${Atomics_LIBRARIES}
)
foreach(_dep LibUring LTTng XPMEM)
  if(${_dep}_FOUND)
    list(APPEND LIBFABRIC_CORE_LIBRARIES ${_dep}::${_dep})
  endif()
endforeach()
if(WIN32)
  list(
    APPEND
    LIBFABRIC_CORE_LIBRARIES
    ws2_32
    iphlpapi
    synchronization
  )
endif()
target_link_libraries(
  libfabric_build_settings INTERFACE ${LIBFABRIC_CORE_LIBRARIES}
)
target_link_libraries(fabric PRIVATE ${LIBFABRIC_CORE_LIBRARIES})

include(LibfabricProvider)
add_subdirectory(prov)
set(LIBFABRIC_PROVIDER_HEADERS "")
foreach(_entry efa/src/fi_ext_efa.h usnic/src/fi_ext_usnic.h
               cxi/include/fi_cxi_ext.h psm2/include/fi_ext_psm2.h
)
  string(REGEX REPLACE "/.*" "" _provider "${_entry}")
  string(TOUPPER "${_provider}" _provider)
  get_filename_component(_header "${_entry}" NAME)
  if(HAVE_${_provider})
    configure_file("prov/${_entry}" "include/rdma/${_header}" COPYONLY)
    list(APPEND LIBFABRIC_PROVIDER_HEADERS
         "${libfabric_SOURCE_DIR}/prov/${_entry}"
    )
  else()
    file(REMOVE "${libfabric_BINARY_DIR}/include/rdma/${_header}")
  endif()
endforeach()
configure_file(cmake/config.h.in config.h @ONLY)

if(BUILD_SHARED_LIBS AND HAVE_SYMVER_SUPPORT)
  set(FI_DIRECT_PROVIDER_API_10 "")
  configure_file(libfabric.map.in libfabric.map @ONLY)
  target_link_options(
    fabric
    PRIVATE "LINKER:--version-script=${libfabric_BINARY_DIR}/libfabric.map"
  )
  # Clang's sanitizer runtimes are supplied by the final executable.
  if(NOT LIBFABRIC_INSTRUMENTATION_LINK_OPTIONS MATCHES "fsanitize=")
    target_link_options(fabric PRIVATE "LINKER:--no-undefined")
  endif()
  set_property(
    TARGET fabric
    APPEND
    PROPERTY LINK_DEPENDS "${libfabric_BINARY_DIR}/libfabric.map"
  )
endif()

if(LIBFABRIC_BUILD_UTILITIES)
  add_subdirectory(util)
endif()
if(LIBFABRIC_BUILD_FABTESTS)
  add_subdirectory(fabtests)
endif()
if(LIBFABRIC_INSTALL)
  include(LibfabricInstall)
endif()
if(LIBFABRIC_BUILD_TESTING)
  enable_testing()
  add_subdirectory(cmake/tests)
endif()
cmake_pop_check_state()
message(
  STATUS "libfabric ${LIBFABRIC_VERSION_FULL}: ${LIBFABRIC_ENABLED_PROVIDERS}"
)
