# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricInstall
----------------

Install libfabric, headers, consumer metadata, and manual pages.

Requires the completed fabric target and enabled provider headers. Includes
LibfabricPackageDependencies, generates relocatable CMake/pkg-config metadata,
and installs only the find modules required by static consumers.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()
install(
  TARGETS fabric
  EXPORT LibfabricTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  DIRECTORY include/rdma
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
)

install(FILES ${LIBFABRIC_PROVIDER_HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rdma"
)
if(WIN32)
  install(FILES include/windows/sys/uio.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sys"
  )
  install(FILES include/windows/poll.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
endif()

# A shared-library consumer needs only the public headers. Static consumers must
# rediscover the private libraries without embedding build-machine paths.
include(LibfabricPackageDependencies)
set(_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Libfabric")
configure_package_config_file(
  cmake/LibfabricConfig.cmake.in "${libfabric_BINARY_DIR}/LibfabricConfig.cmake"
  INSTALL_DESTINATION "${_config_dir}"
)
write_basic_package_version_file(
  "${libfabric_BINARY_DIR}/LibfabricConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)
install(
  EXPORT LibfabricTargets
  NAMESPACE Libfabric::
  DESTINATION "${_config_dir}"
)
install(FILES "${libfabric_BINARY_DIR}/LibfabricConfig.cmake"
              "${libfabric_BINARY_DIR}/LibfabricConfigVersion.cmake"
        DESTINATION "${_config_dir}"
)
# Only install modules needed by the exported static dependency graph.
if(NOT BUILD_SHARED_LIBS)
  foreach(_package IN LISTS LIBFABRIC_DEPENDENCIES)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/Find${_package}.cmake")
      install(FILES "${CMAKE_CURRENT_LIST_DIR}/Find${_package}.cmake"
              DESTINATION "${_config_dir}/modules"
      )
    endif()
  endforeach()
endif()

# pcfiledir makes a staged installation relocatable for relative install dirs.
file(RELATIVE_PATH _pc_prefix "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig"
     "${CMAKE_INSTALL_PREFIX}"
)
set(prefix "\${pcfiledir}/${_pc_prefix}")
set(exec_prefix "\${prefix}")
foreach(_dir lib include)
  if(_dir STREQUAL "lib")
    set(_install_dir "${CMAKE_INSTALL_LIBDIR}")
  else()
    set(_install_dir "${CMAKE_INSTALL_INCLUDEDIR}")
  endif()
  if(IS_ABSOLUTE "${_install_dir}")
    string(REPLACE " " "\\ " ${_dir}dir "${_install_dir}")
  else()
    string(REPLACE " " "\\ " ${_dir}dir "\${prefix}/${_install_dir}")
  endif()
endforeach()
set(PACKAGE_VERSION "${LIBFABRIC_VERSION_FULL}")
set(LIBFABRIC_PKGCONFIG_LINK_OPTIONS "")
if(LIBFABRIC_INSTRUMENTATION_LINK_OPTIONS)
  list(JOIN LIBFABRIC_INSTRUMENTATION_LINK_OPTIONS " "
       LIBFABRIC_PKGCONFIG_LINK_OPTIONS
  )
endif()
configure_file(cmake/libfabric.pc.in libfabric.pc.in @ONLY)
file(
  GENERATE
  OUTPUT "${libfabric_BINARY_DIR}/pkgconfig/$<CONFIG>/libfabric.pc"
  INPUT "${libfabric_BINARY_DIR}/libfabric.pc.in"
)
install(FILES "${libfabric_BINARY_DIR}/pkgconfig/$<CONFIG>/libfabric.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)

# Documentation is shipped pre-generated; source compilation needs no renderer.
foreach(_section 1 3 7)
  file(GLOB _pages "${libfabric_SOURCE_DIR}/man/man${_section}/*.${_section}")
  install(FILES ${_pages} DESTINATION "${CMAKE_INSTALL_MANDIR}/man${_section}")
endforeach()
