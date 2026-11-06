# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindLibNL
---------

Find a complete libnl ABI generation, preferring v3 over v1.

Imported targets: ``LibNL::LibNL``.

Results: ``LibNL_FOUND``, ``LibNL_INCLUDE_DIRS``, ``LibNL_LIBRARIES``.

``LibNL_IS_V3`` and ``LibNL_VERSION_MAJOR`` identify the selected ABI.
Both the v3 core and route libraries are required; generations are never
mixed.

Hints: ``LibNL_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``LibNL3_INCLUDE_DIR``
* ``LibNL3_LIBRARY``
* ``LibNL3_ROUTE_LIBRARY``
* ``LibNL1_INCLUDE_DIR``
* ``LibNL1_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_LIBNL3 QUIET libnl-3.0)
  pkg_check_modules(_LIBNL3_ROUTE QUIET libnl-route-3.0)
endif()

# The two ABI generations have overlapping symbol names. Select a complete
# generation, preferring v3, without pairing v1 headers with v3 libraries.
find_path(
  LibNL3_INCLUDE_DIR
  NAMES netlink/version.h
  HINTS ${LibNL_ROOT} ENV LibNL_ROOT ${_LIBNL3_INCLUDE_DIRS}
  PATH_SUFFIXES include/libnl3 include
)
find_library(
  LibNL3_LIBRARY
  NAMES nl-3
  HINTS ${LibNL_ROOT} ENV LibNL_ROOT ${_LIBNL3_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)
find_library(
  LibNL3_ROUTE_LIBRARY
  NAMES nl-route-3
  HINTS ${LibNL_ROOT} ENV LibNL_ROOT ${_LIBNL3_ROUTE_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)
set(_libnl3_headers FALSE)
if(LibNL3_INCLUDE_DIR)
  file(STRINGS "${LibNL3_INCLUDE_DIR}/netlink/version.h" _version
       REGEX "^#define[ \t]+LIBNL_VER_MAJ[ \t]+3([ \t]|$)"
  )
  if(_version)
    set(_libnl3_headers TRUE)
  endif()
endif()

set(LibNL_IS_V3 FALSE)
set(LibNL_VERSION_MAJOR 1)
if(_libnl3_headers
   AND LibNL3_LIBRARY
   AND LibNL3_ROUTE_LIBRARY
)
  set(LibNL_IS_V3 TRUE)
  set(LibNL_VERSION_MAJOR 3)
  set(LibNL_INCLUDE_DIR "${LibNL3_INCLUDE_DIR}")
  set(LibNL_LIBRARIES "${LibNL3_LIBRARY};${LibNL3_ROUTE_LIBRARY}")
else()
  find_path(
    LibNL1_INCLUDE_DIR
    NAMES netlink/netlink.h
    HINTS ${LibNL_ROOT} ENV LibNL_ROOT
    PATH_SUFFIXES include
  )
  find_library(
    LibNL1_LIBRARY
    NAMES nl
    HINTS ${LibNL_ROOT} ENV LibNL_ROOT
    PATH_SUFFIXES lib lib64
  )
  # A v3 include directory can be supplied directly in CMAKE_PREFIX_PATH. Reject
  # it here instead of silently treating its declarations as v1.
  set(_libnl1_headers "${LibNL1_INCLUDE_DIR}")
  if(EXISTS "${LibNL1_INCLUDE_DIR}/netlink/version.h")
    file(STRINGS "${LibNL1_INCLUDE_DIR}/netlink/version.h" _version
         REGEX "^#define[ \t]+LIBNL_VER_MAJ[ \t]+[0-9]+"
    )
    if(_version MATCHES "LIBNL_VER_MAJ[ \t]+([0-9]+)" AND NOT CMAKE_MATCH_1
                                                          EQUAL 1
    )
      set(_libnl1_headers FALSE)
    endif()
  endif()
  set(LibNL_INCLUDE_DIR "${_libnl1_headers}")
  set(LibNL_LIBRARIES "${LibNL1_LIBRARY};m")
endif()

# Keep the selected main library separate so a missing path cannot be hidden
# inside the multi-library list passed to the standard argument handler.
if(LibNL_IS_V3)
  set(LibNL_LIBRARY "${LibNL3_LIBRARY}")
else()
  set(LibNL_LIBRARY "${LibNL1_LIBRARY}")
endif()
find_package_handle_standard_args(
  LibNL
  REQUIRED_VARS LibNL_LIBRARY LibNL_INCLUDE_DIR
  VERSION_VAR LibNL_VERSION_MAJOR
)
if(LibNL_FOUND)
  set(LibNL_INCLUDE_DIRS "${LibNL_INCLUDE_DIR}")
  if(NOT TARGET LibNL::LibNL)
    add_library(LibNL::LibNL INTERFACE IMPORTED)
    set_target_properties(
      LibNL::LibNL
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LibNL_INCLUDE_DIR}"
                 INTERFACE_LINK_LIBRARIES "${LibNL_LIBRARIES}"
    )
  endif()
endif()
mark_as_advanced(
  LibNL1_INCLUDE_DIR
  LibNL1_LIBRARY
  LibNL3_INCLUDE_DIR
  LibNL3_LIBRARY
  LibNL3_ROUTE_LIBRARY
)
