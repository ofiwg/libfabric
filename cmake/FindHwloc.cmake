# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindHwloc
---------

Find the hardware-locality library and headers.

Imported targets: ``Hwloc::Hwloc``.

Results: ``Hwloc_FOUND``, ``Hwloc_INCLUDE_DIRS``, ``Hwloc_LIBRARIES``.

``Hwloc_VERSION`` is available when selected-library metadata supplies it.
Version requests fail if a suitable version cannot be established.

Hints: ``Hwloc_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``Hwloc_INCLUDE_DIR``
* ``Hwloc_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_HWLOC QUIET IMPORTED_TARGET hwloc)
endif()

# Let CMake honor package roots and explicit cache overrides before pkg-config
# hints. Only reuse pkg-config metadata when it describes the selected library.
find_path(
  Hwloc_INCLUDE_DIR
  NAMES hwloc.h
  HINTS ${_HWLOC_INCLUDE_DIRS}
)
find_library(
  Hwloc_LIBRARY
  NAMES hwloc
  HINTS ${_HWLOC_LIBRARY_DIRS}
)
set(Hwloc_INCLUDE_DIRS "${Hwloc_INCLUDE_DIR}")
set(Hwloc_LIBRARIES "${Hwloc_LIBRARY}")
set(Hwloc_VERSION "")
set(_Hwloc_use_pkgconfig FALSE)
if(PkgConfig_FOUND
   AND _HWLOC_FOUND
   AND Hwloc_LIBRARY IN_LIST _HWLOC_LINK_LIBRARIES
)
  set(_Hwloc_use_pkgconfig TRUE)
  set(Hwloc_VERSION "${_HWLOC_VERSION}")
  set(Hwloc_LIBRARIES "${_HWLOC_LINK_LIBRARIES}")
endif()

find_package_handle_standard_args(
  Hwloc
  REQUIRED_VARS Hwloc_LIBRARY Hwloc_INCLUDE_DIR
  VERSION_VAR Hwloc_VERSION HANDLE_VERSION_RANGE
)
if(Hwloc_FOUND AND NOT TARGET Hwloc::Hwloc)
  if(_Hwloc_use_pkgconfig)
    add_library(Hwloc::Hwloc INTERFACE IMPORTED)
    target_link_libraries(Hwloc::Hwloc INTERFACE PkgConfig::_HWLOC)
  else()
    add_library(Hwloc::Hwloc UNKNOWN IMPORTED)
    set_target_properties(
      Hwloc::Hwloc PROPERTIES IMPORTED_LOCATION "${Hwloc_LIBRARY}"
    )
  endif()
  set_target_properties(
    Hwloc::Hwloc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                            "${Hwloc_INCLUDE_DIR}"
  )
endif()
mark_as_advanced(Hwloc_INCLUDE_DIR Hwloc_LIBRARY)
