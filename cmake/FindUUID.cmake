# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindUUID
--------

Find the UUID library and headers.

Imported targets: ``UUID::UUID``.

Results: ``UUID_FOUND``, ``UUID_INCLUDE_DIRS``, ``UUID_LIBRARIES``.

``UUID_VERSION`` is available when selected-library metadata supplies it.
Version requests fail if a suitable version cannot be established.

Hints: ``UUID_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``UUID_INCLUDE_DIR``
* ``UUID_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_UUID QUIET IMPORTED_TARGET uuid)
endif()

# Let CMake honor package roots and explicit cache overrides before pkg-config
# hints. Only reuse pkg-config metadata when it describes the selected library.
find_path(
  UUID_INCLUDE_DIR
  NAMES uuid/uuid.h
  HINTS ${_UUID_INCLUDE_DIRS}
)
find_library(
  UUID_LIBRARY
  NAMES uuid
  HINTS ${_UUID_LIBRARY_DIRS}
)
set(UUID_INCLUDE_DIRS "${UUID_INCLUDE_DIR}")
set(UUID_LIBRARIES "${UUID_LIBRARY}")
set(UUID_VERSION "")
set(_UUID_use_pkgconfig FALSE)
if(PkgConfig_FOUND
   AND _UUID_FOUND
   AND UUID_LIBRARY IN_LIST _UUID_LINK_LIBRARIES
)
  set(_UUID_use_pkgconfig TRUE)
  set(UUID_VERSION "${_UUID_VERSION}")
  set(UUID_LIBRARIES "${_UUID_LINK_LIBRARIES}")
endif()

find_package_handle_standard_args(
  UUID
  REQUIRED_VARS UUID_LIBRARY UUID_INCLUDE_DIR
  VERSION_VAR UUID_VERSION HANDLE_VERSION_RANGE
)
if(UUID_FOUND AND NOT TARGET UUID::UUID)
  if(_UUID_use_pkgconfig)
    add_library(UUID::UUID INTERFACE IMPORTED)
    target_link_libraries(UUID::UUID INTERFACE PkgConfig::_UUID)
  else()
    add_library(UUID::UUID UNKNOWN IMPORTED)
    set_target_properties(
      UUID::UUID PROPERTIES IMPORTED_LOCATION "${UUID_LIBRARY}"
    )
  endif()
  set_target_properties(
    UUID::UUID PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${UUID_INCLUDE_DIR}"
  )
endif()
mark_as_advanced(UUID_INCLUDE_DIR UUID_LIBRARY)
