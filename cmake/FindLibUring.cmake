# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindLibUring
------------

Find io_uring headers and the liburing library.

Imported targets: ``LibUring::LibUring``.

Results: ``LibUring_FOUND``, ``LibUring_INCLUDE_DIRS``,
``LibUring_LIBRARIES``.

libfabric separately checks the required multishot API before enabling
io_uring; finding the library alone does not establish that capability.

``LibUring_VERSION`` is available when selected-library metadata supplies it.
Version requests fail if a suitable version cannot be established.

Hints: ``LibUring_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``LibUring_INCLUDE_DIR``
* ``LibUring_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_LIBURING QUIET IMPORTED_TARGET liburing>=2.1)
endif()

# Let CMake honor package roots and explicit cache overrides before pkg-config
# hints. Only reuse pkg-config metadata when it describes the selected library.
find_path(
  LibUring_INCLUDE_DIR
  NAMES liburing.h
  HINTS ${_LIBURING_INCLUDE_DIRS}
)
find_library(
  LibUring_LIBRARY
  NAMES uring
  HINTS ${_LIBURING_LIBRARY_DIRS}
)
set(LibUring_INCLUDE_DIRS "${LibUring_INCLUDE_DIR}")
set(LibUring_LIBRARIES "${LibUring_LIBRARY}")
set(LibUring_VERSION "")
set(_LibUring_use_pkgconfig FALSE)
if(PkgConfig_FOUND
   AND _LIBURING_FOUND
   AND LibUring_LIBRARY IN_LIST _LIBURING_LINK_LIBRARIES
)
  set(_LibUring_use_pkgconfig TRUE)
  set(LibUring_VERSION "${_LIBURING_VERSION}")
  set(LibUring_LIBRARIES "${_LIBURING_LINK_LIBRARIES}")
endif()

find_package_handle_standard_args(
  LibUring
  REQUIRED_VARS LibUring_LIBRARY LibUring_INCLUDE_DIR
  VERSION_VAR LibUring_VERSION HANDLE_VERSION_RANGE
)
if(LibUring_FOUND AND NOT TARGET LibUring::LibUring)
  if(_LibUring_use_pkgconfig)
    add_library(LibUring::LibUring INTERFACE IMPORTED)
    target_link_libraries(LibUring::LibUring INTERFACE PkgConfig::_LIBURING)
  else()
    add_library(LibUring::LibUring UNKNOWN IMPORTED)
    set_target_properties(
      LibUring::LibUring PROPERTIES IMPORTED_LOCATION "${LibUring_LIBRARY}"
    )
  endif()
  set_target_properties(
    LibUring::LibUring PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                  "${LibUring_INCLUDE_DIR}"
  )
endif()
mark_as_advanced(LibUring_INCLUDE_DIR LibUring_LIBRARY)
