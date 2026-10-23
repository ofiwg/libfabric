# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindLibCxi
----------

Find the CXI userspace library and headers.

Imported targets: ``LibCxi::LibCxi``.

Results: ``LibCxi_FOUND``, ``LibCxi_INCLUDE_DIRS``, ``LibCxi_LIBRARIES``.

Hints: ``LibCxi_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``LibCxi_INCLUDE_DIR``
* ``LibCxi_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_path(
  LibCxi_INCLUDE_DIR
  NAMES libcxi/libcxi.h
  HINTS ${LibCxi_ROOT} ENV LibCxi_ROOT
  PATH_SUFFIXES include
)

find_library(
  LibCxi_LIBRARY
  NAMES cxi
  HINTS ${LibCxi_ROOT} ENV LibCxi_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  LibCxi REQUIRED_VARS LibCxi_LIBRARY LibCxi_INCLUDE_DIR
)

if(LibCxi_FOUND)
  set(LibCxi_INCLUDE_DIRS "${LibCxi_INCLUDE_DIR}")
  set(LibCxi_LIBRARIES "${LibCxi_LIBRARY}")

  if(NOT TARGET LibCxi::LibCxi)
    add_library(LibCxi::LibCxi UNKNOWN IMPORTED)
    set_target_properties(
      LibCxi::LibCxi
      PROPERTIES IMPORTED_LOCATION "${LibCxi_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${LibCxi_INCLUDE_DIR}"
    )
  endif()

endif()

mark_as_advanced(LibCxi_INCLUDE_DIR LibCxi_LIBRARY)
