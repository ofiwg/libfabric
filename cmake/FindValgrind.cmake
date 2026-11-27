# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindValgrind
------------

Find Valgrind annotation headers; no runtime library is required.

Imported targets: ``Valgrind::Valgrind``.

Results: ``Valgrind_FOUND``, ``Valgrind_INCLUDE_DIRS``.

Hints: ``Valgrind_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``Valgrind_INCLUDE_DIR``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_path(
  Valgrind_INCLUDE_DIR
  NAMES valgrind/memcheck.h
  HINTS ${Valgrind_ROOT} ENV Valgrind_ROOT
  PATH_SUFFIXES include
)

find_package_handle_standard_args(Valgrind REQUIRED_VARS Valgrind_INCLUDE_DIR)

if(Valgrind_FOUND)
  set(Valgrind_INCLUDE_DIRS "${Valgrind_INCLUDE_DIR}")

  if(NOT TARGET Valgrind::Valgrind)
    add_library(Valgrind::Valgrind INTERFACE IMPORTED)
    set_target_properties(
      Valgrind::Valgrind PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                    "${Valgrind_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(Valgrind_INCLUDE_DIR)
