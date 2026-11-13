# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindDSA
-------

Find Intel Data Streaming Accelerator configuration headers.

Results: ``DSA_FOUND``, ``DSA_INCLUDE_DIRS``.

Hints: ``DSA_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``DSA_INCLUDE_DIR``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_path(
  DSA_INCLUDE_DIR
  NAMES accel-config/libaccel_config.h
  HINTS ${DSA_ROOT} ENV DSA_ROOT
  PATH_SUFFIXES include
)

find_package_handle_standard_args(DSA REQUIRED_VARS DSA_INCLUDE_DIR)

if(DSA_FOUND)
  set(DSA_INCLUDE_DIRS "${DSA_INCLUDE_DIR}")
endif()

mark_as_advanced(DSA_INCLUDE_DIR)
