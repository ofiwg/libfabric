# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindCXIDriverHeaders
--------------------

Find the CXI UAPI and Cassini provider hardware headers.

Imported targets: ``CXIDriverHeaders::CXIDriverHeaders``.

Results: ``CXIDriverHeaders_FOUND``, ``CXIDriverHeaders_INCLUDE_DIRS``.

Hints: ``CXIDriverHeaders_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``CXIDriverHeaders_INCLUDE_DIR``
* ``CXIDriverHeaders_CASSINI_INCLUDE_DIR``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_path(
  CXIDriverHeaders_INCLUDE_DIR
  NAMES uapi/misc/cxi.h
  HINTS ${CXIDriverHeaders_ROOT} ENV CXIDriverHeaders_ROOT
  PATH_SUFFIXES include
)

find_path(
  CXIDriverHeaders_CASSINI_INCLUDE_DIR
  NAMES cxi_prov_hw.h
  HINTS ${CXIDriverHeaders_ROOT} ENV CXIDriverHeaders_ROOT
  PATH_SUFFIXES include
)

find_package_handle_standard_args(
  CXIDriverHeaders REQUIRED_VARS CXIDriverHeaders_INCLUDE_DIR
                                 CXIDriverHeaders_CASSINI_INCLUDE_DIR
)

if(CXIDriverHeaders_FOUND)
  set(CXIDriverHeaders_INCLUDE_DIRS
      "${CXIDriverHeaders_INCLUDE_DIR};${CXIDriverHeaders_CASSINI_INCLUDE_DIR}"
  )

  if(NOT TARGET CXIDriverHeaders::CXIDriverHeaders)
    add_library(CXIDriverHeaders::CXIDriverHeaders INTERFACE IMPORTED)
    set_target_properties(
      CXIDriverHeaders::CXIDriverHeaders
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "${CXIDriverHeaders_INCLUDE_DIR};${CXIDriverHeaders_CASSINI_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(
  CXIDriverHeaders_INCLUDE_DIR CXIDriverHeaders_CASSINI_INCLUDE_DIR
)
