# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_path(
  PSM2_INCLUDE_DIR
  NAMES psm2.h
  HINTS ${PSM2_ROOT} ENV PSM2_ROOT
  PATH_SUFFIXES include
)

find_library(
  PSM2_LIBRARY
  NAMES psm2
  HINTS ${PSM2_ROOT} ENV PSM2_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  PSM2 REQUIRED_VARS PSM2_LIBRARY PSM2_INCLUDE_DIR
)

if(PSM2_FOUND)
  set(PSM2_INCLUDE_DIRS "${PSM2_INCLUDE_DIR}")
  set(PSM2_LIBRARIES "${PSM2_LIBRARY}")

  if(NOT TARGET PSM2::PSM2)
    add_library(PSM2::PSM2 UNKNOWN IMPORTED)
    set_target_properties(
      PSM2::PSM2 PROPERTIES IMPORTED_LOCATION "${PSM2_LIBRARY}"
                            INTERFACE_INCLUDE_DIRECTORIES "${PSM2_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(PSM2_INCLUDE_DIR PSM2_LIBRARY)
