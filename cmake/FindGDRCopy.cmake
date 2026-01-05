# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_path(
  GDRCopy_INCLUDE_DIR
  NAMES gdrapi.h
  HINTS ${GDRCopy_ROOT} ENV GDRCopy_ROOT
  PATH_SUFFIXES include
)

find_library(
  GDRCopy_LIBRARY
  NAMES gdrapi
  HINTS ${GDRCopy_ROOT} ENV GDRCopy_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  GDRCopy REQUIRED_VARS GDRCopy_LIBRARY GDRCopy_INCLUDE_DIR
)

if(GDRCopy_FOUND)
  set(GDRCopy_INCLUDE_DIRS "${GDRCopy_INCLUDE_DIR}")
  set(GDRCopy_LIBRARIES "${GDRCopy_LIBRARY}")

  if(NOT TARGET GDRCopy::GDRCopy)
    add_library(GDRCopy::GDRCopy UNKNOWN IMPORTED)
    set_target_properties(
      GDRCopy::GDRCopy
      PROPERTIES IMPORTED_LOCATION "${GDRCopy_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${GDRCopy_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(GDRCopy_INCLUDE_DIR GDRCopy_LIBRARY)
