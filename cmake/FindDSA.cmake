# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_path(
  DSA_INCLUDE_DIR
  NAMES accel-config/libaccel_config.h
  HINTS ${DSA_ROOT} ENV DSA_ROOT
  PATH_SUFFIXES include
)

find_library(
  DSA_LIBRARY
  NAMES accel-config
  HINTS ${DSA_ROOT} ENV DSA_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(DSA REQUIRED_VARS DSA_LIBRARY DSA_INCLUDE_DIR)

if(DSA_FOUND)
  set(DSA_INCLUDE_DIRS "${DSA_INCLUDE_DIR}")
  set(DSA_LIBRARIES "${DSA_LIBRARY}")

  if(NOT TARGET DSA::DSA)
    add_library(DSA::DSA UNKNOWN IMPORTED)
    set_target_properties(
      DSA::DSA PROPERTIES IMPORTED_LOCATION "${DSA_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${DSA_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(DSA_INCLUDE_DIR DSA_LIBRARY)
