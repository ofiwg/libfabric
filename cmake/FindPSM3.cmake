# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

# Try pkg-config first
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_PSM3 QUIET IMPORTED_TARGET libpsm3)
endif()

if(_PSM3_FOUND)
  set(PSM3_FOUND TRUE)
  set(PSM3_VERSION "${_PSM3_VERSION}")
  set(PSM3_INCLUDE_DIRS "${_PSM3_INCLUDE_DIRS}")
  set(PSM3_LIBRARIES "${_PSM3_LIBRARIES}")
  set(PSM3_LIBRARY_DIRS "${_PSM3_LIBRARY_DIRS}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET PSM3::PSM3)
    add_library(PSM3::PSM3 ALIAS PkgConfig::_PSM3)
  endif()
else()
  # Fallback: manual search
  find_path(
    PSM3_INCLUDE_DIR
    NAMES psm2.h
    HINTS ${PSM3_ROOT} ENV PSM3_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    PSM3_LIBRARY
    NAMES psm3
    HINTS ${PSM3_ROOT} ENV PSM3_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    PSM3 REQUIRED_VARS PSM3_LIBRARY PSM3_INCLUDE_DIR
  )

  if(PSM3_FOUND)
    set(PSM3_INCLUDE_DIRS "${PSM3_INCLUDE_DIR}")
    set(PSM3_LIBRARIES "${PSM3_LIBRARY}")

    if(NOT TARGET PSM3::PSM3)
      add_library(PSM3::PSM3 UNKNOWN IMPORTED)
      set_target_properties(
        PSM3::PSM3
        PROPERTIES IMPORTED_LOCATION "${PSM3_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${PSM3_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(PSM3_INCLUDE_DIR PSM3_LIBRARY)
