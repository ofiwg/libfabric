# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_HWLOC QUIET IMPORTED_TARGET hwloc)
endif()

if(_HWLOC_FOUND)
  set(Hwloc_FOUND TRUE)
  set(Hwloc_VERSION "${_HWLOC_VERSION}")
  set(Hwloc_INCLUDE_DIRS "${_HWLOC_INCLUDE_DIRS}")
  set(Hwloc_LIBRARIES "${_HWLOC_LIBRARIES}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET Hwloc::Hwloc)
    add_library(Hwloc::Hwloc ALIAS PkgConfig::_HWLOC)
  endif()
else()
  find_path(
    Hwloc_INCLUDE_DIR
    NAMES hwloc.h
    HINTS ${Hwloc_ROOT} ENV Hwloc_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    Hwloc_LIBRARY
    NAMES hwloc
    HINTS ${Hwloc_ROOT} ENV Hwloc_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    Hwloc REQUIRED_VARS Hwloc_LIBRARY Hwloc_INCLUDE_DIR
  )

  if(Hwloc_FOUND)
    set(Hwloc_INCLUDE_DIRS "${Hwloc_INCLUDE_DIR}")
    set(Hwloc_LIBRARIES "${Hwloc_LIBRARY}")

    if(NOT TARGET Hwloc::Hwloc)
      add_library(Hwloc::Hwloc UNKNOWN IMPORTED)
      set_target_properties(
        Hwloc::Hwloc
        PROPERTIES IMPORTED_LOCATION "${Hwloc_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${Hwloc_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(Hwloc_INCLUDE_DIR Hwloc_LIBRARY)
