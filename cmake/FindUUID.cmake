# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_UUID QUIET IMPORTED_TARGET uuid)
endif()

if(_UUID_FOUND)
  set(UUID_FOUND TRUE)
  set(UUID_INCLUDE_DIRS "${_UUID_INCLUDE_DIRS}")
  set(UUID_LIBRARIES "${_UUID_LIBRARIES}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET UUID::UUID)
    add_library(UUID::UUID ALIAS PkgConfig::_UUID)
  endif()
else()
  find_path(
    UUID_INCLUDE_DIR
    NAMES uuid/uuid.h
    HINTS ${UUID_ROOT} ENV UUID_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    UUID_LIBRARY
    NAMES uuid
    HINTS ${UUID_ROOT} ENV UUID_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    UUID REQUIRED_VARS UUID_LIBRARY UUID_INCLUDE_DIR
  )

  if(UUID_FOUND)
    set(UUID_INCLUDE_DIRS "${UUID_INCLUDE_DIR}")
    set(UUID_LIBRARIES "${UUID_LIBRARY}")

    if(NOT TARGET UUID::UUID)
      add_library(UUID::UUID UNKNOWN IMPORTED)
      set_target_properties(
        UUID::UUID
        PROPERTIES IMPORTED_LOCATION "${UUID_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${UUID_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(UUID_INCLUDE_DIR UUID_LIBRARY)
