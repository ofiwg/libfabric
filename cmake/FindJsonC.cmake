# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_JSONC QUIET IMPORTED_TARGET json-c)
endif()

if(_JSONC_FOUND)
  set(JsonC_FOUND TRUE)
  set(JsonC_INCLUDE_DIRS "${_JSONC_INCLUDE_DIRS}")
  set(JsonC_LIBRARIES "${_JSONC_LIBRARIES}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET JsonC::JsonC)
    add_library(JsonC::JsonC ALIAS PkgConfig::_JSONC)
  endif()
else()
  find_path(
    JsonC_INCLUDE_DIR
    NAMES json-c/json.h
    HINTS ${JsonC_ROOT} ENV JsonC_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    JsonC_LIBRARY
    NAMES json-c
    HINTS ${JsonC_ROOT} ENV JsonC_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    JsonC REQUIRED_VARS JsonC_LIBRARY JsonC_INCLUDE_DIR
  )

  if(JsonC_FOUND)
    set(JsonC_INCLUDE_DIRS "${JsonC_INCLUDE_DIR}")
    set(JsonC_LIBRARIES "${JsonC_LIBRARY}")

    if(NOT TARGET JsonC::JsonC)
      add_library(JsonC::JsonC UNKNOWN IMPORTED)
      set_target_properties(
        JsonC::JsonC
        PROPERTIES IMPORTED_LOCATION "${JsonC_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${JsonC_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(JsonC_INCLUDE_DIR JsonC_LIBRARY)
