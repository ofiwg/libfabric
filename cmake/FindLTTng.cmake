# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_LTTNG QUIET IMPORTED_TARGET lttng-ust)
endif()

if(_LTTNG_FOUND)
  set(LTTng_FOUND TRUE)
  set(LTTng_INCLUDE_DIRS "${_LTTNG_INCLUDE_DIRS}")
  set(LTTng_LIBRARIES "${_LTTNG_LIBRARIES}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET LTTng::LTTng)
    add_library(LTTng::LTTng ALIAS PkgConfig::_LTTNG)
  endif()
else()
  find_path(
    LTTng_INCLUDE_DIR
    NAMES lttng/tracepoint.h
    HINTS ${LTTng_ROOT} ENV LTTng_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    LTTng_LIBRARY
    NAMES lttng-ust
    HINTS ${LTTng_ROOT} ENV LTTng_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    LTTng REQUIRED_VARS LTTng_LIBRARY LTTng_INCLUDE_DIR
  )

  if(LTTng_FOUND)
    set(LTTng_INCLUDE_DIRS "${LTTng_INCLUDE_DIR}")
    set(LTTng_LIBRARIES "${LTTng_LIBRARY}")

    if(NOT TARGET LTTng::LTTng)
      add_library(LTTng::LTTng UNKNOWN IMPORTED)
      set_target_properties(
        LTTng::LTTng
        PROPERTIES IMPORTED_LOCATION "${LTTng_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${LTTng_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(LTTng_INCLUDE_DIR LTTng_LIBRARY)
