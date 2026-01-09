# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_CURL QUIET IMPORTED_TARGET libcurl)
endif()

if(_CURL_FOUND)
  set(LibCurl_FOUND TRUE)
  set(LibCurl_INCLUDE_DIRS "${_CURL_INCLUDE_DIRS}")
  set(LibCurl_LIBRARIES "${_CURL_LIBRARIES}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET LibCurl::LibCurl)
    add_library(LibCurl::LibCurl ALIAS PkgConfig::_CURL)
  endif()
else()
  find_path(
    LibCurl_INCLUDE_DIR
    NAMES curl/curl.h
    HINTS ${LibCurl_ROOT} ENV LibCurl_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    LibCurl_LIBRARY
    NAMES curl
    HINTS ${LibCurl_ROOT} ENV LibCurl_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    LibCurl REQUIRED_VARS LibCurl_LIBRARY LibCurl_INCLUDE_DIR
  )

  if(LibCurl_FOUND)
    set(LibCurl_INCLUDE_DIRS "${LibCurl_INCLUDE_DIR}")
    set(LibCurl_LIBRARIES "${LibCurl_LIBRARY}")

    if(NOT TARGET LibCurl::LibCurl)
      add_library(LibCurl::LibCurl UNKNOWN IMPORTED)
      set_target_properties(
        LibCurl::LibCurl
        PROPERTIES IMPORTED_LOCATION "${LibCurl_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${LibCurl_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(LibCurl_INCLUDE_DIR LibCurl_LIBRARY)
