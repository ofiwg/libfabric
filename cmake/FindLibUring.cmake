# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_LIBURING QUIET IMPORTED_TARGET liburing)
endif()

if(_LIBURING_FOUND)
  set(LibUring_VERSION "${_LIBURING_VERSION}")
  set(LibUring_INCLUDE_DIRS "${_LIBURING_INCLUDE_DIRS}")
  set(LibUring_LIBRARIES "${_LIBURING_LIBRARIES}")
  set(_LIBURING_PKGCONFIG TRUE)

  # Check for multishot support (requires liburing >= 2.1)
  set(CMAKE_REQUIRED_INCLUDES "${LibUring_INCLUDE_DIRS}")
  check_c_source_compiles(
    "
    #include <liburing.h>
    int main() {
      #ifndef io_uring_prep_poll_multishot
      #error no multishot
      #endif
      #ifndef IORING_CQE_F_MORE
      #error no IORING_CQE_F_MORE
      #endif
      return 0;
    }
  "
    LIBURING_HAS_MULTISHOT
  )

  if(LIBURING_HAS_MULTISHOT)
    set(LibUring_FOUND TRUE)
  else()
    set(LibUring_FOUND FALSE)
    message(
      STATUS "liburing found but multishot support missing (requires >= 2.1)"
    )
  endif()
else()
  find_path(
    LibUring_INCLUDE_DIR
    NAMES liburing.h
    HINTS ${LibUring_ROOT} ENV LibUring_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    LibUring_LIBRARY
    NAMES uring
    HINTS ${LibUring_ROOT} ENV LibUring_ROOT
    PATH_SUFFIXES lib lib64
  )

  if(LibUring_LIBRARY AND LibUring_INCLUDE_DIR)
    set(CMAKE_REQUIRED_INCLUDES "${LibUring_INCLUDE_DIR}")
    check_c_source_compiles(
      "
      #include <liburing.h>
      int main() {
        #ifndef io_uring_prep_poll_multishot
        #error no multishot
        #endif
        return 0;
      }
    "
      LIBURING_HAS_MULTISHOT
    )

    if(LIBURING_HAS_MULTISHOT)
      set(LibUring_FOUND TRUE)
      set(LibUring_INCLUDE_DIRS "${LibUring_INCLUDE_DIR}")
      set(LibUring_LIBRARIES "${LibUring_LIBRARY}")
    endif()
  endif()
endif()

find_package_handle_standard_args(LibUring REQUIRED_VARS LibUring_FOUND)

if(LibUring_FOUND AND NOT TARGET LibUring::LibUring)
  if(_LIBURING_PKGCONFIG)
    add_library(LibUring::LibUring ALIAS PkgConfig::_LIBURING)
  else()
    add_library(LibUring::LibUring INTERFACE IMPORTED)
    set_target_properties(
      LibUring::LibUring
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LibUring_INCLUDE_DIRS}"
                 INTERFACE_LINK_LIBRARIES "${LibUring_LIBRARIES}"
    )
  endif()
endif()

mark_as_advanced(LibUring_INCLUDE_DIR LibUring_LIBRARY)
