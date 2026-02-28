# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)

set(LibNL_FOUND FALSE)
set(LibNL_IS_V3 FALSE)
set(LibNL_VERSION_MAJOR 0)

# Try libnl v3 first (preferred)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_LIBNL3 QUIET IMPORTED_TARGET libnl-3.0)
  pkg_check_modules(_LIBNL3_ROUTE QUIET IMPORTED_TARGET libnl-route-3.0)
endif()

if(_LIBNL3_FOUND)
  set(LibNL_FOUND TRUE)
  set(LibNL_IS_V3 TRUE)
  set(LibNL_VERSION_MAJOR 3)
  set(LibNL_INCLUDE_DIRS "${_LIBNL3_INCLUDE_DIRS}")
  set(LibNL_LIBRARIES "${_LIBNL3_LIBRARIES}")
  if(_LIBNL3_ROUTE_FOUND)
    list(APPEND LibNL_LIBRARIES "${_LIBNL3_ROUTE_LIBRARIES}")
  endif()

  # Create our target that links to both libnl and libnl-route
  if(NOT TARGET LibNL::LibNL)
    add_library(LibNL::LibNL INTERFACE IMPORTED)
    target_link_libraries(LibNL::LibNL INTERFACE PkgConfig::_LIBNL3)
    if(_LIBNL3_ROUTE_FOUND)
      target_link_libraries(LibNL::LibNL INTERFACE PkgConfig::_LIBNL3_ROUTE)
    endif()
  endif()
else()
  # Try manual search for libnl v3
  find_path(
    LibNL3_INCLUDE_DIR
    NAMES netlink/version.h
    HINTS ${LibNL_ROOT} ENV LibNL_ROOT
    PATH_SUFFIXES include/libnl3
  )

  find_library(
    LibNL3_LIBRARY
    NAMES nl-3
    HINTS ${LibNL_ROOT} ENV LibNL_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_library(
    LibNL3_ROUTE_LIBRARY
    NAMES nl-route-3
    HINTS ${LibNL_ROOT} ENV LibNL_ROOT
    PATH_SUFFIXES lib lib64
  )

  if(LibNL3_INCLUDE_DIR AND LibNL3_LIBRARY)
    # Verify this is really libnl v3 by checking LIBNL_VER_MAJ
    set(CMAKE_REQUIRED_INCLUDES ${LibNL3_INCLUDE_DIR})
    check_c_source_compiles(
      "
      #include <netlink/netlink.h>
      #include <netlink/version.h>
      #if LIBNL_VER_MAJ != 3
      #error \"Not libnl v3\"
      #endif
      int main() { return 0; }
    "
      _LIBNL3_VERSION_CHECK
    )

    if(_LIBNL3_VERSION_CHECK)
      set(LibNL_FOUND TRUE)
      set(LibNL_IS_V3 TRUE)
      set(LibNL_VERSION_MAJOR 3)
      set(LibNL_INCLUDE_DIRS "${LibNL3_INCLUDE_DIR}")
      set(LibNL_LIBRARIES "${LibNL3_LIBRARY}")
      if(LibNL3_ROUTE_LIBRARY)
        list(APPEND LibNL_LIBRARIES "${LibNL3_ROUTE_LIBRARY}")
      endif()

      if(NOT TARGET LibNL::LibNL)
        add_library(LibNL::LibNL INTERFACE IMPORTED)
        set_target_properties(
          LibNL::LibNL
          PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LibNL3_INCLUDE_DIR}"
                     INTERFACE_LINK_LIBRARIES "${LibNL_LIBRARIES}"
        )
      endif()
    endif()
  endif()
endif()

# If libnl v3 was not found, try libnl v1
if(NOT LibNL_FOUND)
  # Try pkg-config for libnl v1
  if(PkgConfig_FOUND)
    pkg_check_modules(_LIBNL1 QUIET IMPORTED_TARGET libnl-1)
  endif()

  if(_LIBNL1_FOUND)
    set(LibNL_FOUND TRUE)
    set(LibNL_IS_V3 FALSE)
    set(LibNL_VERSION_MAJOR 1)
    set(LibNL_INCLUDE_DIRS "${_LIBNL1_INCLUDE_DIRS}")
    set(LibNL_LIBRARIES "${_LIBNL1_LIBRARIES}")

    if(NOT TARGET LibNL::LibNL)
      add_library(LibNL::LibNL INTERFACE IMPORTED)
      target_link_libraries(LibNL::LibNL INTERFACE PkgConfig::_LIBNL1)
    endif()
  else()
    # Manual search for libnl v1
    find_path(
      LibNL1_INCLUDE_DIR
      NAMES netlink/netlink.h
      HINTS ${LibNL_ROOT} ENV LibNL_ROOT
      PATH_SUFFIXES include
    )

    find_library(
      LibNL1_LIBRARY
      NAMES nl
      HINTS ${LibNL_ROOT} ENV LibNL_ROOT
      PATH_SUFFIXES lib lib64
    )

    if(LibNL1_INCLUDE_DIR AND LibNL1_LIBRARY)
      # Make sure this is NOT libnl v3 (no version.h in the standard include
      # path)
      if(NOT EXISTS "${LibNL1_INCLUDE_DIR}/netlink/version.h")
        set(LibNL_FOUND TRUE)
        set(LibNL_IS_V3 FALSE)
        set(LibNL_VERSION_MAJOR 1)
        set(LibNL_INCLUDE_DIRS "${LibNL1_INCLUDE_DIR}")
        set(LibNL_LIBRARIES "${LibNL1_LIBRARY}" m)

        if(NOT TARGET LibNL::LibNL)
          add_library(LibNL::LibNL INTERFACE IMPORTED)
          set_target_properties(
            LibNL::LibNL
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LibNL1_INCLUDE_DIR}"
                       INTERFACE_LINK_LIBRARIES "${LibNL_LIBRARIES}"
          )
        endif()
      endif()
    endif()
  endif()
endif()

find_package_handle_standard_args(
  LibNL
  REQUIRED_VARS LibNL_FOUND LibNL_LIBRARIES
  VERSION_VAR LibNL_VERSION_MAJOR
)

if(LibNL_FOUND)
  message(STATUS "Found libnl version ${LibNL_VERSION_MAJOR}")
endif()

mark_as_advanced(
  LibNL3_INCLUDE_DIR LibNL3_LIBRARY LibNL3_ROUTE_LIBRARY LibNL1_INCLUDE_DIR
  LibNL1_LIBRARY
)
