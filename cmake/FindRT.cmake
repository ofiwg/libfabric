# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckSymbolExists)

# Check if clock_gettime is already available (in libc on modern systems)
check_symbol_exists(clock_gettime "time.h" _RT_HAVE_CLOCK_GETTIME)

if(_RT_HAVE_CLOCK_GETTIME)
  set(RT_FOUND TRUE)
  set(RT_LIBRARIES "")
else()
  # Try with -lrt
  set(CMAKE_REQUIRED_LIBRARIES rt)
  check_symbol_exists(clock_gettime "time.h" _RT_HAVE_CLOCK_GETTIME_WITH_LIB)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(_RT_HAVE_CLOCK_GETTIME_WITH_LIB)
    set(RT_FOUND TRUE)
    set(RT_LIBRARIES "rt")
  else()
    set(RT_FOUND FALSE)
  endif()
endif()

find_package_handle_standard_args(RT REQUIRED_VARS RT_FOUND)

if(RT_FOUND AND NOT TARGET RT::RT)
  add_library(RT::RT INTERFACE IMPORTED)
  if(RT_LIBRARIES)
    set_target_properties(
      RT::RT PROPERTIES INTERFACE_LINK_LIBRARIES "${RT_LIBRARIES}"
    )
  endif()
endif()
