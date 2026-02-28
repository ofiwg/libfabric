# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckSymbolExists)

# Check if dlopen is available (either in libc or libdl)
check_symbol_exists(dlopen "dlfcn.h" _DL_HAVE_DLOPEN)

if(_DL_HAVE_DLOPEN)
  set(DL_FOUND TRUE)
  set(DL_LIBRARIES "${CMAKE_DL_LIBS}")
else()
  # Try linking with -ldl explicitly
  set(CMAKE_REQUIRED_LIBRARIES dl)
  check_symbol_exists(dlopen "dlfcn.h" _DL_HAVE_DLOPEN_WITH_LIB)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(_DL_HAVE_DLOPEN_WITH_LIB)
    set(DL_FOUND TRUE)
    set(DL_LIBRARIES "dl")
  else()
    set(DL_FOUND FALSE)
  endif()
endif()

find_package_handle_standard_args(DL REQUIRED_VARS DL_FOUND)

if(DL_FOUND AND NOT TARGET DL::DL)
  add_library(DL::DL INTERFACE IMPORTED)
  if(DL_LIBRARIES)
    set_target_properties(
      DL::DL PROPERTIES INTERFACE_LINK_LIBRARIES "${DL_LIBRARIES}"
    )
  endif()
endif()
