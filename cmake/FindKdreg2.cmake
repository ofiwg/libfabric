# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckIncludeFile)

set(Kdreg2_FOUND FALSE)
set(Kdreg2_IN_LINUX FALSE)

# First check linux/kdreg2.h (kernel header)
check_include_file("linux/kdreg2.h" HAVE_LINUX_KDREG2_H)
if(HAVE_LINUX_KDREG2_H)
  set(Kdreg2_FOUND TRUE)
  set(Kdreg2_IN_LINUX TRUE)
  set(Kdreg2_INCLUDE_DIRS "")
else()
  # Check for custom path
  find_path(
    Kdreg2_INCLUDE_DIR
    NAMES kdreg2.h
    HINTS ${Kdreg2_ROOT} ENV Kdreg2_ROOT
    PATH_SUFFIXES include
  )

  if(Kdreg2_INCLUDE_DIR)
    set(Kdreg2_FOUND TRUE)
    set(Kdreg2_IN_LINUX FALSE)
    set(Kdreg2_INCLUDE_DIRS "${Kdreg2_INCLUDE_DIR}")
  endif()
endif()

find_package_handle_standard_args(Kdreg2 REQUIRED_VARS Kdreg2_FOUND)

if(Kdreg2_FOUND AND NOT TARGET Kdreg2::Kdreg2)
  add_library(Kdreg2::Kdreg2 INTERFACE IMPORTED)
  if(Kdreg2_INCLUDE_DIRS)
    set_target_properties(
      Kdreg2::Kdreg2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                "${Kdreg2_INCLUDE_DIRS}"
    )
  endif()
endif()

mark_as_advanced(Kdreg2_INCLUDE_DIR)
