# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_path(
  Numa_INCLUDE_DIR
  NAMES numa.h
  HINTS ${Numa_ROOT} ENV Numa_ROOT
  PATH_SUFFIXES include
)

find_library(
  Numa_LIBRARY
  NAMES numa
  HINTS ${Numa_ROOT} ENV Numa_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  Numa REQUIRED_VARS Numa_LIBRARY Numa_INCLUDE_DIR
)

if(Numa_FOUND)
  set(Numa_INCLUDE_DIRS "${Numa_INCLUDE_DIR}")
  set(Numa_LIBRARIES "${Numa_LIBRARY}")

  if(NOT TARGET Numa::Numa)
    add_library(Numa::Numa UNKNOWN IMPORTED)
    set_target_properties(
      Numa::Numa PROPERTIES IMPORTED_LOCATION "${Numa_LIBRARY}"
                            INTERFACE_INCLUDE_DIRECTORIES "${Numa_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(Numa_INCLUDE_DIR Numa_LIBRARY)
