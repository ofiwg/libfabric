# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_path(
  XPMEM_INCLUDE_DIR
  NAMES xpmem.h
  HINTS ${XPMEM_ROOT} ENV XPMEM_ROOT
  PATH_SUFFIXES include
)

find_library(
  XPMEM_LIBRARY
  NAMES xpmem
  HINTS ${XPMEM_ROOT} ENV XPMEM_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  XPMEM REQUIRED_VARS XPMEM_LIBRARY XPMEM_INCLUDE_DIR
)

if(XPMEM_FOUND)
  set(XPMEM_INCLUDE_DIRS "${XPMEM_INCLUDE_DIR}")
  set(XPMEM_LIBRARIES "${XPMEM_LIBRARY}")

  if(NOT TARGET XPMEM::XPMEM)
    add_library(XPMEM::XPMEM UNKNOWN IMPORTED)
    set_target_properties(
      XPMEM::XPMEM
      PROPERTIES IMPORTED_LOCATION "${XPMEM_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${XPMEM_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(XPMEM_INCLUDE_DIR XPMEM_LIBRARY)
