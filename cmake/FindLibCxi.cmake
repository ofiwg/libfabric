# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_path(
  LibCxi_INCLUDE_DIR
  NAMES libcxi/libcxi.h
  HINTS ${LibCxi_ROOT} ENV LibCxi_ROOT
  PATH_SUFFIXES include
)

find_library(
  LibCxi_LIBRARY
  NAMES cxi
  HINTS ${LibCxi_ROOT} ENV LibCxi_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  LibCxi REQUIRED_VARS LibCxi_LIBRARY LibCxi_INCLUDE_DIR
)

if(LibCxi_FOUND)
  set(LibCxi_INCLUDE_DIRS "${LibCxi_INCLUDE_DIR}")
  set(LibCxi_LIBRARIES "${LibCxi_LIBRARY}")

  if(NOT TARGET LibCxi::LibCxi)
    add_library(LibCxi::LibCxi UNKNOWN IMPORTED)
    set_target_properties(
      LibCxi::LibCxi
      PROPERTIES IMPORTED_LOCATION "${LibCxi_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${LibCxi_INCLUDE_DIR}"
    )
  endif()

  # Feature detection for libcxi functions
  include(CheckSymbolExists)
  include(CMakePushCheckState)

  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${LibCxi_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${LibCxi_LIBRARIES})

  # Check for cxil_modify_cp
  check_symbol_exists(cxil_modify_cp "libcxi/libcxi.h" LibCxi_HAVE_MODIFY_CP)

  # Check for cxil_svc_get_vni_range
  check_symbol_exists(
    cxil_svc_get_vni_range "libcxi/libcxi.h" LibCxi_HAVE_SVC_GET_VNI_RANGE
  )

  # Check for cxil_alloc_trig_cp
  check_symbol_exists(
    cxil_alloc_trig_cp "libcxi/libcxi.h" LibCxi_HAVE_ALLOC_TRIG_CP
  )

  cmake_pop_check_state()
endif()

mark_as_advanced(LibCxi_INCLUDE_DIR LibCxi_LIBRARY)
