# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckSymbolExists)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_RDMACM QUIET IMPORTED_TARGET librdmacm)
endif()

if(_RDMACM_FOUND)
  set(RdmaCM_FOUND TRUE)
  set(RdmaCM_VERSION "${_RDMACM_VERSION}")
  set(RdmaCM_INCLUDE_DIRS "${_RDMACM_INCLUDE_DIRS}")
  set(RdmaCM_LIBRARIES "${_RDMACM_LIBRARIES}")
  set(RdmaCM_LIBRARY_DIRS "${_RDMACM_LIBRARY_DIRS}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET RdmaCM::RdmaCM)
    add_library(RdmaCM::RdmaCM ALIAS PkgConfig::_RDMACM)
  endif()
else()
  find_path(
    RdmaCM_INCLUDE_DIR
    NAMES rdma/rdma_cma.h
    HINTS ${RdmaCM_ROOT} ENV RdmaCM_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    RdmaCM_LIBRARY
    NAMES rdmacm
    HINTS ${RdmaCM_ROOT} ENV RdmaCM_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    RdmaCM REQUIRED_VARS RdmaCM_LIBRARY RdmaCM_INCLUDE_DIR
  )

  if(RdmaCM_FOUND)
    set(RdmaCM_INCLUDE_DIRS "${RdmaCM_INCLUDE_DIR}")
    set(RdmaCM_LIBRARIES "${RdmaCM_LIBRARY}")

    if(NOT TARGET RdmaCM::RdmaCM)
      add_library(RdmaCM::RdmaCM UNKNOWN IMPORTED)
      set_target_properties(
        RdmaCM::RdmaCM
        PROPERTIES IMPORTED_LOCATION "${RdmaCM_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${RdmaCM_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

if(RdmaCM_FOUND)
  include(CMakePushCheckState)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${RdmaCM_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${RdmaCM_LIBRARIES})

  # Check for rdma_establish (modern rdma-core)
  check_symbol_exists(
    rdma_establish "rdma/rdma_cma.h" _RdmaCM_HAVE_RDMA_ESTABLISH
  )
  if(_RdmaCM_HAVE_RDMA_ESTABLISH)
    set(RdmaCM_HAVE_RDMA_ESTABLISH TRUE)
  else()
    set(RdmaCM_HAVE_RDMA_ESTABLISH FALSE)
  endif()

  # Check for rdma_create_qp_ex (required for XRC support)
  check_symbol_exists(
    rdma_create_qp_ex "rdma/rdma_cma.h" _RdmaCM_HAVE_CREATE_QP_EX
  )
  if(_RdmaCM_HAVE_CREATE_QP_EX)
    set(RdmaCM_HAVE_CREATE_QP_EX TRUE)
  else()
    set(RdmaCM_HAVE_CREATE_QP_EX FALSE)
  endif()

  cmake_pop_check_state()
endif()

mark_as_advanced(RdmaCM_INCLUDE_DIR RdmaCM_LIBRARY)
