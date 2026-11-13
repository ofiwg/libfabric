# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindRdmaCM
----------

Find the RDMA connection-manager library and headers.

Imported targets: ``RdmaCM::RdmaCM``.

Results: ``RdmaCM_FOUND``, ``RdmaCM_INCLUDE_DIRS``, ``RdmaCM_LIBRARIES``.

``RdmaCM_VERSION`` is available when selected-library metadata supplies it.
Version requests fail if a suitable version cannot be established.

Hints: ``RdmaCM_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``RdmaCM_INCLUDE_DIR``
* ``RdmaCM_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_RDMACM QUIET IMPORTED_TARGET librdmacm)
endif()

# Let CMake honor package roots and explicit cache overrides before pkg-config
# hints. Only reuse pkg-config metadata when it describes the selected library.
find_path(
  RdmaCM_INCLUDE_DIR
  NAMES rdma/rdma_cma.h
  HINTS ${_RDMACM_INCLUDE_DIRS}
)
find_library(
  RdmaCM_LIBRARY
  NAMES rdmacm
  HINTS ${_RDMACM_LIBRARY_DIRS}
)
set(RdmaCM_INCLUDE_DIRS "${RdmaCM_INCLUDE_DIR}")
set(RdmaCM_LIBRARIES "${RdmaCM_LIBRARY}")
set(RdmaCM_VERSION "")
set(_RdmaCM_use_pkgconfig FALSE)
if(PkgConfig_FOUND
   AND _RDMACM_FOUND
   AND RdmaCM_LIBRARY IN_LIST _RDMACM_LINK_LIBRARIES
)
  set(_RdmaCM_use_pkgconfig TRUE)
  set(RdmaCM_VERSION "${_RDMACM_VERSION}")
  set(RdmaCM_LIBRARIES "${_RDMACM_LINK_LIBRARIES}")
endif()

find_package_handle_standard_args(
  RdmaCM
  REQUIRED_VARS RdmaCM_LIBRARY RdmaCM_INCLUDE_DIR
  VERSION_VAR RdmaCM_VERSION HANDLE_VERSION_RANGE
)
if(RdmaCM_FOUND AND NOT TARGET RdmaCM::RdmaCM)
  if(_RdmaCM_use_pkgconfig)
    add_library(RdmaCM::RdmaCM INTERFACE IMPORTED)
    target_link_libraries(RdmaCM::RdmaCM INTERFACE PkgConfig::_RDMACM)
  else()
    add_library(RdmaCM::RdmaCM UNKNOWN IMPORTED)
    set_target_properties(
      RdmaCM::RdmaCM PROPERTIES IMPORTED_LOCATION "${RdmaCM_LIBRARY}"
    )
  endif()
  set_target_properties(
    RdmaCM::RdmaCM PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                              "${RdmaCM_INCLUDE_DIR}"
  )
endif()
mark_as_advanced(RdmaCM_INCLUDE_DIR RdmaCM_LIBRARY)
