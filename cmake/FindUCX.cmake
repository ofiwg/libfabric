# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindUCX
-------

Find UCP and the UCS, UCT, and UCM libraries required by the UCX provider.

Imported targets: ``UCX::UCX``.

Results: ``UCX_FOUND``, ``UCX_INCLUDE_DIRS``, ``UCX_LIBRARIES``.

Provider-specific API compatibility is checked by ``CheckUCX.cmake``
when building libfabric, rather than by installed consumers.

Hints: ``UCX_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``UCX_INCLUDE_DIR``
* ``UCX_LIBRARY``
* ``UCX_ucs_LIBRARY``
* ``UCX_uct_LIBRARY``
* ``UCX_ucm_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_path(
  UCX_INCLUDE_DIR
  NAMES ucp/api/ucp.h
  HINTS ${UCX_ROOT} ENV UCX_ROOT
  PATH_SUFFIXES include
)

find_library(
  UCX_LIBRARY
  NAMES ucp
  HINTS ${UCX_ROOT} ENV UCX_ROOT
  PATH_SUFFIXES lib lib64
)

foreach(_component ucs uct ucm)
  find_library(
    UCX_${_component}_LIBRARY
    NAMES ${_component}
    HINTS ${UCX_ROOT} ENV UCX_ROOT
    PATH_SUFFIXES lib lib64
  )
  mark_as_advanced(UCX_${_component}_LIBRARY)
endforeach()
find_package_handle_standard_args(
  UCX
  REQUIRED_VARS
    UCX_LIBRARY
    UCX_INCLUDE_DIR
    UCX_ucs_LIBRARY
    UCX_uct_LIBRARY
    UCX_ucm_LIBRARY
)

if(UCX_FOUND)
  set(UCX_INCLUDE_DIRS "${UCX_INCLUDE_DIR}")
  set(UCX_LIBRARIES
      "${UCX_LIBRARY};${UCX_ucs_LIBRARY};${UCX_uct_LIBRARY};${UCX_ucm_LIBRARY}"
  )

  if(NOT TARGET UCX::UCX)
    add_library(UCX::UCX UNKNOWN IMPORTED)
    set_target_properties(
      UCX::UCX
      PROPERTIES
        IMPORTED_LOCATION "${UCX_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${UCX_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES
        "${UCX_ucs_LIBRARY};${UCX_uct_LIBRARY};${UCX_ucm_LIBRARY}"
    )
  endif()

endif()

mark_as_advanced(UCX_INCLUDE_DIR UCX_LIBRARY)
