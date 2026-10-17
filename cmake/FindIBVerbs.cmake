# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindIBVerbs
-----------

Find RDMA verbs headers and the ibverbs runtime library.

Imported targets: ``IBVerbs::Headers``, ``IBVerbs::IBVerbs``.

Results: ``IBVerbs_FOUND``, ``IBVerbs_INCLUDE_DIRS``, ``IBVerbs_LIBRARIES``.

``IBVerbs::Headers`` and ``IBVerbs_Headers_FOUND`` are available with
headers alone, even when the runtime library is missing.

``IBVerbs_VERSION`` is available when selected-library metadata supplies it.
Version requests fail if a suitable version cannot be established.

Hints: ``IBVerbs_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``IBVerbs_INCLUDE_DIR``
* ``IBVerbs_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_IBVERBS QUIET IMPORTED_TARGET libibverbs)
endif()

# Let CMake honor package roots and explicit cache overrides before pkg-config
# hints. Only reuse pkg-config metadata when it describes the selected library.
find_path(
  IBVerbs_INCLUDE_DIR
  NAMES infiniband/verbs.h
  HINTS ${_IBVERBS_INCLUDE_DIRS}
)
find_library(
  IBVerbs_LIBRARY
  NAMES ibverbs
  HINTS ${_IBVERBS_LIBRARY_DIRS}
)
set(IBVerbs_INCLUDE_DIRS "${IBVerbs_INCLUDE_DIR}")
set(IBVerbs_LIBRARIES "${IBVerbs_LIBRARY}")
set(IBVerbs_VERSION "")
set(_IBVerbs_use_pkgconfig FALSE)
if(PkgConfig_FOUND
   AND _IBVERBS_FOUND
   AND IBVerbs_LIBRARY IN_LIST _IBVERBS_LINK_LIBRARIES
)
  set(_IBVerbs_use_pkgconfig TRUE)
  set(IBVerbs_VERSION "${_IBVERBS_VERSION}")
  set(IBVerbs_LIBRARIES "${_IBVERBS_LINK_LIBRARIES}")
endif()

# usNIC needs the ABI headers even when no usable runtime is installed.
set(IBVerbs_Headers_FOUND FALSE)
if(IBVerbs_INCLUDE_DIR)
  set(IBVerbs_Headers_FOUND TRUE)
  if(NOT TARGET IBVerbs::Headers)
    add_library(IBVerbs::Headers INTERFACE IMPORTED)
    set_target_properties(
      IBVerbs::Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                  "${IBVerbs_INCLUDE_DIR}"
    )
  endif()
endif()

find_package_handle_standard_args(
  IBVerbs
  REQUIRED_VARS IBVerbs_LIBRARY IBVerbs_INCLUDE_DIR
  VERSION_VAR IBVerbs_VERSION HANDLE_VERSION_RANGE
)
if(IBVerbs_FOUND AND NOT TARGET IBVerbs::IBVerbs)
  if(_IBVerbs_use_pkgconfig)
    add_library(IBVerbs::IBVerbs INTERFACE IMPORTED)
    target_link_libraries(IBVerbs::IBVerbs INTERFACE PkgConfig::_IBVERBS)
  else()
    add_library(IBVerbs::IBVerbs UNKNOWN IMPORTED)
    set_target_properties(
      IBVerbs::IBVerbs PROPERTIES IMPORTED_LOCATION "${IBVerbs_LIBRARY}"
    )
  endif()
  set_target_properties(
    IBVerbs::IBVerbs PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                "${IBVerbs_INCLUDE_DIR}"
  )
endif()
mark_as_advanced(IBVerbs_INCLUDE_DIR IBVerbs_LIBRARY)
