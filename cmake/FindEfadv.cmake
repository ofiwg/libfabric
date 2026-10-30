# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindEfadv
---------

Find the EFA direct-verbs library and headers.

Imported targets: ``Efadv::Efadv``.

Results: ``Efadv_FOUND``, ``Efadv_INCLUDE_DIRS``, ``Efadv_LIBRARIES``.

``Efadv_VERSION`` is available when selected-library metadata supplies it.
Version requests fail if a suitable version cannot be established.

Hints: ``Efadv_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``Efadv_INCLUDE_DIR``
* ``Efadv_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_EFADV QUIET IMPORTED_TARGET libefa)
endif()

# Let CMake honor package roots and explicit cache overrides before pkg-config
# hints. Only reuse pkg-config metadata when it describes the selected library.
find_path(
  Efadv_INCLUDE_DIR
  NAMES infiniband/efadv.h
  HINTS ${_EFADV_INCLUDE_DIRS}
)
find_library(
  Efadv_LIBRARY
  NAMES efa
  HINTS ${_EFADV_LIBRARY_DIRS}
)
set(Efadv_INCLUDE_DIRS "${Efadv_INCLUDE_DIR}")
set(Efadv_LIBRARIES "${Efadv_LIBRARY}")
set(Efadv_VERSION "")
set(_Efadv_use_pkgconfig FALSE)
if(PkgConfig_FOUND
   AND _EFADV_FOUND
   AND Efadv_LIBRARY IN_LIST _EFADV_LINK_LIBRARIES
)
  set(_Efadv_use_pkgconfig TRUE)
  set(Efadv_VERSION "${_EFADV_VERSION}")
  set(Efadv_LIBRARIES "${_EFADV_LINK_LIBRARIES}")
endif()

find_package_handle_standard_args(
  Efadv
  REQUIRED_VARS Efadv_LIBRARY Efadv_INCLUDE_DIR
  VERSION_VAR Efadv_VERSION HANDLE_VERSION_RANGE
)
if(Efadv_FOUND AND NOT TARGET Efadv::Efadv)
  if(_Efadv_use_pkgconfig)
    add_library(Efadv::Efadv INTERFACE IMPORTED)
    target_link_libraries(Efadv::Efadv INTERFACE PkgConfig::_EFADV)
  else()
    add_library(Efadv::Efadv UNKNOWN IMPORTED)
    set_target_properties(
      Efadv::Efadv PROPERTIES IMPORTED_LOCATION "${Efadv_LIBRARY}"
    )
  endif()
  set_target_properties(
    Efadv::Efadv PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                            "${Efadv_INCLUDE_DIR}"
  )
endif()
mark_as_advanced(Efadv_INCLUDE_DIR Efadv_LIBRARY)
