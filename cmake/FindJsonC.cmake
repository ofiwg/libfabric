# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindJsonC
---------

Find json-c, preferring its upstream CMake config package.

Imported targets: ``JsonC::JsonC``.

Results: ``JsonC_FOUND``, ``JsonC_INCLUDE_DIRS``, ``JsonC_LIBRARIES``.

``JsonC_VERSION`` is available when selected-library metadata supplies it.
Version requests fail if a suitable version cannot be established.

Hints: ``JsonC_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``JsonC_INCLUDE_DIR``
* ``JsonC_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

# Prefer the upstream config package, forwarding version constraints.
set(_JsonC_version_args "${JsonC_FIND_VERSION}")
if(JsonC_FIND_VERSION_RANGE)
  set(_JsonC_version_args "${JsonC_FIND_VERSION_RANGE}")
elseif(JsonC_FIND_VERSION_EXACT)
  list(APPEND _JsonC_version_args EXACT)
endif()
find_package(json-c ${_JsonC_version_args} CONFIG QUIET)
if(json-c_FOUND AND TARGET json-c::json-c)
  set(JsonC_VERSION "${json-c_VERSION}")
  get_target_property(
    JsonC_INCLUDE_DIRS json-c::json-c INTERFACE_INCLUDE_DIRECTORIES
  )
  set(JsonC_LIBRARIES json-c::json-c)
  find_package_handle_standard_args(
    JsonC
    REQUIRED_VARS json-c_FOUND
    VERSION_VAR JsonC_VERSION HANDLE_VERSION_RANGE
  )
  if(JsonC_FOUND AND NOT TARGET JsonC::JsonC)
    add_library(JsonC::JsonC INTERFACE IMPORTED)
    target_link_libraries(JsonC::JsonC INTERFACE json-c::json-c)
  endif()
  return()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_JSONC QUIET IMPORTED_TARGET json-c)
endif()

# Let CMake honor package roots and explicit cache overrides before pkg-config
# hints. Only reuse pkg-config metadata when it describes the selected library.
find_path(
  JsonC_INCLUDE_DIR
  NAMES json-c/json.h
  HINTS ${_JSONC_INCLUDE_DIRS}
)
find_library(
  JsonC_LIBRARY
  NAMES json-c
  HINTS ${_JSONC_LIBRARY_DIRS}
)
set(JsonC_INCLUDE_DIRS "${JsonC_INCLUDE_DIR}")
set(JsonC_LIBRARIES "${JsonC_LIBRARY}")
set(JsonC_VERSION "")
set(_JsonC_use_pkgconfig FALSE)
if(PkgConfig_FOUND
   AND _JSONC_FOUND
   AND JsonC_LIBRARY IN_LIST _JSONC_LINK_LIBRARIES
)
  set(_JsonC_use_pkgconfig TRUE)
  set(JsonC_VERSION "${_JSONC_VERSION}")
  set(JsonC_LIBRARIES "${_JSONC_LINK_LIBRARIES}")
endif()

find_package_handle_standard_args(
  JsonC
  REQUIRED_VARS JsonC_LIBRARY JsonC_INCLUDE_DIR
  VERSION_VAR JsonC_VERSION HANDLE_VERSION_RANGE
)
if(JsonC_FOUND AND NOT TARGET JsonC::JsonC)
  if(_JsonC_use_pkgconfig)
    add_library(JsonC::JsonC INTERFACE IMPORTED)
    target_link_libraries(JsonC::JsonC INTERFACE PkgConfig::_JSONC)
  else()
    add_library(JsonC::JsonC UNKNOWN IMPORTED)
    set_target_properties(
      JsonC::JsonC PROPERTIES IMPORTED_LOCATION "${JsonC_LIBRARY}"
    )
  endif()
  set_target_properties(
    JsonC::JsonC PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                            "${JsonC_INCLUDE_DIR}"
  )
endif()
mark_as_advanced(JsonC_INCLUDE_DIR JsonC_LIBRARY)
