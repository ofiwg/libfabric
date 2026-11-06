# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindLTTng
---------

Find the LTTng userspace tracing library and headers.

Imported targets: ``LTTng::LTTng``.

Results: ``LTTng_FOUND``, ``LTTng_INCLUDE_DIRS``, ``LTTng_LIBRARIES``.

``LTTng_VERSION`` is available when selected-library metadata supplies it.
Version requests fail if a suitable version cannot be established.

Hints: ``LTTng_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``LTTng_INCLUDE_DIR``
* ``LTTng_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_LTTNG QUIET IMPORTED_TARGET lttng-ust)
endif()

# Let CMake honor package roots and explicit cache overrides before pkg-config
# hints. Only reuse pkg-config metadata when it describes the selected library.
find_path(
  LTTng_INCLUDE_DIR
  NAMES lttng/tracepoint.h
  HINTS ${_LTTNG_INCLUDE_DIRS}
)
find_library(
  LTTng_LIBRARY
  NAMES lttng-ust
  HINTS ${_LTTNG_LIBRARY_DIRS}
)
set(LTTng_INCLUDE_DIRS "${LTTng_INCLUDE_DIR}")
set(LTTng_LIBRARIES "${LTTng_LIBRARY}")
set(LTTng_VERSION "")
set(_LTTng_use_pkgconfig FALSE)
if(PkgConfig_FOUND
   AND _LTTNG_FOUND
   AND LTTng_LIBRARY IN_LIST _LTTNG_LINK_LIBRARIES
)
  set(_LTTng_use_pkgconfig TRUE)
  set(LTTng_VERSION "${_LTTNG_VERSION}")
  set(LTTng_LIBRARIES "${_LTTNG_LINK_LIBRARIES}")
endif()

find_package_handle_standard_args(
  LTTng
  REQUIRED_VARS LTTng_LIBRARY LTTng_INCLUDE_DIR
  VERSION_VAR LTTng_VERSION HANDLE_VERSION_RANGE
)
if(LTTng_FOUND AND NOT TARGET LTTng::LTTng)
  if(_LTTng_use_pkgconfig)
    add_library(LTTng::LTTng INTERFACE IMPORTED)
    target_link_libraries(LTTng::LTTng INTERFACE PkgConfig::_LTTNG)
  else()
    add_library(LTTng::LTTng UNKNOWN IMPORTED)
    set_target_properties(
      LTTng::LTTng PROPERTIES IMPORTED_LOCATION "${LTTng_LIBRARY}"
    )
  endif()
  set_target_properties(
    LTTng::LTTng PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                            "${LTTng_INCLUDE_DIR}"
  )
endif()
mark_as_advanced(LTTng_INCLUDE_DIR LTTng_LIBRARY)
