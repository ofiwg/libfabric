# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricVersion
----------------

Read release and ABI metadata from configure.ac and Makefile.am.

Include from the source root before project(). Sets LIBFABRIC_VERSION_NUMBER,
LIBFABRIC_VERSION_SUFFIX, LIBFABRIC_VERSION_FULL, LIBFABRIC_SOVERSION, and
LIBFABRIC_ABI_VERSION; registers both inputs for automatic reconfiguration.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()
# Keep the release version and ABI version aligned with the existing build.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/configure.ac" _version_line
     REGEX "^AC_INIT"
)
string(REGEX MATCH "\\[([0-9]+\\.[0-9]+\\.[0-9]+)([^]]*)\\]" _match
             "${_version_line}"
)
if(NOT _match)
  message(FATAL_ERROR "Cannot read the release version from configure.ac")
endif()
set(LIBFABRIC_VERSION_NUMBER "${CMAKE_MATCH_1}")
set(LIBFABRIC_VERSION_SUFFIX "${CMAKE_MATCH_2}")
set(LIBFABRIC_VERSION_FULL
    "${LIBFABRIC_VERSION_NUMBER}${LIBFABRIC_VERSION_SUFFIX}"
)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/Makefile.am" _abi_line
     REGEX "version-info [0-9]+:[0-9]+:[0-9]+"
)
string(REGEX MATCH "version-info ([0-9]+):([0-9]+):([0-9]+)" _match
             "${_abi_line}"
)
if(NOT _match)
  message(FATAL_ERROR "Cannot read the library ABI version from Makefile.am")
endif()
math(EXPR LIBFABRIC_SOVERSION "${CMAKE_MATCH_1} - ${CMAKE_MATCH_3}")
set(LIBFABRIC_ABI_VERSION
    "${LIBFABRIC_SOVERSION}.${CMAKE_MATCH_3}.${CMAKE_MATCH_2}"
)

set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS configure.ac Makefile.am
)
