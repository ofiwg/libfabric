# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindGDRCopy
-----------

Find NVIDIA GDRCopy headers and its optional runtime library.

Imported targets: ``GDRCopy::GDRCopy``.

Results: ``GDRCopy_FOUND``, ``GDRCopy_INCLUDE_DIRS``, ``GDRCopy_LIBRARIES``.

Components ``Headers`` and ``Runtime`` distinguish dlopen builds from
linked builds. The default is required ``Runtime`` (headers and library).
The imported runtime target is created only when that runtime is found.

Hints: ``GDRCopy_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``GDRCopy_INCLUDE_DIR``
* ``GDRCopy_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
# The Headers component supports builds that load the runtime with dlopen. With
# no components, require both headers and the linkable runtime.
include(FindPackageHandleStandardArgs)
if(NOT GDRCopy_FIND_COMPONENTS)
  set(GDRCopy_FIND_COMPONENTS Runtime)
  set(GDRCopy_FIND_REQUIRED_Runtime TRUE)
endif()

find_path(
  GDRCopy_INCLUDE_DIR
  NAMES gdrapi.h
  HINTS ${GDRCopy_ROOT} ENV GDRCopy_ROOT
  PATH_SUFFIXES include
)
set(GDRCopy_Headers_FOUND ${GDRCopy_INCLUDE_DIR})
set(GDRCopy_Runtime_FOUND FALSE)
if("Runtime" IN_LIST GDRCopy_FIND_COMPONENTS)
  find_library(
    GDRCopy_LIBRARY
    NAMES gdrapi
    HINTS ${GDRCopy_ROOT} ENV GDRCopy_ROOT
    PATH_SUFFIXES lib lib64
  )
  set(GDRCopy_Runtime_FOUND ${GDRCopy_LIBRARY})
endif()

find_package_handle_standard_args(
  GDRCopy
  REQUIRED_VARS GDRCopy_INCLUDE_DIR
  HANDLE_COMPONENTS
)
if(GDRCopy_FOUND)
  set(GDRCopy_INCLUDE_DIRS "${GDRCopy_INCLUDE_DIR}")
  if(GDRCopy_Runtime_FOUND)
    set(GDRCopy_LIBRARIES "${GDRCopy_LIBRARY}")
    if(NOT TARGET GDRCopy::GDRCopy)
      add_library(GDRCopy::GDRCopy UNKNOWN IMPORTED)
      set_target_properties(
        GDRCopy::GDRCopy
        PROPERTIES IMPORTED_LOCATION "${GDRCopy_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${GDRCopy_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()
mark_as_advanced(GDRCopy_INCLUDE_DIR GDRCopy_LIBRARY)
