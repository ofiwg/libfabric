# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindLevelZero
-------------

Find Intel Level Zero headers and its optional runtime loader.

Imported targets: ``LevelZero::LevelZero``.

Results: ``LevelZero_FOUND``, ``LevelZero_INCLUDE_DIRS``,
``LevelZero_LIBRARIES``.

Components ``Headers`` and ``Runtime`` distinguish dlopen builds from
linked builds. The default is required ``Runtime`` (headers and library).
The imported runtime target is created only when that runtime is found.

``LevelZero_HAS_DRM``, ``LevelZero_HAS_LIBDRM``, and
``LevelZero_DRM_INCLUDE_DIR`` describe optional i915 headers used by PSM3.

Hints: ``LevelZero_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``LevelZero_INCLUDE_DIR``
* ``LevelZero_LIBRARY``
* ``LevelZero_DRM_INCLUDE_DIR``

Unguarded: each find_package call must reevaluate its request and scope.
]]
# The Headers component supports builds that load the runtime with dlopen. With
# no components, require both headers and the linkable runtime.
include(FindPackageHandleStandardArgs)
if(NOT LevelZero_FIND_COMPONENTS)
  set(LevelZero_FIND_COMPONENTS Runtime)
  set(LevelZero_FIND_REQUIRED_Runtime TRUE)
endif()

find_path(
  LevelZero_INCLUDE_DIR
  NAMES level_zero/ze_api.h
  HINTS ${LevelZero_ROOT} ENV LevelZero_ROOT
  PATH_SUFFIXES include
)
set(LevelZero_Headers_FOUND ${LevelZero_INCLUDE_DIR})
set(LevelZero_Runtime_FOUND FALSE)
if("Runtime" IN_LIST LevelZero_FIND_COMPONENTS)
  find_library(
    LevelZero_LIBRARY
    NAMES ze_loader
    HINTS ${LevelZero_ROOT} ENV LevelZero_ROOT
    PATH_SUFFIXES lib lib64
  )
  set(LevelZero_Runtime_FOUND ${LevelZero_LIBRARY})
endif()

find_package_handle_standard_args(
  LevelZero
  REQUIRED_VARS LevelZero_INCLUDE_DIR
  HANDLE_COMPONENTS
)
if(LevelZero_FOUND)
  set(LevelZero_INCLUDE_DIRS "${LevelZero_INCLUDE_DIR}")
  if(LevelZero_Runtime_FOUND)
    set(LevelZero_LIBRARIES "${LevelZero_LIBRARY}")
    if(NOT TARGET LevelZero::LevelZero)
      add_library(LevelZero::LevelZero UNKNOWN IMPORTED)
      set_target_properties(
        LevelZero::LevelZero
        PROPERTIES IMPORTED_LOCATION "${LevelZero_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()
mark_as_advanced(LevelZero_INCLUDE_DIR LevelZero_LIBRARY)

set(LevelZero_HAS_DRM FALSE)
set(LevelZero_HAS_LIBDRM FALSE)
if(LevelZero_FOUND)
  find_path(LevelZero_DRM_INCLUDE_DIR NAMES drm/i915_drm.h libdrm/i915_drm.h)
  if(LevelZero_DRM_INCLUDE_DIR)
    list(APPEND LevelZero_INCLUDE_DIRS "${LevelZero_DRM_INCLUDE_DIR}")
    if(EXISTS "${LevelZero_DRM_INCLUDE_DIR}/drm/i915_drm.h")
      set(LevelZero_HAS_DRM TRUE)
    else()
      set(LevelZero_HAS_LIBDRM TRUE)
    endif()
  endif()
endif()
mark_as_advanced(LevelZero_DRM_INCLUDE_DIR)
