# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckIncludeFile)

find_path(
  LevelZero_INCLUDE_DIR
  NAMES level_zero/ze_api.h
  HINTS ${LevelZero_ROOT} ENV LevelZero_ROOT
  PATH_SUFFIXES include
)

find_library(
  LevelZero_LIBRARY
  NAMES ze_loader
  HINTS ${LevelZero_ROOT} ENV LevelZero_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  LevelZero REQUIRED_VARS LevelZero_LIBRARY LevelZero_INCLUDE_DIR
)

# Initialize DRM detection results
set(LevelZero_HAS_DRM FALSE)
set(LevelZero_HAS_LIBDRM FALSE)

if(LevelZero_FOUND)
  set(LevelZero_INCLUDE_DIRS "${LevelZero_INCLUDE_DIR}")
  set(LevelZero_LIBRARIES "${LevelZero_LIBRARY}")

  if(NOT TARGET LevelZero::LevelZero)
    add_library(LevelZero::LevelZero UNKNOWN IMPORTED)
    set_target_properties(
      LevelZero::LevelZero
      PROPERTIES IMPORTED_LOCATION "${LevelZero_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIR}"
    )
  endif()

  # Check for DRM headers (needed for Level Zero dmabuf support)
  check_include_file("drm/i915_drm.h" _LevelZero_HAVE_DRM_I915)
  if(_LevelZero_HAVE_DRM_I915)
    set(LevelZero_HAS_DRM TRUE)
  else()
    check_include_file("libdrm/i915_drm.h" _LevelZero_HAVE_LIBDRM_I915)
    if(_LevelZero_HAVE_LIBDRM_I915)
      set(LevelZero_HAS_LIBDRM TRUE)
    endif()
  endif()
endif()

mark_as_advanced(LevelZero_INCLUDE_DIR LevelZero_LIBRARY)
