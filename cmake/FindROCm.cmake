# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindROCm
--------

Find ROCr headers and the optional HSA runtime, preferring its config package.

Imported targets: ``ROCm::ROCm``.

Results: ``ROCm_FOUND``, ``ROCm_INCLUDE_DIRS``, ``ROCm_LIBRARIES``.

Components ``Headers`` and ``Runtime`` distinguish dlopen builds from
linked builds. The default is required ``Runtime`` (headers and library).
The imported runtime target is created only when that runtime is found.

Hints: ``ROCm_ROOT`` or ``CMAKE_PREFIX_PATH``.
Advanced cache overrides:

* ``ROCm_INCLUDE_DIR``
* ``ROCm_LIBRARY``

Unguarded: each find_package call must reevaluate its request and scope.
]]
# Prefer the upstream runtime package when linking; dlopen needs only headers.
include(FindPackageHandleStandardArgs)
if(NOT ROCm_FIND_COMPONENTS)
  set(ROCm_FIND_COMPONENTS Runtime)
  set(ROCm_FIND_REQUIRED_Runtime TRUE)
endif()
set(ROCm_Runtime_FOUND FALSE)
if("Runtime" IN_LIST ROCm_FIND_COMPONENTS)
  find_package(
    hsa-runtime64
    CONFIG
    QUIET
    HINTS
    ${ROCm_ROOT}
    ENV
    ROCm_ROOT
    /opt/rocm
  )
endif()
if("Runtime" IN_LIST ROCm_FIND_COMPONENTS AND TARGET
                                              hsa-runtime64::hsa-runtime64
)
  get_target_property(
    ROCm_INCLUDE_DIRS hsa-runtime64::hsa-runtime64
    INTERFACE_INCLUDE_DIRECTORIES
  )
  set(ROCm_Runtime_FOUND TRUE)
  if(NOT TARGET ROCm::ROCm)
    add_library(ROCm::ROCm INTERFACE IMPORTED)
    target_link_libraries(ROCm::ROCm INTERFACE hsa-runtime64::hsa-runtime64)
  endif()
else()
  find_path(
    ROCm_INCLUDE_DIR
    NAMES hsa/hsa_ext_amd.h
    HINTS ${ROCm_ROOT}
          ENV
          ROCm_ROOT
          /opt/rocm
          /opt/rocm/hsa
    PATH_SUFFIXES include
  )
  set(ROCm_INCLUDE_DIRS "${ROCm_INCLUDE_DIR}")
  if("Runtime" IN_LIST ROCm_FIND_COMPONENTS)
    find_library(
      ROCm_LIBRARY
      NAMES hsa-runtime64
      HINTS ${ROCm_ROOT}
            ENV
            ROCm_ROOT
            /opt/rocm
            /opt/rocm/hsa
      PATH_SUFFIXES lib lib64
    )
    set(ROCm_Runtime_FOUND ${ROCm_LIBRARY})
    if(ROCm_LIBRARY
       AND ROCm_INCLUDE_DIR
       AND NOT TARGET ROCm::ROCm
    )
      add_library(ROCm::ROCm UNKNOWN IMPORTED)
      set_target_properties(
        ROCm::ROCm
        PROPERTIES IMPORTED_LOCATION "${ROCm_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${ROCm_INCLUDE_DIR}"
      )
    endif()
    set(ROCm_LIBRARIES "${ROCm_LIBRARY}")
  endif()
endif()
set(ROCm_Headers_FOUND ${ROCm_INCLUDE_DIRS})
find_package_handle_standard_args(
  ROCm
  REQUIRED_VARS ROCm_INCLUDE_DIRS
  HANDLE_COMPONENTS
)
mark_as_advanced(ROCm_INCLUDE_DIR ROCm_LIBRARY)
