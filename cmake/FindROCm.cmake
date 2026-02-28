# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckSymbolExists)

find_path(
  ROCm_INCLUDE_DIR
  NAMES hsa/hsa_ext_amd.h
  HINTS ${ROCm_ROOT} ENV ROCm_ROOT /opt/rocm /opt/rocm/hsa
  PATH_SUFFIXES include
)

find_library(
  ROCm_LIBRARY
  NAMES hsa-runtime64
  HINTS ${ROCm_ROOT} ENV ROCm_ROOT /opt/rocm /opt/rocm/hsa
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  ROCm REQUIRED_VARS ROCm_LIBRARY ROCm_INCLUDE_DIR
)

# Initialize dmabuf detection result
set(ROCm_HAS_DMABUF FALSE)

if(ROCm_FOUND)
  set(ROCm_INCLUDE_DIRS "${ROCm_INCLUDE_DIR}")
  set(ROCm_LIBRARIES "${ROCm_LIBRARY}")

  if(NOT TARGET ROCm::ROCm)
    add_library(ROCm::ROCm UNKNOWN IMPORTED)
    set_target_properties(
      ROCm::ROCm PROPERTIES IMPORTED_LOCATION "${ROCm_LIBRARY}"
                            INTERFACE_INCLUDE_DIRECTORIES "${ROCm_INCLUDE_DIR}"
    )
  endif()

  # Check for HSA dmabuf export support
  set(CMAKE_REQUIRED_INCLUDES ${ROCm_INCLUDE_DIR})
  check_symbol_exists(
    hsa_amd_portable_export_dmabuf "hsa/hsa_ext_amd.h" _ROCm_HAVE_DMABUF_EXPORT
  )
  check_symbol_exists(
    HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED "hsa/hsa.h" _ROCm_HAVE_DMABUF_INFO
  )
  if(_ROCm_HAVE_DMABUF_EXPORT AND _ROCm_HAVE_DMABUF_INFO)
    set(ROCm_HAS_DMABUF TRUE)
  endif()
  unset(CMAKE_REQUIRED_INCLUDES)
endif()

mark_as_advanced(ROCm_INCLUDE_DIR ROCm_LIBRARY)
