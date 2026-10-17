# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricHmemDependencies
-------------------------

Discover SDKs for explicitly enabled accelerator backends.

Called by LibfabricDependencies before provider checks. Runtime-loading
backends require headers only; linked backends require runtime targets.
Sets package discovery results consumed by LibfabricFeatureDetection and
LibfabricHmem. CUDA linking uses CMake's upstream FindCUDAToolkit.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()
foreach(
  _backend
  CUDAToolkit
  GDRCopy
  LevelZero
  ROCm
  Neuron
  SynapseAI
)
  set(${_backend}_FOUND FALSE)
endforeach()
if(LIBFABRIC_ENABLE_GDRCOPY AND NOT LIBFABRIC_ENABLE_CUDA)
  message(FATAL_ERROR "GDRCopy requires LIBFABRIC_ENABLE_CUDA=ON")
endif()
# GPU/Accelerator support
if(LIBFABRIC_ENABLE_CUDA)
  if(LIBFABRIC_CUDA_DLOPEN)
    # No CUDA compiler or runtime library is required for runtime loading.
    find_path(
      LIBFABRIC_CUDA_INCLUDE_DIR cuda.h
      HINTS ${CUDAToolkit_ROOT}
            ENV
            CUDAToolkit_ROOT
            ENV
            CUDA_PATH
      PATH_SUFFIXES include REQUIRED
    )
    find_path(
      LIBFABRIC_CUDART_INCLUDE_DIR cuda_runtime.h
      HINTS ${LIBFABRIC_CUDA_INCLUDE_DIR}
            ${CUDAToolkit_ROOT}
            ENV
            CUDAToolkit_ROOT
            ENV
            CUDA_PATH
      PATH_SUFFIXES include REQUIRED
    )
    find_path(
      LIBFABRIC_NVML_INCLUDE_DIR nvml.h
      HINTS ${LIBFABRIC_CUDA_INCLUDE_DIR}
            ${CUDAToolkit_ROOT}
            ENV
            CUDAToolkit_ROOT
            ENV
            CUDA_PATH
      PATH_SUFFIXES include REQUIRED
    )
    mark_as_advanced(
      LIBFABRIC_CUDA_INCLUDE_DIR LIBFABRIC_CUDART_INCLUDE_DIR
      LIBFABRIC_NVML_INCLUDE_DIR
    )
    set(CUDAToolkit_INCLUDE_DIRS
        ${LIBFABRIC_CUDA_INCLUDE_DIR} ${LIBFABRIC_CUDART_INCLUDE_DIR}
        ${LIBFABRIC_NVML_INCLUDE_DIR}
    )
    set(CUDAToolkit_FOUND TRUE)
  else()
    find_package(CUDAToolkit REQUIRED)
    if(NOT TARGET CUDA::nvml)
      message(FATAL_ERROR "CUDA support requires the NVML development library")
    endif()
  endif()
  if(LIBFABRIC_ENABLE_GDRCOPY AND CUDAToolkit_FOUND)
    if(LIBFABRIC_GDRCOPY_DLOPEN)
      find_package(GDRCopy REQUIRED COMPONENTS Headers)
    else()
      find_package(GDRCopy REQUIRED)
    endif()
  endif()
endif()

if(LIBFABRIC_ENABLE_ZE)
  if(LIBFABRIC_ZE_DLOPEN)
    find_package(LevelZero REQUIRED COMPONENTS Headers)
  else()
    find_package(LevelZero REQUIRED)
  endif()
endif()

if(LIBFABRIC_ENABLE_ROCR)
  if(LIBFABRIC_ROCR_DLOPEN)
    find_package(ROCm REQUIRED COMPONENTS Headers)
  else()
    find_package(ROCm REQUIRED)
  endif()
  include(CheckROCm)
endif()

if(LIBFABRIC_ENABLE_NEURON)
  find_package(Neuron REQUIRED)
endif()

if(LIBFABRIC_ENABLE_SYNAPSEAI)
  find_package(SynapseAI REQUIRED)
endif()
