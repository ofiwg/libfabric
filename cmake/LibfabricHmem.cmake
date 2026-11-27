# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricHmem
-------------

Attach accelerator headers and select directly linked HMEM runtimes.

Requires libfabric_build_settings and completed feature detection. Adds
private SDK includes to that interface target and sets LIBFABRIC_HMEM_LIBRARIES.
Runtime-loaded backends do not add link-time runtime dependencies.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()

# Header requirements apply to the core and all providers. Runtime libraries are
# linked only for backends configured without dlopen.
set(LIBFABRIC_HMEM_LIBRARIES "")
foreach(
  _backend
  CUDAToolkit
  GDRCopy
  LevelZero
  ROCm
  Neuron
  SynapseAI
)
  if(${_backend}_FOUND)
    target_include_directories(
      libfabric_build_settings SYSTEM INTERFACE ${${_backend}_INCLUDE_DIRS}
    )
  endif()
endforeach()
if(HAVE_CUDA AND NOT LIBFABRIC_CUDA_DLOPEN)
  list(
    APPEND
    LIBFABRIC_HMEM_LIBRARIES
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvml
  )
endif()
if(HAVE_GDRCOPY AND NOT LIBFABRIC_GDRCOPY_DLOPEN)
  list(APPEND LIBFABRIC_HMEM_LIBRARIES GDRCopy::GDRCopy)
endif()
if(HAVE_ZE AND NOT LIBFABRIC_ZE_DLOPEN)
  list(APPEND LIBFABRIC_HMEM_LIBRARIES LevelZero::LevelZero)
endif()
if(HAVE_ROCR AND NOT LIBFABRIC_ROCR_DLOPEN)
  list(APPEND LIBFABRIC_HMEM_LIBRARIES ROCm::ROCm)
endif()
