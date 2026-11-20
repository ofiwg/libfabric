# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
CheckCUDADmabuf
---------------

Probe CUDA DMA-BUF and context synchronization declarations.

Requires CUDAToolkit_INCLUDE_DIRS. Sets CUDAToolkit_HAS_DMABUF,
CUDAToolkit_HAS_DMABUF_MAPPING_TYPE_PCIE, and
CUDAToolkit_HAS_CTX_SYNC_MEMOPS using compile-only checks.

Unguarded: repeated callers need results in their own scope. CMake caches
individual probes internally. Required-check state is saved and restored.
This module is build-only and is not installed for consumers.
]]
include(CheckCSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state()
set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
# Compile only: runtime loading needs declarations, not link-time CUDA symbols.
set(_saved_try_compile_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_c_source_compiles(
  "
  #include <cuda.h>
  int main(void) {
    (void)&cuMemGetHandleForAddressRange;
    return CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED + CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD;
  }"
  CUDAToolkit_HAS_DMABUF
)
check_c_source_compiles(
  "
  #include <cuda.h>
  int main(void) {
    return CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
  }"
  CUDAToolkit_HAS_DMABUF_MAPPING_TYPE_PCIE
)
check_c_source_compiles(
  "
  #include <cuda.h>
  int main(void) {
    (void)&cuCtxGetFlags;
    (void)&cuCtxSetFlags;
    return CU_CTX_SYNC_MEMOPS;
  }"
  CUDAToolkit_HAS_CTX_SYNC_MEMOPS
)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_saved_try_compile_type}")
cmake_pop_check_state()
