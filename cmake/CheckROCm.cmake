# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
CheckROCm
---------

Probe ROCr DMA-BUF declarations.

Requires ROCm_INCLUDE_DIRS. Sets ROCm_HAS_DMABUF; no runtime library is
needed for the declaration check.

Unguarded: repeated callers need results in their own scope. CMake caches
individual probes internally. Required-check state is saved and restored.
This module is build-only and is not installed for consumers.
]]
include(CheckCSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state()
set(CMAKE_REQUIRED_INCLUDES ${ROCm_INCLUDE_DIRS})
check_c_source_compiles(
  "
  #include <hsa/hsa_ext_amd.h>
  int main(void) {
    (void) &hsa_amd_portable_export_dmabuf;
    return HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED;
  }"
  ROCm_HAS_DMABUF
)
cmake_pop_check_state()
