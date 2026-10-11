# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# LPP regression suite; accelerator helpers load their runtimes dynamically.
fabtests_add_test(
  NAME lpp_regression
  CATEGORY lpp
  SOURCES src/rcq_data.c
          src/main.c
          src/ipc.c
          src/error.c
          src/test_util.c
          src/rma.c
          src/msg.c
          src/atomic.c
)
target_include_directories(
  lpp_regression PRIVATE ${libfabric_SOURCE_DIR}/prov/lpp/include
)
if(HAVE_CUDA OR HAVE_ROCR)
  target_sources(
    lpp_regression PRIVATE src/hmem_util.c src/hmem_cuda.c src/hmem_rocm.c
  )
  target_compile_definitions(lpp_regression PRIVATE USE_HMEM)
endif()
if(HAVE_CUDA)
  target_compile_definitions(lpp_regression PRIVATE USE_CUDA)
endif()
if(HAVE_ROCR)
  # The core needs HSA headers; this regression helper additionally uses HIP.
  find_path(
    LIBFABRIC_FABTESTS_HIP_INCLUDE_DIR hip/hip_runtime_api.h
    HINTS ${ROCm_ROOT} ENV ROCM_PATH
    PATH_SUFFIXES include REQUIRED
  )
  mark_as_advanced(LIBFABRIC_FABTESTS_HIP_INCLUDE_DIR)
  target_include_directories(
    lpp_regression SYSTEM PRIVATE ${LIBFABRIC_FABTESTS_HIP_INCLUDE_DIR}
  )
  target_compile_definitions(
    lpp_regression PRIVATE USE_ROCM __HIP_PLATFORM_AMD__
  )
endif()
