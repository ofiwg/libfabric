# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
if(NOT LIBFABRIC_BUILD_FABTESTS)
  return()
endif()

message(STATUS "")
message(STATUS "=== Fabtests Configuration ===")

# =============================================================================
# Common Library
# =============================================================================

# Platform-specific sources
if(LIBFABRIC_MACOS)
  set(FABTESTS_OSD_SOURCES common/osx/osd.c)
  set(FABTESTS_OSD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/osx)
elseif(LIBFABRIC_FREEBSD)
  set(FABTESTS_OSD_SOURCES "")
  set(FABTESTS_OSD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/freebsd)
elseif(WIN32)
  set(FABTESTS_OSD_SOURCES common/windows/getopt.c common/windows/osd.c)
  set(FABTESTS_OSD_INCLUDE_DIR
      ${CMAKE_CURRENT_SOURCE_DIR}/include/windows
      ${CMAKE_CURRENT_SOURCE_DIR}/include/windows/getopt
  )
else()
  # Linux/Unix
  set(FABTESTS_OSD_SOURCES "")
  set(FABTESTS_OSD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/unix)
endif()

# HMEM (heterogeneous memory) sources All hmem_*.c files must be compiled
# unconditionally because hmem.c references all backends. When a backend is
# unavailable, the hmem_*.c file compiles to stub functions that return
# -FI_ENOSYS.
set(FABTESTS_HMEM_SOURCES
    common/hmem.c common/hmem_cuda.c common/hmem_ze.c common/hmem_rocr.c
    common/hmem_neuron.c common/hmem_synapseai.c
)

# Common shared library sources
set(FABTESTS_COMMON_SOURCES common/shared.c common/ofi_atomic.c
                            ${FABTESTS_OSD_SOURCES} ${FABTESTS_HMEM_SOURCES}
)

# Create the common library
add_library(fabtests_common STATIC ${FABTESTS_COMMON_SOURCES})

target_include_directories(
  fabtests_common
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${FABTESTS_OSD_INCLUDE_DIR}
         ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}
)

target_link_libraries(
  fabtests_common
  PUBLIC fabric Threads::Threads $<$<BOOL:${LIBFABRIC_WINDOWS}>:ws2_32>
         $<$<BOOL:${LIBFABRIC_WINDOWS}>:iphlpapi>
)

# Add GPU libraries if enabled
if(LIBFABRIC_ENABLE_CUDA AND CUDAToolkit_FOUND)
  target_compile_definitions(fabtests_common PRIVATE HAVE_CUDA)
  if(NOT LIBFABRIC_CUDA_DLOPEN)
    target_link_libraries(fabtests_common PUBLIC CUDA::cudart)
  endif()
endif()

if(LIBFABRIC_ENABLE_ZE AND LevelZero_FOUND)
  target_compile_definitions(fabtests_common PRIVATE HAVE_ZE)
  if(NOT LIBFABRIC_ZE_DLOPEN)
    target_link_libraries(fabtests_common PUBLIC LevelZero::LevelZero)
  endif()
endif()

if(LIBFABRIC_ENABLE_ROCR AND ROCm_FOUND)
  target_compile_definitions(fabtests_common PRIVATE HAVE_ROCR)
  if(NOT LIBFABRIC_ROCR_DLOPEN)
    target_link_libraries(fabtests_common PUBLIC ROCm::ROCm)
  endif()
endif()

# Platform definitions
target_compile_definitions(
  fabtests_common PRIVATE ${LIBFABRIC_PLATFORM_DEFINITIONS}
)

# =============================================================================
# Helper function to add a fabtest executable
# =============================================================================
function(fabtests_add_test)
  cmake_parse_arguments(TEST "" "NAME;CATEGORY" "SOURCES;EXTRA_LIBS" ${ARGN})

  if(NOT TEST_NAME)
    message(FATAL_ERROR "fabtests_add_test: NAME is required")
  endif()

  add_executable(${TEST_NAME} ${TEST_SOURCES})

  target_link_libraries(${TEST_NAME} PRIVATE fabtests_common ${TEST_EXTRA_LIBS})

  # Install to libexec/fabtests/<category>
  if(TEST_CATEGORY)
    install(
      TARGETS ${TEST_NAME}
      RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/fabtests/${TEST_CATEGORY}
              COMPONENT fabtests
    )
  else()
    install(TARGETS ${TEST_NAME}
            RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/fabtests
                    COMPONENT fabtests
    )
  endif()
endfunction()

# =============================================================================
# Benchmark Tests
# =============================================================================

if(LIBFABRIC_FABTESTS_BENCHMARKS)
  message(STATUS "Fabtests benchmarks: enabled")

  # Benchmark shared library
  add_library(fabtests_benchmark STATIC benchmarks/benchmark_shared.c)
  target_link_libraries(fabtests_benchmark PUBLIC fabtests_common)

  set(BENCHMARK_TESTS
      dgram_pingpong
      msg_bw
      msg_pingpong
      rdm_bw
      rdm_bw_mt
      rdm_cntr_pingpong
      rdm_pingpong
      rdm_tagged_bw
      rdm_tagged_pingpong
      rma_bw
      rma_pingpong
      rma_tx_completion
  )

  foreach(_test IN LISTS BENCHMARK_TESTS)
    fabtests_add_test(
      NAME
      fi_${_test}
      CATEGORY
      benchmarks
      SOURCES
      benchmarks/${_test}.c
      EXTRA_LIBS
      fabtests_benchmark
    )
  endforeach()
endif()

# =============================================================================
# Functional Tests
# =============================================================================

if(LIBFABRIC_FABTESTS_FUNCTIONAL)
  message(STATUS "Fabtests functional: enabled")

  set(FUNCTIONAL_TESTS
      av_xfer
      cm_data
      cq_data
      dgram
      flood
      inject_test
      loopback
      msg
      msg_sockets
      multi_ep
      multi_mr
      multi_recv
      rdm
      rdm_atomic
      rdm_deferred_wq
      rdm_multi_client
      rdm_multi_domain
      rdm_rma_event
      rdm_rma_trigger
      rdm_shared_av
      rdm_stress
      rdm_tagged_peek
      recv_cancel
      resmgmt_test
      scalable_ep
      shared_ctx
      unexpected_msg
      unmap_mem
  )

  foreach(_test IN LISTS FUNCTIONAL_TESTS)
    fabtests_add_test(
      NAME fi_${_test} CATEGORY functional SOURCES functional/${_test}.c
    )
  endforeach()

  # msg_epoll requires epoll (Linux only)
  if(HAVE_EPOLL)
    fabtests_add_test(
      NAME fi_msg_epoll CATEGORY functional SOURCES functional/msg_epoll.c
    )
  endif()

  # mcast test
  fabtests_add_test(
    NAME fi_mcast CATEGORY functional SOURCES functional/mcast.c
  )
endif()

# =============================================================================
# Unit Tests
# =============================================================================

if(LIBFABRIC_FABTESTS_UNIT)
  message(STATUS "Fabtests unit: enabled")

  # Unit test common library
  add_library(fabtests_unit_common STATIC unit/common.c)
  target_link_libraries(fabtests_unit_common PUBLIC fabtests_common)

  set(UNIT_TESTS
      av_test
      cntr_test
      cq_test
      dom_test
      eq_test
      getinfo_test
      mr_test
      mr_cache_evict
      setopt_test
  )

  foreach(_test IN LISTS UNIT_TESTS)
    fabtests_add_test(
      NAME
      fi_${_test}
      CATEGORY
      unit
      SOURCES
      unit/${_test}.c
      EXTRA_LIBS
      fabtests_unit_common
    )
  endforeach()
endif()

# =============================================================================
# Multinode Tests
# =============================================================================

if(LIBFABRIC_FABTESTS_MULTINODE)
  message(STATUS "Fabtests multinode: enabled")

  set(MULTINODE_COMMON_SOURCES multinode/src/harness.c multinode/src/pattern.c
                               multinode/src/timing.c
  )

  add_library(fabtests_multinode_common STATIC ${MULTINODE_COMMON_SOURCES})
  target_include_directories(
    fabtests_multinode_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/multinode/include
  )
  target_link_libraries(fabtests_multinode_common PUBLIC fabtests_common)

  # Main multinode test
  fabtests_add_test(
    NAME
    fi_multinode
    CATEGORY
    multinode
    SOURCES
    multinode/src/core.c
    EXTRA_LIBS
    fabtests_multinode_common
  )

  # Multinode collective test
  fabtests_add_test(
    NAME
    fi_multinode_coll
    CATEGORY
    multinode
    SOURCES
    multinode/src/core_coll.c
    EXTRA_LIBS
    fabtests_multinode_common
  )
endif()

# =============================================================================
# Ubertest
# =============================================================================

if(LIBFABRIC_FABTESTS_UBERTEST)
  message(STATUS "Fabtests ubertest: enabled")

  set(UBERTEST_SOURCES
      ubertest/config.c
      ubertest/connect.c
      ubertest/cq.c
      ubertest/domain.c
      ubertest/ep.c
      ubertest/test_ctrl.c
      ubertest/uber.c
      ubertest/verify.c
      ubertest/xfer.c
  )

  fabtests_add_test(NAME fi_ubertest SOURCES ${UBERTEST_SOURCES})
endif()

# =============================================================================
# Regression Tests
# =============================================================================

fabtests_add_test(
  NAME fi_sighandler_test CATEGORY regression SOURCES
  regression/sighandler_test.c
)

# =============================================================================
# Provider-specific Tests: EFA
# =============================================================================

if(LIBFABRIC_LINUX)
  # EFA provider tests (Linux only)
  set(EFA_TEST_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/prov/efa/src)

  # Common EFA test sources
  set(EFA_COMMON_SOURCES ${EFA_TEST_SOURCES_DIR}/efa_exhaust_mr_reg_common.c
                         ${EFA_TEST_SOURCES_DIR}/efa_rnr_shared.c
  )

  # EFA tests that don't require special dependencies
  fabtests_add_test(
    NAME fi_efa_info_test CATEGORY prov/efa SOURCES
    ${EFA_TEST_SOURCES_DIR}/efa_info_test.c
  )

  fabtests_add_test(
    NAME fi_efa_mmap_test CATEGORY prov/efa SOURCES
    ${EFA_TEST_SOURCES_DIR}/efa_mmap_test.c
  )

  fabtests_add_test(
    NAME fi_efa_rdma_checker CATEGORY prov/efa SOURCES
    ${EFA_TEST_SOURCES_DIR}/efa_rdma_checker.c
  )

  fabtests_add_test(
    NAME
    fi_efa_exhaust_mr_reg_rdm_pingpong
    CATEGORY
    prov/efa
    SOURCES
    ${EFA_TEST_SOURCES_DIR}/efa_exhaust_mr_reg_rdm_pingpong.c
    ${EFA_TEST_SOURCES_DIR}/efa_exhaust_mr_reg_common.c
  )

  fabtests_add_test(
    NAME fi_efa_implicit_av_test CATEGORY prov/efa SOURCES
    ${EFA_TEST_SOURCES_DIR}/efa_implicit_av_test.c
  )

  fabtests_add_test(
    NAME fi_efa_multi_ep_mt CATEGORY prov/efa SOURCES
    ${EFA_TEST_SOURCES_DIR}/multi_ep_mt.c
  )

  fabtests_add_test(
    NAME fi_efa_multi_ep_stress CATEGORY prov/efa SOURCES
    ${EFA_TEST_SOURCES_DIR}/multi_ep_stress.c
  )

  fabtests_add_test(
    NAME fi_efa_rdm_remote_exit_early CATEGORY prov/efa SOURCES
    ${EFA_TEST_SOURCES_DIR}/rdm_remote_exit_early.c
  )

  fabtests_add_test(
    NAME
    fi_efa_rdm_rnr_queue_resend
    CATEGORY
    prov/efa
    SOURCES
    ${EFA_TEST_SOURCES_DIR}/rdm_rnr_queue_resend.c
    ${EFA_TEST_SOURCES_DIR}/efa_rnr_shared.c
  )

  fabtests_add_test(
    NAME
    fi_efa_rdm_rnr_read_cq_error
    CATEGORY
    prov/efa
    SOURCES
    ${EFA_TEST_SOURCES_DIR}/rdm_rnr_read_cq_error.c
    ${EFA_TEST_SOURCES_DIR}/efa_rnr_shared.c
  )

  # EFA GDA (GPU Direct Async) test - requires CUDA
  if(LIBFABRIC_FABTESTS_EFA_GDA AND CUDAToolkit_FOUND)
    message(STATUS "Fabtests EFA GDA: enabled")

    # Build the efagda CUDA library
    enable_language(CUDA)

    add_library(
      efagda SHARED ${EFA_TEST_SOURCES_DIR}/efagda/efagda.cpp
                    ${EFA_TEST_SOURCES_DIR}/efagda/cuda_kernel.cu
    )

    target_include_directories(
      efagda PUBLIC ${EFA_TEST_SOURCES_DIR}/efagda ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(efagda PUBLIC CUDA::cudart CUDA::cuda_driver)

    set_target_properties(
      efagda PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE
                                                      ON
    )

    # Build the EFA GDA test
    fabtests_add_test(
      NAME
      fi_efa_gda
      CATEGORY
      prov/efa
      SOURCES
      ${EFA_TEST_SOURCES_DIR}/efa_gda.c
      EXTRA_LIBS
      efagda
      CUDA::cudart
      CUDA::cuda_driver
    )

    # Add include directory for efagda.h
    target_include_directories(
      fi_efa_gda PRIVATE ${EFA_TEST_SOURCES_DIR}/efagda
    )

    # Install efagda library
    install(TARGETS efagda LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                                   COMPONENT fabtests
    )
  endif()
endif()

# =============================================================================
# Install Scripts and Test Configs
# =============================================================================

# Install scripts
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
  DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/fabtests/scripts
  COMPONENT fabtests
  USE_SOURCE_PERMISSIONS FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.sh"
  PATTERN "runfabtests.sh"
)

# Install test configs
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_configs/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/fabtests/test_configs
  COMPONENT fabtests
  FILES_MATCHING
  PATTERN "*.test"
  PATTERN "*.json"
)

# Install man pages
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/man/
  DESTINATION ${CMAKE_INSTALL_MANDIR}/man7
  COMPONENT fabtests
  FILES_MATCHING
  PATTERN "*.7"
)

message(STATUS "")
