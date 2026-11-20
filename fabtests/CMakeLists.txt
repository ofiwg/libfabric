# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
if(NOT LIBFABRIC_BUILD_FABTESTS)
  return()
endif()

message(STATUS "=== Fabtests Configuration ===")
check_symbol_exists(random_r "stdlib.h" FABTESTS_HAVE_RANDOM_R)

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
    common/hmem.c
    common/hmem_cuda.c
    common/hmem_ze.c
    common/hmem_rocr.c
    common/hmem_neuron.c
    common/hmem_synapseai.c
)

# Common shared library sources
set(FABTESTS_COMMON_SOURCES common/shared.c common/ofi_atomic.c
                            ${FABTESTS_OSD_SOURCES} ${FABTESTS_HMEM_SOURCES}
)

# Create the common library
add_library(fabtests_common STATIC ${FABTESTS_COMMON_SOURCES})

target_include_directories(
  fabtests_common
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
         ${CMAKE_CURRENT_SOURCE_DIR}
         ${FABTESTS_OSD_INCLUDE_DIR}
         ${libfabric_SOURCE_DIR}/include
         ${libfabric_BINARY_DIR}
)

target_link_libraries(
  fabtests_common
  PUBLIC Libfabric::fabric Threads::Threads
         $<$<BOOL:${LIBFABRIC_WINDOWS}>:ws2_32>
         $<$<BOOL:${LIBFABRIC_WINDOWS}>:iphlpapi>
)

target_link_libraries(
  fabtests_common PUBLIC libfabric_build_settings ${LIBFABRIC_CORE_LIBRARIES}
)

target_compile_definitions(
  fabtests_common
  PUBLIC
    HAVE_RANDOM_R=$<BOOL:${FABTESTS_HAVE_RANDOM_R}>
    HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE=$<AND:$<BOOL:${HAVE_CUDA}>,$<BOOL:${CUDAToolkit_HAS_DMABUF_MAPPING_TYPE_PCIE}>>
)

# =============================================================================
# Helper function to add a fabtest executable
# =============================================================================
# NAME and SOURCES define the executable; CATEGORY selects its IDE folder.
# EXTRA_LIBS adds dependencies beyond the shared fabtests_common requirements.
function(fabtests_add_test)
  cmake_parse_arguments(
    PARSE_ARGV
    0
    TEST
    ""
    "NAME;CATEGORY"
    "SOURCES;EXTRA_LIBS"
  )

  if(TEST_UNPARSED_ARGUMENTS
     OR NOT TEST_NAME
     OR NOT TEST_SOURCES
  )
    message(FATAL_ERROR "Invalid fabtest definition: ${ARGN}")
  endif()

  add_executable(${TEST_NAME} ${TEST_SOURCES})
  set_target_properties(
    ${TEST_NAME} PROPERTIES FOLDER "fabtests/${TEST_CATEGORY}"
  )

  target_link_libraries(${TEST_NAME} PRIVATE fabtests_common ${TEST_EXTRA_LIBS})

  if(LIBFABRIC_INSTALL)
    install(TARGETS ${TEST_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
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
      NAME fi_${_test}
      CATEGORY benchmarks
      SOURCES benchmarks/${_test}.c
      EXTRA_LIBS fabtests_benchmark
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
      NAME fi_${_test}
      CATEGORY functional
      SOURCES functional/${_test}.c
    )
  endforeach()

  # msg_epoll requires epoll (Linux only)
  if(HAVE_EPOLL)
    fabtests_add_test(
      NAME fi_msg_epoll
      CATEGORY functional
      SOURCES functional/msg_epoll.c
    )
  endif()

  # mcast test
  fabtests_add_test(
    NAME fi_mcast
    CATEGORY functional
    SOURCES functional/mcast.c
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
      check_hmem
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
      NAME fi_${_test}
      CATEGORY unit
      SOURCES unit/${_test}.c
      EXTRA_LIBS fabtests_unit_common
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
    NAME fi_multinode
    CATEGORY multinode
    SOURCES multinode/src/core.c
    EXTRA_LIBS fabtests_multinode_common
  )

  # Multinode collective test
  fabtests_add_test(
    NAME fi_multinode_coll
    CATEGORY multinode
    SOURCES multinode/src/core_coll.c
    EXTRA_LIBS fabtests_multinode_common
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
  NAME sighandler_test
  CATEGORY regression
  SOURCES regression/sighandler_test.c
)
if(NOT WIN32)
  fabtests_add_test(
    NAME pep_close_eq_read
    CATEGORY regression
    SOURCES regression/pep_close_eq_read.c
  )
  add_subdirectory(component)
endif()

if(HAVE_LPP AND LIBFABRIC_FABTESTS_LPP)
  add_subdirectory(prov/lpp)
endif()
if(HAVE_VERBS AND LIBFABRIC_FABTESTS_VERBS)
  fabtests_add_test(
    NAME fi_nic_affinity_test
    CATEGORY verbs
    SOURCES prov/verbs/src/nic_affinity_test.c unit/common.c
  )
endif()

# =============================================================================
# Provider-specific Tests: EFA
# =============================================================================

if(HAVE_EFA AND LIBFABRIC_FABTESTS_EFA)
  set(_efa prov/efa/src)
  foreach(_test efa_info_test efa_implicit_av_test efa_mr_test)
    fabtests_add_test(NAME fi_${_test} SOURCES ${_efa}/${_test}.c)
  endforeach()
  foreach(_test rdm_remote_exit_early rdm_rnr_queue_resend
                rdm_rnr_read_cq_error
  )
    string(REPLACE "rdm_rnr" "rnr" _name "${_test}")
    fabtests_add_test(
      NAME fi_efa_${_name} SOURCES ${_efa}/${_test}.c ${_efa}/efa_rnr_shared.c
    )
  endforeach()
  fabtests_add_test(
    NAME fi_efa_multi_ep_stress
    SOURCES ${_efa}/multi_ep_stress.c ${_efa}/efa_shared.c
            ${_efa}/efa_rnr_shared.c
  )
  fabtests_add_test(
    NAME fi_mr_abort SOURCES ${_efa}/mr_abort.c ${_efa}/efa_shared.c
  )
  fabtests_add_test(
    NAME fi_efa_runt_read_no_handshake
    SOURCES ${_efa}/efa_runt_read_no_handshake.c benchmarks/benchmark_shared.c
  )
  fabtests_add_test(
    NAME fi_efa_rma_bw SOURCES ${_efa}/efa_rma_bw.c ${_efa}/efa_shared.c
                               benchmarks/benchmark_shared.c
  )
  if(Efadv_FOUND)
    fabtests_add_test(
      NAME fi_efa_mmap_test
      SOURCES ${_efa}/efa_mmap_test.c
      EXTRA_LIBS Efadv::Efadv IBVerbs::IBVerbs
    )
    fabtests_add_test(
      NAME fi_efa_exhaust_mr_reg_rdm_pingpong
      SOURCES ${_efa}/efa_exhaust_mr_reg_rdm_pingpong.c
              ${_efa}/efa_exhaust_mr_reg_common.c benchmarks/benchmark_shared.c
      EXTRA_LIBS Efadv::Efadv IBVerbs::IBVerbs
    )
    if(Efadv_HAVE_RDMA_SIZE
       AND Efadv_HAVE_CAPS_RDMA_WRITE
       AND Efadv_HAVE_CAPS_UNSOLICITED_WRITE_RECV
    )
      fabtests_add_test(
        NAME fi_efa_rdma_checker
        SOURCES ${_efa}/efa_rdma_checker.c
        EXTRA_LIBS Efadv::Efadv IBVerbs::IBVerbs
      )
    endif()
    if(Efadv_HAVE_CREATE_COMP_CNTR)
      fabtests_add_test(
        NAME fi_efa_hw_cntr
        SOURCES ${_efa}/efa_hw_cntr_test.c benchmarks/benchmark_shared.c
        EXTRA_LIBS Efadv::Efadv IBVerbs::IBVerbs
      )
    endif()
  endif()
endif()

# =============================================================================
# Install Scripts and Test Configs
# =============================================================================

if(LIBFABRIC_INSTALL)
  install(
    PROGRAMS scripts/runfabtests.sh
             scripts/runfabtests.py
             scripts/runmultinode.sh
             scripts/runmultinode.py
             scripts/rft_yaml_to_junit_xml
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT fabtests
  )

  # Install test configs
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_configs/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fabtests/test_configs
    COMPONENT fabtests
    FILES_MATCHING
    PATTERN "*.test"
    PATTERN "*.json"
    PATTERN "*.exclude"
  )
  install(
    DIRECTORY pytest/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fabtests/pytest
    COMPONENT fabtests
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.yaml"
    PATTERN "*.ini"
  )

  # Install man pages
  foreach(_section 1 7)
    install(
      DIRECTORY man/man${_section}/
      DESTINATION ${CMAKE_INSTALL_MANDIR}/man${_section}
      COMPONENT fabtests
      FILES_MATCHING
      PATTERN "*.${_section}"
    )
  endforeach()

endif()
