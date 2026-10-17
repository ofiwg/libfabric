# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# PSM2 provider (Intel Performance Scaled Messaging 2)

if(NOT PSM2_FOUND)
  return()
endif()

include(CheckSymbolExists)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES PSM2::PSM2)
check_symbol_exists(psm2_ep_disconnect2 "psm2.h" PSM2_HAVE_EP_DISCONNECT2)
check_symbol_exists(
  psm2_am_register_handlers_2 "psm2_am.h" _HAVE_PSM2_AM_REGISTER_HANDLERS_2
)
check_symbol_exists(psm2_mq_fp_msg "psm2_mq.h" _HAVE_PSM2_MQ_FP_MSG)
check_symbol_exists(
  psm2_mq_ipeek_dequeue_multi "psm2_mq.h" _HAVE_PSM2_MQ_REQ_USER
)
check_symbol_exists(psm2_info_query "psm2.h" _HAVE_PSM2_INFO_QUERY)
cmake_pop_check_state()
if(NOT PSM2_HAVE_EP_DISCONNECT2)
  message(STATUS "Provider psm2: requires psm2_ep_disconnect2")
  return()
endif()

libfabric_add_provider(
  NAME psm2
  SOURCES src/psmx2_am.c
          src/psmx2_atomic.c
          src/psmx2_attr.c
          src/psmx2_av.c
          src/psmx2_cm.c
          src/psmx2_cntr.c
          src/psmx2_cq.c
          src/psmx2_domain.c
          src/psmx2_ep.c
          src/psmx2_fabric.c
          src/psmx2_init.c
          src/psmx2_mr.c
          src/psmx2_msg.c
          src/psmx2_rma.c
          src/psmx2_tagged.c
          src/psmx2_trx_ctxt.c
          src/psmx2_util.c
          src/psmx2_wait.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
  COMPILE_DEFINITIONS
    HAVE_PSM2_SRC=0
    HAVE_PSM2_AM_REGISTER_HANDLERS_2=$<BOOL:${_HAVE_PSM2_AM_REGISTER_HANDLERS_2}>
    HAVE_PSM2_MQ_FP_MSG=$<AND:$<BOOL:${LIBFABRIC_PSM2_MQ_FP_MSG}>,$<BOOL:${_HAVE_PSM2_MQ_FP_MSG}>>
    HAVE_PSM2_MQ_REQ_USER=$<AND:$<BOOL:${LIBFABRIC_PSM2_MQ_REQ_USER}>,$<BOOL:${_HAVE_PSM2_MQ_REQ_USER}>>
    HAVE_PSM2_INFO_QUERY=$<BOOL:${_HAVE_PSM2_INFO_QUERY}>
  DEPENDS PSM2
  LINK_LIBRARIES PSM2::PSM2
)
if(TARGET libfabric_provider_psm2 AND LIBFABRIC_PSM2_DLOPEN)
  set_target_properties(libfabric_provider_psm2 PROPERTIES OUTPUT_NAME psmx2-fi)
endif()
