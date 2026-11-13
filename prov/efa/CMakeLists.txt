# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# EFA provider (requires libibverbs and libefa)

# EFA is only supported on 64-bit x86_64 and aarch64
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(STATUS "EFA provider: disabled (requires 64-bit system)")
  return()
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|aarch64|arm64)$")
  message(
    STATUS "EFA provider: disabled (only supported on x86_64 and aarch64)"
  )
  return()
endif()

set(EFA_SOURCES
    src/efa_device.c
    src/efa_thread_annotations.c
    src/efa_hmem.c
    src/efa_shm.c
    src/efa_av.c
    src/efa_av_array.c
    src/efa_ah.c
    src/rdm/efa_rdm_av.c
    src/rdm/efa_rdm_domain.c
    src/efa_domain.c
    src/efa_domain_util.c
    src/efa_fabric.c
    src/efa_fabric_util.c
    src/rdm/efa_rdm_fabric.c
    src/efa_mr.c
    src/efa_strerror.c
    src/efa_user_info.c
    src/efa_prov_info.c
    src/efa_fork_support.c
    src/efa_tp_def.c
    src/efa_base_ep.c
    src/efa_direct_ope.c
    src/efa_prov.c
    src/efa_env.c
    src/efa_cntr.c
    src/efa_hw_cntr.c
    src/efa_msg.c
    src/efa_rma.c
    src/efa_wr.c
    src/efa_cq.c
    src/efa_ep.c
    src/efa_data_path_direct.c
    src/rdm/efa_rdm_peer.c
    src/rdm/efa_rdm_cq.c
    src/rdm/efa_rdm_cntr.c
    src/rdm/efa_rdm_ep_utils.c
    src/rdm/efa_rdm_ep_fiops.c
    src/rdm/efa_rdm_rma.c
    src/rdm/efa_rdm_msg.c
    src/rdm/efa_rdm_pke.c
    src/rdm/efa_rdm_pke_utils.c
    src/rdm/efa_rdm_pke_rtm.c
    src/rdm/efa_rdm_pke_rtr.c
    src/rdm/efa_rdm_pke_rtw.c
    src/rdm/efa_rdm_pke_rta.c
    src/rdm/efa_rdm_pke_req.c
    src/rdm/efa_rdm_pke_nonreq.c
    src/rdm/efa_rdm_pke_cmd.c
    src/rdm/efa_rdm_pke_print.c
    src/rdm/efa_rdm_pkt_type.c
    src/rdm/efa_rdm_ope.c
    src/rdm/efa_rdm_rxe_map.c
    src/rdm/efa_rdm_atomic.c
    src/rdm/efa_rdm_tracepoint_def.c
    src/rdm/efa_rdm_srx.c
    src/rdm/efa_rdm_util.c
    src/rdm/efa_rdm_mr.c
    src/rdm/efa_rdm_proto.c
    src/rdm/protocols/efa_rdm_proto_atomic.c
    src/rdm/protocols/efa_rdm_proto_eager.c
    src/rdm/protocols/efa_rdm_proto_eager_write.c
    src/rdm/protocols/efa_rdm_proto_longread_write.c
    src/rdm/protocols/efa_rdm_proto_longcts_write.c
    src/rdm/protocols/efa_rdm_proto_medium.c
    src/rdm/protocols/efa_rdm_proto_short_rtr.c
    src/rdm/protocols/efa_rdm_proto_longcts.c
    src/rdm/protocols/efa_rdm_proto_longread.c
    src/rdm/protocols/efa_rdm_proto_runtread.c
    src/rdm/protocols/efa_rdm_proto_longcts_rtr.c
    src/rdm/protocols/efa_rdm_proto_zero_copy.c
)

# Build compile definitions from detected features
set(EFA_COMPILE_DEFS "")

# IBVerbs feature detection results
if(IBVerbs_FOUND)
  list(
    APPEND EFA_COMPILE_DEFS
    "HAVE_IBV_IS_FORK_INITIALIZED=$<BOOL:${IBVerbs_HAVE_IS_FORK_INITIALIZED}>"
  )

  list(
    APPEND
    EFA_COMPILE_DEFS
    "HAVE_EFA_DATA_IN_ORDER_ALIGNED_128_BYTES=$<BOOL:${IBVerbs_HAVE_DATA_IN_ORDER_ALIGNED_128_BYTES}>"
  )

  if(IBVerbs_HAVE_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY)
    list(APPEND EFA_COMPILE_DEFS
         "HAVE_IBV_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY=1"
    )
  else()
    list(APPEND EFA_COMPILE_DEFS
         "HAVE_IBV_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY=0"
    )
  endif()

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_EFA_DMABUF_MR=$<BOOL:${IBVerbs_HAVE_DMABUF_MR}>"
  )

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_EFA_CQ_NOTIFICATION=$<BOOL:${IBVerbs_HAVE_CQ_NOTIFICATION}>"
  )
endif()

# Efadv feature detection results
if(Efadv_FOUND)
  list(APPEND EFA_COMPILE_DEFS "HAVE_RDMA_SIZE=$<BOOL:${Efadv_HAVE_RDMA_SIZE}>")

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_CAPS_RNR_RETRY=$<BOOL:${Efadv_HAVE_CAPS_RNR_RETRY}>"
  )

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_CAPS_RDMA_WRITE=$<BOOL:${Efadv_HAVE_CAPS_RDMA_WRITE}>"
  )

  list(
    APPEND
    EFA_COMPILE_DEFS
    "HAVE_CAPS_UNSOLICITED_WRITE_RECV=$<BOOL:${Efadv_HAVE_CAPS_UNSOLICITED_WRITE_RECV}>"
  )

  list(
    APPEND
    EFA_COMPILE_DEFS
    "HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF=$<BOOL:${Efadv_HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF}>"
  )

  list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_CQ_EX=$<BOOL:${Efadv_HAVE_CQ_EX}>")

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_EFADV_QUERY_MR=$<BOOL:${Efadv_HAVE_QUERY_MR}>"
  )

  list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_SL=$<BOOL:${Efadv_HAVE_SL}>")

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_EFADV_QUERY_QP_WQS=$<BOOL:${Efadv_HAVE_QUERY_QP_WQS}>"
  )

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_EFADV_QUERY_CQ=$<BOOL:${Efadv_HAVE_QUERY_CQ}>"
  )

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_EFADV_CQ_ATTR_DB=$<BOOL:${Efadv_HAVE_CQ_ATTR_DB}>"
  )

  list(APPEND EFA_COMPILE_DEFS
       "HAVE_EFA_DATA_PATH_DIRECT=$<BOOL:${Efadv_HAVE_DATA_PATH_DIRECT}>"
  )
endif()

foreach(_feature WR_PROCESSING_HINTS WQ_ATTR_CAPS CREATE_COMP_CNTR COMP_ACTION)
  list(APPEND EFA_COMPILE_DEFS
       "HAVE_EFADV_${_feature}=$<BOOL:${Efadv_HAVE_${_feature}}>"
  )
endforeach()
list(
  APPEND EFA_COMPILE_DEFS
  "HAVE_INLINE_BUF_SIZE_EX=$<BOOL:${Efadv_HAVE_INLINE_BUF_SIZE_EX}>"
  "HAVE_IBV_QUERY_COMP_CNTR_CAPS=$<BOOL:${Efadv_HAVE_IBV_QUERY_COMP_CNTR_CAPS}>"
)

# EFA unit test support (disabled by default in CMake build)
list(APPEND EFA_COMPILE_DEFS "EFA_UNIT_TEST=0")

libfabric_add_provider(
  NAME efa
  SOURCES ${EFA_SOURCES}
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/rdm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rdm/protocols
  COMPILE_DEFINITIONS ${EFA_COMPILE_DEFS}
  DEPENDS IBVerbs Efadv
  LINK_LIBRARIES Efadv::Efadv IBVerbs::IBVerbs
  LINUX_ONLY
)

if(HAVE_EFA
   AND SynapseAI_FOUND
   AND NOT IBVerbs_HAVE_DMABUF_MR
)
  message(
    FATAL_ERROR
      "EFA with SynapseAI requires ibv_reg_dmabuf_mr (rdma-core >= 34.0)"
  )
endif()

# EFA memory poisoning option
if(HAVE_EFA AND LIBFABRIC_EFA_POISONING)
  target_compile_definitions(
    libfabric_provider_efa PRIVATE ENABLE_EFA_POISONING=1
  )
endif()
