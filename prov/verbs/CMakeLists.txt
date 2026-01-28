# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Verbs provider (requires libibverbs and librdmacm)

# Build compile definitions from detected features
set(VERBS_COMPILE_DEFS "")

if(IBVerbs_FOUND)
  # VERBS_HAVE_QUERY_EX - ibv_query_device_ex support
  if(IBVerbs_HAVE_QUERY_EX)
    list(APPEND VERBS_COMPILE_DEFS "VERBS_HAVE_QUERY_EX=1")
  else()
    list(APPEND VERBS_COMPILE_DEFS "VERBS_HAVE_QUERY_EX=0")
  endif()

  # VERBS_HAVE_DMABUF_MR - ibv_reg_dmabuf_mr support
  if(IBVerbs_HAVE_DMABUF_MR)
    list(APPEND VERBS_COMPILE_DEFS "VERBS_HAVE_DMABUF_MR=1")
  else()
    list(APPEND VERBS_COMPILE_DEFS "VERBS_HAVE_DMABUF_MR=0")
  endif()

  # VERBS_HAVE_XRC - IBV_QPT_XRC_SEND support (requires rdma_create_qp_ex too)
  # Per autotools: XRC requires both IBV_QPT_XRC_SEND and rdma_create_qp_ex
  if(IBVerbs_HAVE_XRC_SEND AND RdmaCM_HAVE_CREATE_QP_EX)
    list(APPEND VERBS_COMPILE_DEFS "VERBS_HAVE_XRC=1")
  else()
    list(APPEND VERBS_COMPILE_DEFS "VERBS_HAVE_XRC=0")
  endif()
endif()

if(RdmaCM_FOUND)
  # VERBS_HAVE_RDMA_ESTABLISH - rdma_establish support
  if(RdmaCM_HAVE_RDMA_ESTABLISH)
    list(APPEND VERBS_COMPILE_DEFS "VERBS_HAVE_RDMA_ESTABLISH=1")
  else()
    list(APPEND VERBS_COMPILE_DEFS "VERBS_HAVE_RDMA_ESTABLISH=0")
  endif()
endif()

libfabric_add_provider(
  NAME verbs
  SOURCES src/verbs_cm.c
          src/verbs_cm_xrc.c
          src/verbs_cq.c
          src/verbs_dgram_av.c
          src/verbs_dgram_ep_msg.c
          src/verbs_domain.c
          src/verbs_domain_xrc.c
          src/verbs_ep.c
          src/verbs_eq.c
          src/verbs_info.c
          src/verbs_init.c
          src/verbs_mr.c
          src/verbs_msg.c
          src/verbs_profile.c
          src/verbs_rma.c
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<IF:$<BOOL:${LIBFABRIC_WINDOWS}>,${CMAKE_CURRENT_SOURCE_DIR}/include/windows,${CMAKE_CURRENT_SOURCE_DIR}/include/linux>
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  COMPILE_DEFINITIONS ${VERBS_COMPILE_DEFS}
  DEPENDS IBVerbs RdmaCM
  LINK_LIBRARIES IBVerbs::IBVerbs RdmaCM::RdmaCM
)
