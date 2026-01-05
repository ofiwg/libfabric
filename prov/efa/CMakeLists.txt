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
    src/efa_ah.c
    src/efa_av.c
    src/efa_base_ep.c
    src/efa_cntr.c
    src/efa_conn.c
    src/efa_cq.c
    src/efa_data_path_direct.c
    src/efa_device.c
    src/efa_domain.c
    src/efa_env.c
    src/efa_ep.c
    src/efa_fabric.c
    src/efa_fork_support.c
    src/efa_hmem.c
    src/efa_mr.c
    src/efa_msg.c
    src/efa_prov.c
    src/efa_prov_info.c
    src/efa_rma.c
    src/efa_shm.c
    src/efa_strerror.c
    src/efa_tp_def.c
    src/efa_user_info.c
)

# EFA RDM sources
file(GLOB EFA_RDM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/rdm/*.c")
list(APPEND EFA_SOURCES ${EFA_RDM_SOURCES})

# EFA DGRAM sources
file(GLOB EFA_DGRAM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/dgram/*.c")
list(APPEND EFA_SOURCES ${EFA_DGRAM_SOURCES})

# Build compile definitions from detected features
set(EFA_COMPILE_DEFS "")

# IBVerbs feature detection results
if(IBVerbs_FOUND)
  if(IBVerbs_HAVE_IS_FORK_INITIALIZED)
    list(APPEND EFA_COMPILE_DEFS "HAVE_IBV_IS_FORK_INITIALIZED=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_IBV_IS_FORK_INITIALIZED=0")
  endif()

  if(IBVerbs_HAVE_DATA_IN_ORDER_ALIGNED_128_BYTES)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFA_DATA_IN_ORDER_ALIGNED_128_BYTES=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFA_DATA_IN_ORDER_ALIGNED_128_BYTES=0")
  endif()

  if(IBVerbs_HAVE_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY)
    list(APPEND EFA_COMPILE_DEFS
         "HAVE_IBV_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY=1"
    )
  else()
    list(APPEND EFA_COMPILE_DEFS
         "HAVE_IBV_QUERY_QP_DATA_IN_ORDER_DEVICE_ONLY=0"
    )
  endif()

  if(IBVerbs_HAVE_DMABUF_MR)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFA_DMABUF_MR=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFA_DMABUF_MR=0")
  endif()

  if(IBVerbs_HAVE_CQ_NOTIFICATION)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFA_CQ_NOTIFICATION=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFA_CQ_NOTIFICATION=0")
  endif()
endif()

# Efadv feature detection results
if(Efadv_FOUND)
  if(Efadv_HAVE_RDMA_SIZE)
    list(APPEND EFA_COMPILE_DEFS "HAVE_RDMA_SIZE=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_RDMA_SIZE=0")
  endif()

  if(Efadv_HAVE_CAPS_RNR_RETRY)
    list(APPEND EFA_COMPILE_DEFS "HAVE_CAPS_RNR_RETRY=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_CAPS_RNR_RETRY=0")
  endif()

  if(Efadv_HAVE_CAPS_RDMA_WRITE)
    list(APPEND EFA_COMPILE_DEFS "HAVE_CAPS_RDMA_WRITE=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_CAPS_RDMA_WRITE=0")
  endif()

  if(Efadv_HAVE_CAPS_UNSOLICITED_WRITE_RECV)
    list(APPEND EFA_COMPILE_DEFS "HAVE_CAPS_UNSOLICITED_WRITE_RECV=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_CAPS_UNSOLICITED_WRITE_RECV=0")
  endif()

  if(Efadv_HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF)
    list(APPEND EFA_COMPILE_DEFS "HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF=0")
  endif()

  if(Efadv_HAVE_CQ_EX)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_CQ_EX=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_CQ_EX=0")
  endif()

  if(Efadv_HAVE_QUERY_MR)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_QUERY_MR=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_QUERY_MR=0")
  endif()

  if(Efadv_HAVE_SL)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_SL=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_SL=0")
  endif()

  if(Efadv_HAVE_QUERY_QP_WQS)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_QUERY_QP_WQS=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_QUERY_QP_WQS=0")
  endif()

  if(Efadv_HAVE_QUERY_CQ)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_QUERY_CQ=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_QUERY_CQ=0")
  endif()

  if(Efadv_HAVE_CQ_ATTR_DB)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_CQ_ATTR_DB=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFADV_CQ_ATTR_DB=0")
  endif()

  if(Efadv_HAVE_DATA_PATH_DIRECT)
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFA_DATA_PATH_DIRECT=1")
  else()
    list(APPEND EFA_COMPILE_DEFS "HAVE_EFA_DATA_PATH_DIRECT=0")
  endif()
endif()

# EFA unit test support (disabled by default in CMake build)
list(APPEND EFA_COMPILE_DEFS "EFA_UNIT_TEST=0")

libfabric_add_provider(
  NAME efa
  SOURCES ${EFA_SOURCES}
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/rdm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dgram
  COMPILE_DEFINITIONS ${EFA_COMPILE_DEFS}
  DEPENDS IBVerbs Efadv
  LINK_LIBRARIES IBVerbs::IBVerbs Efadv::Efadv
  LINUX_ONLY
)

# EFA memory poisoning option
if(HAVE_EFA AND LIBFABRIC_EFA_POISONING)
  target_compile_definitions(fabric PRIVATE ENABLE_EFA_POISONING=1)
endif()
