# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# CXI provider (HPE Cassini) Requires libcxi and the CXI driver UAPI headers
# (uapi/misc/cxi.h)

set(CXI_SOURCES
    src/cxip_atomic.c
    src/cxip_av.c
    src/cxip_avset.c
    src/cxip_coll.c
    src/cxip_coll_trace.c
    src/cxip_cmdq.c
    src/cxip_cntr.c
    src/cxip_cq.c
    src/cxip_ctrl.c
    src/cxip_curl.c
    src/cxip_dom.c
    src/cxip_ep.c
    src/cxip_eq.c
    src/cxip_evtq.c
    src/cxip_fabric.c
    src/cxip_faults.c
    src/cxip_if.c
    src/cxip_info.c
    src/cxip_iomm.c
    src/cxip_mr.c
    src/cxip_msg.c
    src/cxip_msg_hpc.c
    src/cxip_msg_rnr.c
    src/cxip_nic.c
    src/cxip_portals_table.c
    src/cxip_pte.c
    src/cxip_ptelist_buf.c
    src/cxip_rdzv_pte.c
    src/cxip_repsum.c
    src/cxip_req_buf.c
    src/cxip_rma.c
    src/cxip_rxc.c
    src/cxip_telemetry.c
    src/cxip_txc.c
    src/cxip_zbcoll.c
)

set(CXI_LINK_LIBRARIES "")
set(CXI_COMPILE_DEFINITIONS "")

if(LibCxi_FOUND)
  list(APPEND CXI_LINK_LIBRARIES LibCxi::LibCxi)

  # Feature detection defines
  if(LibCxi_HAVE_MODIFY_CP)
    list(APPEND CXI_COMPILE_DEFINITIONS "CXI_HAVE_MODIFY_CP=1")
  endif()

  if(LibCxi_HAVE_SVC_GET_VNI_RANGE)
    list(APPEND CXI_COMPILE_DEFINITIONS "CXI_HAVE_SVC_GET_VNI_RANGE=1")
  endif()

  if(LibCxi_HAVE_ALLOC_TRIG_CP)
    list(APPEND CXI_COMPILE_DEFINITIONS "CXI_HAVE_ALLOC_TRIG_CP=1")
  endif()
endif()

if(CXIDriverHeaders_FOUND)
  list(APPEND CXI_LINK_LIBRARIES CXIDriverHeaders::CXIDriverHeaders)
endif()

# Handle curl dlopen option
if(LIBFABRIC_CXI_CURL_DLOPEN)
  list(APPEND CXI_COMPILE_DEFINITIONS "ENABLE_CXI_CURL_DLOPEN=1")
else()
  list(APPEND CXI_COMPILE_DEFINITIONS "ENABLE_CXI_CURL_DLOPEN=0")
  if(LibCurl_FOUND)
    list(APPEND CXI_LINK_LIBRARIES LibCurl::LibCurl)
  endif()
endif()

# Handle json-c dlopen option
if(LIBFABRIC_CXI_JSON_DLOPEN)
  list(APPEND CXI_COMPILE_DEFINITIONS "ENABLE_CXI_JSON_DLOPEN=1")
else()
  list(APPEND CXI_COMPILE_DEFINITIONS "ENABLE_CXI_JSON_DLOPEN=0")
  if(JsonC_FOUND)
    list(APPEND CXI_LINK_LIBRARIES JsonC::JsonC)
  endif()
endif()

# CXI needs libm
list(APPEND CXI_LINK_LIBRARIES m)

libfabric_add_provider(
  NAME cxi
  SOURCES ${CXI_SOURCES}
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
  COMPILE_DEFINITIONS ${CXI_COMPILE_DEFINITIONS}
  DEPENDS LibCxi CXIDriverHeaders
  LINK_LIBRARIES ${CXI_LINK_LIBRARIES}
  LINUX_ONLY
)
