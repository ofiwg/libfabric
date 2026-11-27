# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
CheckEfadv
----------

Validate EFA direct-verbs symbols and optional features.

Requires Efadv and IBVerbs discovery; sets Efadv_HAVE_* and clears
Efadv_FOUND when the baseline API cannot be linked.

Efadv_HAVE_COMP_ACTION requires the create/destroy verbs, inline-size
query, and the completion-action block offset in efadv_wq_attr.

Unguarded: repeated callers need results in their own scope. CMake caches
individual probes internally. Required-check state is saved and restored.
This module is build-only and is not installed for consumers.
]]
include(CheckCSourceCompiles)
include(CheckSymbolExists)
include(CMakePushCheckState)

if(Efadv_FOUND AND IBVerbs_FOUND)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${Efadv_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES Efadv::Efadv IBVerbs::IBVerbs)

  check_symbol_exists(efadv_query_ah "infiniband/efadv.h" Efadv_HAVE_QUERY_AH)
  if(NOT Efadv_HAVE_QUERY_AH)
    set(Efadv_FOUND FALSE)
    cmake_pop_check_state()
    return()
  endif()

  # Check for struct efadv_device_attr.max_rdma_size
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      struct efadv_device_attr attr;
      (void)attr.max_rdma_size;
      return 0;
    }
  "
    Efadv_HAVE_RDMA_SIZE
  )

  # Check for EFADV_DEVICE_ATTR_CAPS_RNR_RETRY
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      int x = EFADV_DEVICE_ATTR_CAPS_RNR_RETRY;
      return x;
    }
  "
    Efadv_HAVE_CAPS_RNR_RETRY
  )

  # Check for EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      int x = EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE;
      return x;
    }
  "
    Efadv_HAVE_CAPS_RDMA_WRITE
  )

  # Check for EFADV_DEVICE_ATTR_CAPS_UNSOLICITED_WRITE_RECV
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      int x = EFADV_DEVICE_ATTR_CAPS_UNSOLICITED_WRITE_RECV;
      return x;
    }
  "
    Efadv_HAVE_CAPS_UNSOLICITED_WRITE_RECV
  )

  # Check for EFADV_DEVICE_ATTR_CAPS_CQ_WITH_EXT_MEM_DMABUF
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      int x = EFADV_DEVICE_ATTR_CAPS_CQ_WITH_EXT_MEM_DMABUF;
      return x;
    }
  "
    Efadv_HAVE_CAPS_CQ_WITH_EXT_MEM_DMABUF
  )

  # Check for extended CQ support (efadv_create_cq, efadv_cq_from_ibv_cq_ex,
  # efadv_wc_read_sgid)
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      (void)efadv_create_cq;
      (void)efadv_cq_from_ibv_cq_ex;
      (void)efadv_wc_read_sgid;
      return 0;
    }
  "
    Efadv_HAVE_CQ_EX
  )

  # Check for efadv_query_mr and related
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      (void)efadv_query_mr;
      struct efadv_mr_attr attr;
      (void)attr.rdma_recv_ic_id;
      int x = EFADV_MR_ATTR_VALIDITY_RDMA_READ_IC_ID;
      return x;
    }
  "
    Efadv_HAVE_QUERY_MR
  )

  # Check for struct efadv_qp_init_attr.sl
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      struct efadv_qp_init_attr attr;
      (void)attr.sl;
      return 0;
    }
  "
    Efadv_HAVE_SL
  )

  # Check for efadv_query_qp_wqs
  check_symbol_exists(
    efadv_query_qp_wqs "infiniband/efadv.h" Efadv_HAVE_QUERY_QP_WQS
  )

  # Check for efadv_query_cq
  check_symbol_exists(efadv_query_cq "infiniband/efadv.h" Efadv_HAVE_QUERY_CQ)

  # Check for struct efadv_cq_attr.doorbell
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main() {
      struct efadv_cq_attr attr;
      (void)attr.doorbell;
      return 0;
    }
  "
    Efadv_HAVE_CQ_ATTR_DB
  )

  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main(void) {
      struct efadv_qp_init_attr attr;
      (void)attr.wr_flags;
      (void)&efadv_qp_from_ibv_qp_ex;
      (void)&efadv_wr_set_processing_hints;
      return EFADV_WR_PROCESSING_HINT_BURST_PPS_SENSITIVE;
    }"
    Efadv_HAVE_WR_PROCESSING_HINTS
  )
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main(void) { struct efadv_wq_attr attr; (void)attr.caps; return 0; }
    "
    Efadv_HAVE_WQ_ATTR_CAPS
  )
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main(void) { struct efadv_device_attr attr; (void)attr.inline_buf_size_ex; return 0; }
    "
    Efadv_HAVE_INLINE_BUF_SIZE_EX
  )
  check_symbol_exists(
    efadv_create_comp_cntr "infiniband/efadv.h" Efadv_HAVE_CREATE_COMP_CNTR
  )
  check_symbol_exists(
    ibv_query_comp_cntr_caps "infiniband/verbs.h"
    Efadv_HAVE_IBV_QUERY_COMP_CNTR_CAPS
  )
  check_symbol_exists(
    efadv_create_mem_comp_action "infiniband/efadv.h"
    Efadv_HAVE_CREATE_MEM_COMP_ACTION
  )
  check_symbol_exists(
    efadv_destroy_comp_action "infiniband/efadv.h"
    Efadv_HAVE_DESTROY_COMP_ACTION
  )
  check_symbol_exists(
    efadv_get_max_inline_data "infiniband/efadv.h"
    Efadv_HAVE_GET_MAX_INLINE_DATA
  )
  check_c_source_compiles(
    "
    #include <infiniband/efadv.h>
    int main(void) {
      struct efadv_wq_attr attr;
      (void)attr.comp_action_with_data_block_offset;
      return 0;
    }"
    Efadv_HAVE_ACTION_BLOCK_OFFSET
  )
  cmake_pop_check_state()

  # Compute derived features
  if(Efadv_HAVE_QUERY_QP_WQS AND Efadv_HAVE_QUERY_CQ)
    set(Efadv_HAVE_DATA_PATH_DIRECT TRUE)
  else()
    set(Efadv_HAVE_DATA_PATH_DIRECT FALSE)
  endif()
  if(Efadv_HAVE_CREATE_MEM_COMP_ACTION
     AND Efadv_HAVE_DESTROY_COMP_ACTION
     AND Efadv_HAVE_GET_MAX_INLINE_DATA
     AND Efadv_HAVE_ACTION_BLOCK_OFFSET
  )
    set(Efadv_HAVE_COMP_ACTION TRUE)
  else()
    set(Efadv_HAVE_COMP_ACTION FALSE)
  endif()
endif()
