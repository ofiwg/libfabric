# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# UCX provider
libfabric_add_provider(
  NAME ucx
  SOURCES src/ucx_av.c
          src/ucx_callbacks.c
          src/ucx_cm.c
          src/ucx_cntr.c
          src/ucx_core.c
          src/ucx_cq.c
          src/ucx_domain.c
          src/ucx_ep.c
          src/ucx_fabric.c
          src/ucx_init.c
          src/ucx_msg.c
          src/ucx_rma.c
          src/ucx_tagged.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
  DEPENDS UCX
  LINK_LIBRARIES UCX::UCX
  LINUX_ONLY
)
