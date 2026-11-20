# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# TCP provider
libfabric_add_provider(
  NAME tcp
  SOURCES src/xnet_attr.c
          src/xnet_av.c
          src/xnet_cm.c
          src/xnet_cq.c
          src/xnet_domain.c
          src/xnet_ep.c
          src/xnet_eq.c
          src/xnet_fabric.c
          src/xnet_init.c
          src/xnet_msg.c
          src/xnet_pep.c
          src/xnet_profile.c
          src/xnet_progress.c
          src/xnet_rdm.c
          src/xnet_rdm_cm.c
          src/xnet_rma.c
          src/xnet_srx.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
)
