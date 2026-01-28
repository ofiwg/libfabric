# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# RXM provider (utility provider)
libfabric_add_provider(
  NAME rxm
  SOURCES src/rxm_atomic.c
          src/rxm_attr.c
          src/rxm_conn.c
          src/rxm_cq.c
          src/rxm_domain.c
          src/rxm_ep.c
          src/rxm_eq.c
          src/rxm_fabric.c
          src/rxm_hmem.c
          src/rxm_init.c
          src/rxm_msg.c
          src/rxm_rma.c
          src/rxm_tagged.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
)
