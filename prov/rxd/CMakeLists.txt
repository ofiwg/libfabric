# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# RXD provider (utility provider)
libfabric_add_provider(
  NAME rxd
  SOURCES src/rxd_atomic.c
          src/rxd_attr.c
          src/rxd_av.c
          src/rxd_cntr.c
          src/rxd_cq.c
          src/rxd_domain.c
          src/rxd_ep.c
          src/rxd_fabric.c
          src/rxd_init.c
          src/rxd_msg.c
          src/rxd_rma.c
          src/rxd_tagged.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
)
