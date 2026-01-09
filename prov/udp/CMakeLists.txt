# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# UDP provider
libfabric_add_provider(
  NAME udp
  SOURCES src/udpx_attr.c src/udpx_cq.c src/udpx_domain.c src/udpx_ep.c
          src/udpx_fabric.c src/udpx_init.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
)
