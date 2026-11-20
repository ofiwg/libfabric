# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Multi-rail provider
libfabric_add_provider(
  NAME mrail
  SOURCES src/mrail_attr.c
          src/mrail_av.c
          src/mrail_cq.c
          src/mrail_domain.c
          src/mrail_ep.c
          src/mrail_fabric.c
          src/mrail_init.c
          src/mrail_rma.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
)
