# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# LNX (Linux Network eXpress) provider
libfabric_add_provider(
  NAME lnx
  SOURCES src/lnx_av.c
          src/lnx_cq.c
          src/lnx_domain.c
          src/lnx_ep.c
          src/lnx_init.c
          src/lnx_mr.c
          src/lnx_ops.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
  LINUX_ONLY
)
