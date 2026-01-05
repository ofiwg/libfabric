# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# DMABUF peer memory hook (Linux only - uses /dev/dmabuf_reg device)
libfabric_add_hook_provider(
  NAME dmabuf_peer_mem
  LINUX_ONLY
  SOURCES src/hook_dmabuf_peer_mem.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/src
)
