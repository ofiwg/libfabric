# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Sockets provider Requires shm_open (in libc or librt)

include(CheckSymbolExists)

# Check for shm_open (in libc or librt)
check_symbol_exists(shm_open "sys/mman.h" SOCKETS_HAVE_SHM_OPEN)
if(NOT SOCKETS_HAVE_SHM_OPEN)
  # Try with -lrt
  set(CMAKE_REQUIRED_LIBRARIES rt)
  check_symbol_exists(shm_open "sys/mman.h" SOCKETS_HAVE_SHM_OPEN_RT)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT SOCKETS_HAVE_SHM_OPEN_RT)
    message(STATUS "Sockets provider: disabled (shm_open not found)")
    return()
  endif()
endif()

set(SOCKETS_LINK_LIBRARIES "")
if(RT_FOUND)
  list(APPEND SOCKETS_LINK_LIBRARIES RT::RT)
endif()

libfabric_add_provider(
  NAME sockets
  SOURCES src/sock_atomic.c
          src/sock_attr.c
          src/sock_av.c
          src/sock_cntr.c
          src/sock_comm.c
          src/sock_conn.c
          src/sock_cq.c
          src/sock_ctx.c
          src/sock_dom.c
          src/sock_ep.c
          src/sock_ep_dgram.c
          src/sock_ep_msg.c
          src/sock_ep_rdm.c
          src/sock_eq.c
          src/sock_fabric.c
          src/sock_mr.c
          src/sock_msg.c
          src/sock_poll.c
          src/sock_progress.c
          src/sock_rma.c
          src/sock_rx_entry.c
          src/sock_trigger.c
          src/sock_wait.c
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  LINK_LIBRARIES ${SOCKETS_LINK_LIBRARIES}
)
