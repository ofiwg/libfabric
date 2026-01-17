# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# usNIC provider (Cisco usNIC) Requires libnl for route resolution (either libnl
# v1 or libnl v3)

set(USNIC_COMPILE_DEFS "")
set(USNIC_LINK_LIBRARIES "")

if(LibNL_FOUND)
  list(APPEND USNIC_LINK_LIBRARIES LibNL::LibNL)

  # Set HAVE_LIBNL3 based on detected version
  if(LibNL_IS_V3)
    list(APPEND USNIC_COMPILE_DEFS "HAVE_LIBNL3=1")
  else()
    list(APPEND USNIC_COMPILE_DEFS "HAVE_LIBNL3=0")
  endif()
endif()

# Note: The fake usnic verbs driver (usdf_fake_ibv.c) is not supported in the
# CMake build. It was only needed for very old libibverbs versions that lack
# verbs_register_driver(). Modern rdma-core (any version from the last ~8 years)
# has this function. If you need the fake driver for legacy systems, use the
# autotools build instead.
set(USNIC_BUILD_FAKE_VERBS_DRIVER 0)
list(APPEND USNIC_COMPILE_DEFS
     "USNIC_BUILD_FAKE_VERBS_DRIVER=${USNIC_BUILD_FAKE_VERBS_DRIVER}"
)

# usnic_direct library sources (low-level usNIC interface)
set(USNIC_DIRECT_SOURCES
    src/usnic_direct/libnl_utils_common.c
    src/usnic_direct/usd_caps.c
    src/usnic_direct/usd_dest.c
    src/usnic_direct/usd_device.c
    src/usnic_direct/usd_enum.c
    src/usnic_direct/usd_event.c
    src/usnic_direct/usd_ib_cmd.c
    src/usnic_direct/usd_ib_sysfs.c
    src/usnic_direct/usd_mem.c
    src/usnic_direct/usd_poll.c
    src/usnic_direct/usd_post.c
    src/usnic_direct/usd_post_ud_pio_udp.c
    src/usnic_direct/usd_post_ud_raw.c
    src/usnic_direct/usd_post_ud_udp.c
    src/usnic_direct/usd_queues.c
    src/usnic_direct/usd_socket.c
    src/usnic_direct/usd_vnic.c
    src/usnic_direct/usnic_ip_utils.c
    src/usnic_direct/vnic_cq.c
    src/usnic_direct/vnic_dev.c
    src/usnic_direct/vnic_intr.c
    src/usnic_direct/vnic_rq.c
    src/usnic_direct/vnic_wq.c
)

# Provider sources
set(USNIC_SOURCES
    src/usdf_av.c
    src/usdf_cm.c
    src/usdf_cq.c
    src/usdf_dgram.c
    src/usdf_domain.c
    src/usdf_endpoint.c
    src/usdf_ep_dgram.c
    src/usdf_eq.c
    src/usdf_ext.c
    src/usdf_fabric.c
    src/usdf_mem.c
    src/usdf_pep.c
    src/usdf_poll.c
    src/usdf_progress.c
    src/usdf_socket.c
    src/usdf_timer.c
    src/usdf_wait.c
)

# Add usnic_direct library sources
list(APPEND USNIC_SOURCES ${USNIC_DIRECT_SOURCES})

if(USNIC_BUILD_FAKE_VERBS_DRIVER)
  list(APPEND USNIC_SOURCES src/usdf_fake_ibv.c)
endif()

libfabric_add_provider(
  NAME usnic
  SOURCES ${USNIC_SOURCES}
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
               ${CMAKE_CURRENT_SOURCE_DIR}/src/usnic_direct
  COMPILE_DEFINITIONS ${USNIC_COMPILE_DEFS}
  DEPENDS IBVerbs LibNL
  LINK_LIBRARIES IBVerbs::IBVerbs ${USNIC_LINK_LIBRARIES}
  LINUX_ONLY
)
