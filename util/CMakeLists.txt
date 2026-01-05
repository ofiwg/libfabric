# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Utility programs

# fi_info - query fabric interfaces
libfabric_add_utility(NAME fi_info SOURCES info.c)

# fi_strerror - error string lookup
libfabric_add_utility(NAME fi_strerror SOURCES strerror.c)

# fi_pingpong - simple ping-pong benchmark
libfabric_add_utility(NAME fi_pingpong SOURCES pingpong.c)

# fi_mon_sampler - monitor sampler (only if monitor hook is enabled)
libfabric_add_utility(
  NAME fi_mon_sampler
  SOURCES mon_sampler.c
  CONDITION HAVE_MONITOR
)

# Install utilities
install(TARGETS fi_info fi_strerror fi_pingpong
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(TARGET fi_mon_sampler)
  install(TARGETS fi_mon_sampler RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
