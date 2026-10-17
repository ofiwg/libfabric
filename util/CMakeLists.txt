# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
set(_utilities info strerror pingpong)
if(HAVE_MONITOR)
  list(APPEND _utilities mon_sampler)
endif()
foreach(_utility IN LISTS _utilities)
  add_executable(fi_${_utility} ${_utility}.c)
  target_include_directories(
    fi_${_utility} PRIVATE "${libfabric_BINARY_DIR}" "${libfabric_SOURCE_DIR}"
  )
  target_link_libraries(
    fi_${_utility} PRIVATE Libfabric::fabric libfabric_build_settings
  )
  if(WIN32)
    target_sources(fi_${_utility} PRIVATE windows/getopt/getopt.cpp)
    target_include_directories(fi_${_utility} PRIVATE windows/getopt)
    target_link_libraries(fi_${_utility} PRIVATE ws2_32 iphlpapi)
  endif()
  if(LIBFABRIC_INSTALL)
    install(TARGETS fi_${_utility} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endforeach()
