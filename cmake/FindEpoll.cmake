# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CheckSymbolExists)
include(FindPackageHandleStandardArgs)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  check_symbol_exists(epoll_create "sys/epoll.h" _HAVE_EPOLL_CREATE)
  check_symbol_exists(epoll_create1 "sys/epoll.h" _HAVE_EPOLL_CREATE1)

  if(_HAVE_EPOLL_CREATE1 OR _HAVE_EPOLL_CREATE)
    set(Epoll_FOUND TRUE)
  else()
    set(Epoll_FOUND FALSE)
  endif()
else()
  set(Epoll_FOUND FALSE)
endif()

find_package_handle_standard_args(Epoll REQUIRED_VARS Epoll_FOUND)
