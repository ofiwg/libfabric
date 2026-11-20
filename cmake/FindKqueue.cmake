# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
FindKqueue
----------

Probe kqueue availability on Darwin and FreeBSD.

Results: ``Kqueue_FOUND``.

Requires an enabled C compiler. Probe results use CMake's internal cache.

Unguarded: each find_package call must reevaluate its request and scope.
]]
include(CheckSymbolExists)
include(FindPackageHandleStandardArgs)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  check_symbol_exists(kqueue "sys/event.h" _HAVE_KQUEUE)

  if(_HAVE_KQUEUE)
    set(Kqueue_FOUND TRUE)
  else()
    set(Kqueue_FOUND FALSE)
  endif()
else()
  set(Kqueue_FOUND FALSE)
endif()

find_package_handle_standard_args(Kqueue REQUIRED_VARS Kqueue_FOUND)
