# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)
include(CheckSymbolExists)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  check_c_source_compiles(
    "
    #include <net/if.h>
    #include <sys/types.h>
    #include <linux/ethtool.h>
    #include <linux/sockios.h>
    #include <sys/ioctl.h>
    int main() {
      unsigned long ioctl_req = SIOCETHTOOL;
      struct ethtool_cmd cmd = { .cmd = ETHTOOL_GSET };
      long speed = cmd.speed;
      return 0;
    }
  "
    Ethtool_FOUND
  )

  if(Ethtool_FOUND)
    check_symbol_exists(
      ethtool_cmd_speed "linux/ethtool.h" Ethtool_HAS_CMD_SPEED
    )
    check_symbol_exists(
      SPEED_UNKNOWN "linux/ethtool.h" Ethtool_HAS_SPEED_UNKNOWN
    )
  endif()
else()
  set(Ethtool_FOUND FALSE)
endif()

find_package_handle_standard_args(Ethtool REQUIRED_VARS Ethtool_FOUND)
