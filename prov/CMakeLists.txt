# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Simple providers (no external dependencies)
add_subdirectory(tcp)
add_subdirectory(udp)
add_subdirectory(sockets)
add_subdirectory(rxm)
add_subdirectory(rxd)
add_subdirectory(mrail)
add_subdirectory(sm2)
add_subdirectory(lnx)

# Providers with external dependencies
add_subdirectory(verbs)
add_subdirectory(efa)
add_subdirectory(shm)
add_subdirectory(psm2)
add_subdirectory(psm3)
add_subdirectory(opx)
add_subdirectory(cxi)
add_subdirectory(ucx)
add_subdirectory(usnic)
add_subdirectory(lpp)

# Hook providers
add_subdirectory(hook)
