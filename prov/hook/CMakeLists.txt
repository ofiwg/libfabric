# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Hook providers The base hook infrastructure is included in the core library
# These are optional hook providers

# Perf hook
add_subdirectory(perf)

# Trace hook
add_subdirectory(trace)

# Profile hook
add_subdirectory(profile)

# Monitor hook
add_subdirectory(monitor)

# Debug hook
add_subdirectory(hook_debug)

# HMEM hook
add_subdirectory(hook_hmem)

# DMABUF peer mem hook
add_subdirectory(dmabuf_peer_mem)
