# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricSources
----------------

Maintain core, API, utility, hook, and collective source manifests.

Requires feature detection and LIBFABRIC_PLATFORM_SOURCES. Sets FABRIC_*_SOURCES
lists used to construct the object libraries. Keep these manifests aligned
with the corresponding Autotools Makefile.include files.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()
set(FABRIC_CORE_SOURCES
    src/fabric.c
    src/fi_tostr.c
    src/perf.c
    src/log.c
    src/var.c
    src/abi_1_0.c
    src/hmem.c
    src/hmem_rocr.c
    src/hmem_cuda.c
    src/hmem_cuda_gdrcopy.c
    src/hmem_ze.c
    src/hmem_neuron.c
    src/hmem_synapseai.c
    src/hmem_ipc_cache.c
    src/xpmem.c
    src/xpmem_cache.c
    src/common.c
    src/enosys.c
    src/rbtree.c
    src/tree.c
    src/fasthash.c
    src/indexer.c
    src/mem.c
    src/iov.c
    src/ofi_str.c
)

# Add io_uring source if available
if(HAVE_LIBURING)
  list(APPEND FABRIC_CORE_SOURCES src/iouring.c)
endif()

# Add rdpmc source if available
if(HAVE_LINUX_PERF_RDPMC)
  list(APPEND FABRIC_CORE_SOURCES src/linux/rdpmc.c)
endif()

# Utility provider sources (always included)
set(FABRIC_UTIL_SOURCES
    prov/util/src/util_atomic.c
    prov/util/src/util_attr.c
    prov/util/src/util_av.c
    prov/util/src/rxm_av.c
    prov/util/src/util_cq.c
    prov/util/src/util_cntr.c
    prov/util/src/util_domain.c
    prov/util/src/util_ep.c
    prov/util/src/util_pep.c
    prov/util/src/util_eq.c
    prov/util/src/util_fabric.c
    prov/util/src/util_main.c
    prov/util/src/util_poll.c
    prov/util/src/util_wait.c
    prov/util/src/util_buf.c
    prov/util/src/util_mr_map.c
    prov/util/src/util_ns.c
    prov/util/src/util_srx.c
    prov/util/src/util_mem_monitor.c
    prov/util/src/util_mem_hooks.c
    prov/util/src/util_mr_cache.c
    prov/util/src/cuda_mem_monitor.c
    prov/util/src/rocr_mem_monitor.c
    prov/util/src/ze_mem_monitor.c
    prov/util/src/cuda_ipc_monitor.c
    prov/util/src/rocr_ipc_monitor.c
    prov/util/src/ze_ipc_monitor.c
    prov/util/src/xpmem_monitor.c
    prov/util/src/kdreg2_mem_monitor.c
    prov/util/src/uffd_mem_monitor.c
    prov/util/src/import_mem_monitor.c
    prov/util/src/util_profile.c
)

# Hook infrastructure sources (always included)
set(FABRIC_HOOK_SOURCES
    prov/hook/src/hook.c
    prov/hook/src/hook_av.c
    prov/hook/src/hook_cm.c
    prov/hook/src/hook_cntr.c
    prov/hook/src/hook_cq.c
    prov/hook/src/hook_domain.c
    prov/hook/src/hook_ep.c
    prov/hook/src/hook_eq.c
    prov/hook/src/hook_wait.c
    prov/hook/src/hook_xfer.c
)

# Collective infrastructure sources (always included)
set(FABRIC_COLL_SOURCES
    prov/coll/src/coll_attr.c
    prov/coll/src/coll_av.c
    prov/coll/src/coll_av_set.c
    prov/coll/src/coll_coll.c
    prov/coll/src/coll_cq.c
    prov/coll/src/coll_domain.c
    prov/coll/src/coll_ep.c
    prov/coll/src/coll_eq.c
    prov/coll/src/coll_fabric.c
    prov/coll/src/coll_init.c
)

set(FABRIC_PUBLIC_SOURCES
    src/fabric.c
    src/fi_tostr.c
    src/perf.c
    src/log.c
    src/var.c
    src/abi_1_0.c
)
set(FABRIC_COMMON_SOURCES ${FABRIC_CORE_SOURCES})
list(REMOVE_ITEM FABRIC_COMMON_SOURCES ${FABRIC_PUBLIC_SOURCES})
list(
  APPEND
  FABRIC_COMMON_SOURCES
  ${FABRIC_UTIL_SOURCES}
  ${FABRIC_COLL_SOURCES}
  ${LIBFABRIC_PLATFORM_SOURCES}
)
