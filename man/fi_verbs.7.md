---
layout: page
title: fi_verbs(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_verbs \- The Verbs Fabric Provider

# OVERVIEW

The verbs provider enables applications using OFI to be run over any verbs
hardware (Infiniband, iWarp, etc). It uses the Linux Verbs API for network
transport and provides a translation of OFI calls to appropriate verbs API calls.
It uses librdmacm for communication management and libibverbs for other control
and data transfer operations.

# SUPPORTED FEATURES

The verbs provider supports a subset of OFI features.

### Endpoint types
FI_EP_MSG, FI_EP_RDM

New change in libfabric v1.6:
FI_EP_RDM is supported through the OFI RxM utility provider. This is done
automatically when the app requests FI_EP_RDM endpoint. Please refer the
man page for RxM provider to learn more. The provider's internal support
for RDM endpoints is deprecated and would be removed from libfabric v1.7
onwards. Till then apps can explicitly request the internal RDM support by
disabling ofi_rxm provider through FI_PROVIDER env variable (FI_PROVIDER=^ofi_rxm).

### Endpoint capabilities and features

#### MSG endpoints
FI_MSG, FI_RMA, FI_ATOMIC and shared receive contexts.

#### RDM endpoints (internal - deprecated)
FI_MSG, FI_TAGGED, FI_RMA

#### DGRAM endpoints
FI_MSG

### Modes
Verbs provider requires applications to support the following modes:

#### FI_EP_MSG endpoint type

  * FI_LOCAL_MR / FI_MR_LOCAL mr mode.

  * FI_RX_CQ_DATA for applications that want to use RMA. Applications must
    take responsibility of posting receives for any incoming CQ data.

#### FI_EP_RDM endpoint type (internal - deprecated)

  * FI_CONTEXT

### Addressing Formats
Supported addressing formats include
  * MSG and RDM (internal - deprecated) EPs support:
    FI_SOCKADDR, FI_SOCKADDR_IN, FI_SOCKADDR_IN6, FI_SOCKADDR_IB
  * DGRAM supports:
    FI_ADDR_IB_UD

### Progress
Verbs provider supports FI_PROGRESS_AUTO: Asynchronous operations make forward
progress automatically.

### Operation flags
Verbs provider supports FI_INJECT, FI_COMPLETION, FI_REMOTE_CQ_DATA,
FI_TRANSMIT_COMPLETE.

### Msg Ordering
Verbs provider support the following message ordering:

  * Read after Read

  * Read after Write

  * Read after Send

  * Write after Write

  * Write after Send

  * Send after Write

  * Send after Send

and the following completion ordering:

  * TX contexts: FI_ORDER_STRICT
  * RX contexts: FI_ORDER_DATA

### Fork
Verbs provider supports the fork system call by default. See the limitations section
for restrictions. It can be turned off by setting the FI_FORK_UNSAFE environment
variable to "yes". This can improve the performance of memory registrations but it
also makes the use of fork unsafe.

### Memory Registration Cache
The verbs provider features a memory registration cache. This speeds up memory
registration calls from applications by caching registrations of frequently used
memory regions. The user can control the maximum combined size of all cache entries
and the maximum number of cache entries with the environment variables
FI_VERBS_MR_MAX_CACHED_SIZE and FI_VERBS_MR_MAX_CACHED_CNT respectively. Look below
in the environment variables section for details.

Note:
The memory registration cache framework hooks into alloc and free calls to monitor
the memory regions. If this doesn't work as expected caching would not be optimal.

# LIMITATIONS

### Memory Regions
Only FI_MR_BASIC mode is supported. Adding regions via s/g list is supported only
up to a s/g list size of 1. No support for binding memory regions to a counter.

### Wait objects
Only FI_WAIT_FD wait object is supported only for FI_EP_MSG endpoint type.
Wait sets are not supported.

### Resource Management
Application has to make sure CQs are not overrun as this cannot be detected
by the provider.

### Unsupported Features
The following features are not supported in verbs provider:

#### Unsupported Capabilities
FI_NAMED_RX_CTX, FI_DIRECTED_RECV, FI_TRIGGER, FI_RMA_EVENT

#### Other unsupported features
Scalable endpoints, FABRIC_DIRECT

#### Unsupported features specific to MSG endpoints
  * Counters, FI_SOURCE, FI_TAGGED, FI_PEEK, FI_CLAIM, fi_cancel, fi_ep_alias,
    shared TX context, cq_readfrom operations.
  * Completion flags are not reported if a request posted to an endpoint completes
    in error.

#### Unsupported features specific to RDM (internal - deprecated) endpoints
The RDM support for verbs have the following limitations:

  * Supports iovs of only size 1.

  * Wait objects are not supported.

  * Not thread safe.

### Fork
The support for fork in the provider has the following limitations:

  * Fabric resources like endpoint, CQ, EQ, etc. should not be used in the
    forked process.
  * The memory registered using fi_mr_reg has to be page aligned since ibv_reg_mr
    marks the entire page that a memory region belongs to as not to be re-mapped
    when the process is forked (MADV_DONTFORK).

# RUNTIME PARAMETERS

The verbs provider checks for the following environment variables.

### Common variables:

*FI_VERBS_TX_SIZE*
:  Default maximum tx context size (default: 384)

*FI_VERBS_RX_SIZE*
:  Default maximum rx context size (default: 384)

*FI_VERBS_TX_IOV_LIMIT*
: Default maximum tx iov_limit (default: 4). Note: RDM (internal - deprecated) EP type supports only 1

*FI_VERBS_RX_IOV_LIMIT*
: Default maximum rx iov_limit (default: 4). Note: RDM (internal - deprecated) EP type supports only 1

*FI_VERBS_INLINE_SIZE*
:  Default maximum inline size. Actual inject size returned in fi_info may be greater (default: 64)

*FI_VERBS_MIN_RNR_TIMER*
: Set min_rnr_timer QP attribute (0 - 31) (default: 12)

*FI_VERBS_USE_ODP*
: Enable On-Demand-Paging (ODP) experimental feature. The feature is supported only
  on Mellanox OFED (default: 0)

*FI_VERBS_CQREAD_BUNCH_SIZE*
: The number of entries to be read from the verbs completion queue at a time (default: 8).

*FI_VERBS_IFACE*
: The prefix or the full name of the network interface associated with the verbs
  device (default: ib)

*FI_VERBS_MR_CACHE_ENABLE*
: Enable Memory Registration caching (default: 0)

*FI_VERBS_MR_MAX_CACHED_CNT*
: Maximum number of cache entries (default: 4096)

*FI_VERBS_MR_MAX_CACHED_SIZE*
: Maximum total size of cache entries (default: 4 GB)

### Variables specific to RDM (internal - deprecated) endpoints

*FI_VERBS_RDM_BUFFER_NUM*
: The number of pre-registered buffers for buffered operations between the endpoints,
  must be a power of 2 (default: 8).

*FI_VERBS_RDM_BUFFER_SIZE*
: The maximum size of a buffered operation (bytes) (default: platform specific).

*FI_VERBS_RDM_RNDV_SEG_SIZE*
: The segment size for zero copy protocols (bytes)(default: 1073741824).

*FI_VERBS_RDM_THREAD_TIMEOUT*
: The wake up timeout of the helper thread (usec) (default: 100).

*FI_VERBS_RDM_EAGER_SEND_OPCODE*
: The operation code that will be used for eager messaging. Only IBV_WR_SEND and
  IBV_WR_RDMA_WRITE_WITH_IMM are supported. The last one is not applicable for iWarp.
  (default: IBV_WR_SEND)

*FI_VERBS_RDM_CM_THREAD_AFFINITY*
: If specified, bind the CM thread to the indicated range(s) of Linux virtual processor ID(s). This option is currently not supported on OS X. Usage: id_start[-id_end[:stride]][,]

### Variables specific to DGRAM endpoints

*FI_VERBS_DGRAM_USE_NAME_SERVER*
: The option that enables/disables OFI Name Server thread. The NS thread is used to
  resolve IP-addresses to provider specific addresses (default: 1, if "OMPI_COMM_WORLD_RANK"
  and "PMI_RANK" environment variables aren't defined)

*FI_VERBS_NAME_SERVER_PORT*
: The port on which Name Server thread listens incoming connections and requests (default: 5678)

### Environment variables notes
The fi_info utility would give the up-to-date information on environment variables:
fi_info -p verbs -e

# Troubleshooting / Known issues

When running an app over verbs provider with Valgrind, there may be reports of
memory leak in functions from dependent libraries (e.g. libibverbs, librdmacm).
These leaks are safe to ignore.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
