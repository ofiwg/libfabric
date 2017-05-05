---
layout: page
title: fi_verbs(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The Verbs Fabric Provider

# OVERVIEW

The verbs provider enables applications using OFI to be run over any verbs
hardware (Infiniband, iWarp, etc). It uses the Linux Verbs API for network
transport and provides a translation of OFI calls to appropriate verbs API calls.
It uses librdmacm for communication management and libibverbs for other control
and data transfer operations.

# SUPPORTED FEATURES

The verbs provider supports a subset of OFI features.

### Endpoint types
FI_EP_MSG, FI_EP_RDM (Experimental support FI_MSG, FI_TAGGED, FI_RMA interfaces).

### Endpoint capabilities and features

#### MSG endpoints
FI_MSG, FI_RMA, FI_ATOMIC and shared receive contexts.

#### RDM endpoints
FI_MSG, FI_TAGGED, FI_RMA

### Modes
Verbs provider requires applications to support the following modes:

  * FI_LOCAL_MR for all applications.

  * FI_RX_CQ_DATA for applications that want to use RMA. Applications must
    take responsibility of posting receives for any incoming CQ data.

  * FI_CONTEXT for applications making uses of the experimental FI_EP_RDM capability.

### Addressing Formats
Supported addressing formats include FI_SOCKADDR, FI_SOCKADDR_IN, FI_SOCKADDR_IN6,
FI_SOCKADDR_IB

### Progress
Verbs provider supports FI_PROGRESS_AUTO: Asynchronous operations make forward
progress automatically.

### Operation flags
Verbs provider supports FI_INJECT, FI_COMPLETION, FI_REMOTE_CQ_DATA.

### Msg Ordering
Verbs provider support the following messaging ordering on the TX side:

  * Read after Read

  * Read after Write

  * Read after Send

  * Write after Write

  * Write after Send

  * Send after Write

  * Send after Send

### Fork
Verbs provider supports the fork system call by default. See the limitations section
for restrictions. It can be turned off by setting the FI_FORK_UNSAFE environment
variable to "yes". This can improve the performance of memory registrations but it
also makes the use of fork unsafe.

# LIMITATIONS

### Memory Regions
Only FI_MR_BASIC mode is supported. Adding regions via s/g list is not supported.
Generic fi_mr_regattr is not supported. No support for binding memory regions to
a counter.

### Wait objects
Only FI_WAIT_FD wait object is supported. Wait sets are not supported.

### Resource Management
Application has to make sure CQs are not overrun as this cannot be detected
by the provider.

### Unsupported Features
The following features are not supported in verbs provider:

#### Unsupported Endpoint types
FI_EP_DGRAM

#### Unsupported Capabilities
FI_NAMED_RX_CTX, FI_DIRECTED_RECV, FI_TRIGGER, FI_MULTI_RECV, FI_RMA_EVENT, FI_FENCE

#### Other unsupported features
Scalable endpoints, FABRIC_DIRECT

#### Unsupported features specific to MSG endpoints
Counters, FI_SOURCE, FI_TAGGED, FI_PEEK, FI_CLAIM, fi_cancel, fi_ep_alias,
shared TX context, cq_readfrom operations.

#### Unsupported features specific to RDM endpoints
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

### Variables specific to RDM endpoints

*FI_VERBS_IFACE*
: The prefix or the full name of the network interface associated with the IB device (default: ib)

*FI_VERBS_RDM_BUFFER_NUM*
: The number of pre-registered buffers for buffered operations between the endpoints,
  must be a power of 2 (default: 8).

*FI_VERBS_RDM_BUFFER_SIZE*
: The maximum size of a buffered operation (bytes) (default: platform specific).

*FI_VERBS_RDM_RNDV_SEG_SIZE*
: The segment size for zero copy protocols (bytes)(default: 1073741824).

*FI_VERBS_RDM_CQREAD_BUNCH_SIZE*
: The number of entries to be read from the verbs completion queue at a time (default: 8).

*FI_VERBS_RDM_THREAD_TIMEOUT*
: The wake up timeout of the helper thread (usec) (default: 100).

*FI_VERBS_RDM_EAGER_SEND_OPCODE*
: The operation code that will be used for eager messaging. Only IBV_WR_SEND and
  IBV_WR_RDMA_WRITE_WITH_IMM are supported. The last one is not applicable for iWarp.
  (default: IBV_WR_SEND)

### Environment variables notes
The fi_info utility would give the up-to-date information on environment variables:
fi_info -p verbs -e

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
