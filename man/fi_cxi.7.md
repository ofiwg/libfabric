---
layout: page
title: fi_cxi(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_cxi \- The CXI Fabric Provider

# OVERVIEW

The CXI provider enables libfabric on Cray's Slingshot network. Slingshot is
comprised of the Rosetta switch and Cassini NIC. Slingshot is an
Ethernet-compliant network. However, The provider takes advantage of proprietary
extensions to support HPC applications.

The CXI provider supports reliable, connection-less endpoint semantics. It
supports two-sided messaging interfaces with message matching offloaded by the
Cassini NIC. It also supports one-sided RMA and AMO interfaces, light-weight
counting events, triggered operations (via the deferred work API), and
fabric-accelerated small reductions.

# REQUIREMENTS

The CXI Provider requires Cassini's optimized HPC protocol which is only
supported in combination with the Rosetta switch.

The provider uses the libCXI library for control operations and a set of
Cassini-specific header files to enable direct hardware access in the data path.

# SUPPORTED FEATURES

The CXI provider supports a subset of OFI features.

## Endpoint types

The provider supports the *FI_EP_RDM* endpoint type, including scalable
endpoints.

## Address vectors

The provider implements both the *FI_AV_MAP* and *FI_AV_TABLE* address vector
types. *FI_EVENT* is unsupported.

## Memory registration modes

The provider implements scalable memory registration. The provider requires
*FI_MR_ENDPOINT*.

## Data transfer operations

The following data transfer interfaces are supported: *FI_ATOMIC*, *FI_MSG*,
*FI_RMA*, *FI_TAGGED*.  See DATA TRANSFER OPERATIONS below for more details.

## Completion events

The CXI provider supports all CQ event formats. Wait objects are not currently
supported.

## Modes

The CXI provider does not require any operation modes.

## Progress

The CXI provider currently supports *FI_PROGRESS_MANUAL* data and control
progress modes.

## Multi-threading

The CXI provider does not currently optimize for threading model. Data transfer
and control interfaces are always considered thread-safe.

## Wait Objects

The CXI provider does not currently support wait objects.

## Additional Features

The CXI provider also supports the following capabilities and features:

* *FI_MULTI_RECV*
* *FI_SOURCE*
* *FI_NAMED_RX_CTX*
* *FI_SHARED_AV*
* *FI_RM_ENABLED*
* *FI_RMA_EVENT*
* *FI_REMOTE_CQ_DATA*
* *FI_MORE*
* *FI_FENCE*

## Addressing Format

The CXI provider uses a proprietary address format. This format includes fields
for NIC Address and PID. NIC Address is the topological address of the NIC
endpoint on the fabric. All OFI Endpoints sharing a Domain share the same NIC
Address. PID (for Port ID or Process ID, adopted from the Portals 4
specification), is analogous to an IP socket port number. Valid PIDs are in the
range [0-510].

A third component of Slingshot network addressing is the Virtual Network ID
(VNI). VNI is a protection key used by the Slingshot network to provide
isolation between applications. A VNI defines an isolated PID space for a given
NIC. Therefore, Endpoints must use the same VNI in order to communicate. Note
that VNI is not a field of the CXI address, but rather is specified as part of
the OFI Endpoint auth_key. The combination of NIC Address, VNI, and PID is
unique to a single OFI Endpoint within a Slingshot fabric.

The NIC Address of an OFI Endpoint is inherited from the Domain. By default, a
PID is automatically assigned to an Endpoint when it is enabled. The address of
an Endpoint can be queried using fi_getname. The address received from
fi_getname may then be inserted into a peer's Address Vector. The resulting FI
address may then be used to perform an RDMA operation.

Alternatively, a client may manage PID assignment. fi_getinfo may be used to
create an fi_info structure that can be used to create an Endpoint with a
client-specified address. To achieve this, use fi_getinfo with the *FI_SOURCE*
flag set and set node and service strings to represent the local NIC interface
and PID to be assigned to the Endpoint. The NIC interface string should match
the name of an available CXI domain (in the format cxi[0-9]). The PID string
will be interpreted as a 9-bit integer. Address conflicts will be detected when
the Endpoint is enabled.

A Scalable Endpoint is assigned one PID for each pair of TX/RX contexts supported.

## Authorization Keys

The CXI authorization key format is defined by struct cxi_auth_key. This
structure is defined in fi_cxi_ext.h.

```c
struct cxi_auth_key {
	uint32_t svc_id;
	uint16_t vni;
};
```

The CXI authorization key format includes a VNI and CXI service ID. VNI is a
component of the CXI Endpoint address that provides isolation. A CXI service is
a software container which defines a set of local CXI resources, VNIs, and
Traffic Classes which a process can access.

Two endpoints must use the same VNI in order to communicate. Generally, a
parallel application should be assigned to a unique VNI on the fabric in order
to achieve network traffic and address isolation. Typically a privileged
entity, like a job launcher, will allocate one or more VNIs for use by an
application.

The CXI service API is provided by libCXI. It enables a privileged entity, like
an application launcher, to control an unprivileged process's access to NIC
resources. Generally, a parallel application should be assigned to a unique CXI
service in order to control access to local resources, VNIs, and Traffic
Classes.

An application provided authorization key is optional. If an authorization key is
not provided by the application, a default VNI and service will be assigned.
Isolation is not guaranteed when using a default VNI and service.

A custom authorization key must be provided during Domain allocation. An
Endpoint will inherit the parent Domain's VNI and service ID. It is an error to
create an Endpoint with VNI or service ID that does not match the parent
Domain.

The expected application launch workflow for a CXI-integrated launcher is as
follows:

1. A parallel application is launched.
2. The launcher allocates one or more VNIs for use by the application.
3. The launcher communicates with compute node daemons where the application
   will be run.
4. The launcher compute node daemon configures local CXI interfaces. libCXI is
   used to allocate one or more services for the application. The service will
   define the local resources, VNIs, and Traffic Classes that the application
   may access. Service allocation policies must be defined by the launcher.
   libCXI returns an ID to represent a service.
5. The launcher forks application processes.
6. The launcher provides one or more service IDs and VNI values to the
   application processes.
7. Application processes select from the list of available service IDs and VNIs
   to form an authorization key to use for Endpoint allocation.

## Address Vectors

Currently, the CXI provider supports both *FI_AV_TABLE* and *FI_AV_MAP* with the
same internal implementation. Optimizations are planned for *FI_AV_MAP*. In the
future, when using *FI_AV_MAP*, the CXI address will be encoded in the FI address.
This will avoid per-operation node address translation and reduce AV memory
footprint.

The CXI provider uses the *FI_SYMMETRIC* AV flag for optimization. When a
client guarantees that all processes have symmetric AV layout, the provider
uses FI addresses for source address matching (rather than physical addresses).
This reduces the overhead for source address matching during two-sided Receive
operations.

## Operation flags

The CXI provider supports the following Operation flags:

*FI_MORE*
:   When *FI_MORE* is specified in a data transfer operation, the provider will
    defer submission of RDMA commands to hardware. When one or more data
    transfer operations is performed using *FI_MORE*, followed by an operation
    without *FI_MORE*, the provider will submit the entire batch of queued
    operations to hardware using a single PCIe transaction, improving PCIe
    efficiency.

    When *FI_MORE* is used, queued commands will not be submitted to hardware
    until another data transfer operation is performed without *FI_MORE*.

*FI_TRANSMIT_COMPLETE*
:   By default, all CXI provider completion events satisfy the requirements of
    the 'transmit complete' completion level. Transmit complete events are
    generated when the intiator receives an Ack from the target NIC. The Ack is
    generated once all data has been received by the target NIC. Transmit
    complete events do not guarantee that data is visibile to the target
    process.

*FI_DELIVERY_COMPLETE*
:   When the 'delivery complete' completion level is used, the event guarantees
    that data is visible to the target process. To support this, hardware at
    the target performs a zero-byte read operation to flush data across the
    PCIe bus before generating an Ack. Flushing reads are performed
    unconditionally and will lead to higher latency.

*FI_MATCH_COMPLETE*
:   When the 'match complete' completion level is used, the event guarantees
    that the message has been matched to a client-provided buffer. All messages
    longer than the eager threshold support this guarantee. When 'match
    complete' is used with a Send that is shorter than the eager threshold, an
    additional handshake may be performed by the provider to notify the
    initiator that the Send has been matched.

The CXI provider also supports the following operation flags:

* *FI_INJECT*
* *FI_FENCE*
* *FI_COMPLETION*
* *FI_REMOTE_CQ_DATA*

## Scalable Endpoints

The CXI provider supports Scalable Endpoints (SEPs). A pair of TX/RX contexts
is generally used by a single thread. For that reason, a pair of TX/RX contexts
shares transmit and receive resources.

Each pair of contexts is assigned one PID value. It follows that a SEP with 10
TX and RX contexts is assigned 10 PIDs. A client-specified PID value will be
used as the base PID value for a SEP. For example, a SEP with 10 TX and RX
contexts with an assigned PID of 100 will use PIDs 100-109.

Due to a hardware matching limitation, a SEP that supports messaging (*FI_MSG* or
*FI_TAGGED*) and *FI_DIRECTED_RECV* must use an AV with *FI_SYMMETRIC* set.

## Messaging

The CXI provider supports both tagged (*FI_TAGGED*) and untagged (*FI_MSG*)
two-sided messaging interfaces. In the normal case, message matching is
performed by hardware. In certain low resource conditions, the responsibility to
perform message matching may be transferred to software. This is transparently
handled by the provider.

If a Send operation arrives at a node where there is no matching Receive
operation posted, it is considered unexpected. Unexpected messages are
supported. The provider manages buffers to hold unexpected message data.

Unexpected message handling is transparent to clients. Despite that, clients
should take care to avoid excessive use of unexpected messages by pre-posting
Receive operations. An unexpected message ties up hardware and memory resources
until it is matched with a user buffer.

The CXI provider implements several message protocols internally. Message
protocol is selected based on payload length. Short messages are transferred
using the eager protocol. In the eager protocol, the entire message payload is
sent along with the message header. If an eager message arrives unexpectedly,
the entire message is buffered at the target until it is matched to a Receive
operation.

Long messages are transferred using a rendezvous protocol. The provider
implements two rendezvous protocols: offloaded and eager. The threshold at which
the rendezvous protocol is used is controlled with the *FI_CXI_RDZV_THRESHOLD*
environment variable.

In the offloaded rendezvous protocol, a portion of the message payload is sent along
with the message header. Once the header is matched to a Receive operation, the
remainder of the payload is pulled from the source using an RDMA Get operation.
If the message arrives unexpectedly, the eager portion of the payload is
buffered at the target until it is matched to a Receive operation. In the
normal case, the Get is performed by hardware and the operation completes without
software progress.

In the eager rendezvous protocol, the entire payload is sent along with the
message header. If the message matches a pre-posted Receive operation, the
entire payload is written directly to the matched Receive buffer. If the message
arrives unexpectedly, the message header is saved and the entire payload is
dropped. Later, when the message is matched to a Receive operation, the entire
payload is pulled from the source using an RDMA Get operation.

The rendezvous protcol is controlled using the *FI_CXI_RDZV_OFFLOAD* environment
variable. The provider uses the offloaded rendezvous protocol by default.

## Message Ordering

The CXI provider supports the following ordering rules:

* All message Send operations are always ordered.
* RMA Writes may be ordered by specifying *FI_ORDER_RMA_WAW*.
* AMOs may be ordered by specifying *FI_ORDER_AMO_{WAW|WAR|RAW|RAR}*.
* RMA Writes may be ordered with respect to AMOs by specifying *FI_ORDER_WAW*.
  Fetching AMOs may be used to perform short reads that are ordered with
  respect to RMA Writes.

Ordered RMA size limits are set as follows:

* *max_order_waw_size* is -1. RMA Writes and non-fetching AMOs of any size are
  ordered with respect to each other.
* *max_order_raw_size* is -1. Fetching AMOs of any size are ordered with
  respect to RMA Writes and non-fetching AMOs.
* *max_order_war_size* is -1. RMA Writes and non-fetching AMOs of any size are
  ordered with respect to fetching AMOs.

## PCIe Ordering

Generally, PCIe writes are strictly ordered. As an optimization, PCIe TLPs may
have the Relaxed Order (RO) bit set to allow writes to be reordered. Cassini
sets the RO bit in PCIe TLPs when possible. Cassini sets PCIe RO as follows:

* Ordering of messaging operations is established using completion events.
  Therefore, all PCIe TLPs related to two-sided message payloads will have RO
  set.
* Every PCIe TLP associated with an unordered RMA or AMO operation will have RO
  cleared.
* PCIe TLPs associated with the last packet of an ordered RMA or AMO operation
  will have RO cleared.
* PCIe TLPs associated with the body packets (all except the last packet of an
  operation) of an ordered RMA operation will have RO set.

## Translation

The CXI provider supports two translation mechanisms: Address Translation
Services (ATS) and NIC Translation Agent (NTA). Use the environment variable
*FI_CXI_ATS* to select between translation mechanisms.

ATS refers to NIC support for PCIe rev. 4 ATS, PRI and PASID features. ATS
enables the NIC to efficiently access the entire virtual address space of a
process. ATS mode currently supports AMD hosts using the iommu_v2 API.

The NTA is an on-NIC translation unit. The NTA supports two-level page tables
and additional hugepage sizes. Most CPUs support 2MB and 1GB hugepage sizes.
Other hugepage sizes may be supported by SW to enable the NIC to cache more
address space.

ATS and NTA both support on-demand paging (ODP) in the event of a page fault.
Use the environment variable *FI_CXI_ODP* to enable ODP.

With ODP enabled, buffers used for data transfers are not required to be backed
by physical memory. An un-populated buffer that is referenced by the NIC will
incur a network page fault. Network page faults will significantly impact
application performance. Clients should take care to pre-populate buffers used
for data-tranfer operations to avoid network page faults. Copy-on-write
semantics work as expected with ODP.

With ODP disabled, all buffers used for data transfers are backed by pinned
physical memory. Using Pinned mode avoids any overhead due to network page
faults but requires all buffers to be backed by physical memory. Copy-on-write
semantics are broken when using pinned memory. See the Fork section for more
information.

## Translation Cache

Mapping a buffer for use by the NIC is an expensive operation. To avoid this
penalty for each data transfer operation, the CXI provider maintains an internal
translation cache.

When using the ATS translation mode, the provider does not maintain translations
for individual buffers. It follows that translation caching is not required.

## Fork

The CXI provider supports pinned and demand-paged translation modes. When using
pinned memory, accessing an RDMA buffer from a forked child process is not
supported and may lead to undefined behavior. To avoid issues, fork safety can
be enabled by defining the environment variables CXI_FORK_SAFE and
CXI_FORK_SAFE_HP.

## GPUs

GPU support is planned.

# OPTIMIZATION

## Optimized MRs

The CXI provider has two separate MR implementations: standard and optimized.
Standard MRs are designed to support applications which require a large number
of remote memory regions. Optimized MRs are designed to support one-sided
programming models that allocate a small number of large remote memory windows.
The CXI provider can achieve higher RMA Write rates when targeting an optimized
MR.

Both types of MRs are allocated using fi_mr_reg. MRs with client-provided key in
the range [0-99] are optimized MRs. MRs with key greater or equal to 100 are
standard MRs. An application may create a mix of standard and optimized MRs. To
disable the use of optimized MRs, set environment variable
*FI_CXI_OPTIMIZED_MRS=false*. When disabled, all MR keys are available and all MRs
are implemented as standard MRs. All communicating processes must agree on the
use of optimized MRs.

## Optimized RMA

Optimized MRs are one requirement for the use of low overhead packet formats
which enable higher RMA Write rates. An RMA Write will use the low overhead
format when all the following requirements are met:

* The Write targets an optimized MR
* The target MR does not require remote completion notifications (no
  *FI_RMA_EVENT*)
* The Write does not have ordering requirements (no *FI_RMA_WAW*)

Theoretically, Cassini has resources to support 64k standard MRs or 2k optimized
MRs. Practically, the limits are much lower and depend greatly on application
behavior.

Hardware counters can be used to validate the use of the low overhead packets.
The counter C_CNTR_IXE_RX_PTL_RESTRICTED_PKT counts the number of low overhead
packets received at the target NIC. Counter C_CNTR_IXE_RX_PTL_UNRESTRICTED_PKT
counts the number of ordered RDMA packets received at the target NIC.

Message rate performance may be further optimized by avoiding target counting
events. To avoid counting events, do not bind a counter to the MR. To validate
optimal writes without target counting events, monitor the counter:
C_CNTR_LPE_PLEC_HITS.

## Unreliable AMOs

By default, all AMOs are resilient to intermittent packet loss in the network.
Cassini implements a connection-based reliability model to support reliable
execution of AMOs.

The connection-based reliability model may be disabled for AMOs in order to
increase message rate. With reliability disabled, a lost AMO packet will result
in operation failure. A failed AMO will be reported to the client in a
completion event as usual. Unreliable AMOs may be useful for applications that
can tolerate intermittent AMO failures or those where the benefit of increased
message rate outweighs by the cost of restarting after a failure.

Unreliable, non-fetching AMOs may be performed by specifying the
*FI_CXI_UNRELIABLE* flag. Unreliable, fetching AMOs are not supported. Unreliable
AMOs must target an optimized MR and cannot use remote completion notification.
Unreliable AMOs are not ordered.

## High Rate Put

High Rate Put (HRP) is a feature that increases message rate performance of RMA
and unreliable non-fetching AMO operations at the expense of global ordering
guarantees.

HRP responses are generated by the fabric egress port. Responses are coalesced
by the fabric to achieve higher message rates. The completion event for an HRP
operation guarantees delivery but does not guarantee global ordering. If global
ordering is needed following an HRP operation, the source may follow the
operation with a normal, fenced Put.

HRP RMA and unreliable AMO operations may be performed by specifying the
*FI_CXI_HRP* flag. HRP AMOs must also use the *FI_CXI_UNRELIABLE* flag. Monitor the
hardware counter C_CNTR_HNI_HRP_ACK at the initiator to validate that HRP is in
use.

## Counters

Cassini offloads light-weight counting events for certain types of operations.
The rules for offloading are:

* Counting events for RMA and AMO source events are always offloaded.
* Counting events for RMA and AMO target events are always offloaded.
* Counting events for Sends are offloaded when message size is less than the
  rendezvous threshold.
* Counting events for message Receives are never offloaded by default.

Software progress is required to update counters unless the criteria for
offloading are met.

# RUNTIME PARAMETERS

The CXI provider checks for the following environment variables:

*FI_CXI_ODP*
:   Enables on-demand paging. If disabled, all DMA buffers are pinned.

*FI_CXI_ATS*
:   Enables PCIe ATS. If disabled, the NTA mechanism is used.

*FI_CXI_ATS_MLOCK_MODE*
:   Sets ATS mlock mode. The mlock() system call may be used in conjunction
    with ATS to help avoid network page faults. Valid values are "off" and
    "all". When mlock mode is "off", the provider does not use mlock(). An
    application using ATS without mlock() may experience network page faults,
    reducing network performance. When ats_mlock_mode is set to "all", the
    provider uses mlockall() during initialization with ATS. mlockall() causes
    all mapped addresses to be locked in RAM at all times. This helps to avoid
    most network page faults. Using mlockall() may increase pressure on
    physical memory.  Ignored when ODP is disabled.

*FI_CXI_RDZV_OFFLOAD*
:   Enables offloaded rendezvous messaging protocol.

*FI_CXI_RDZV_THRESHOLD*
:   Message size threshold for rendezvous protocol.

*FI_CXI_FC_RECOVERY*
:   Enables message flow-control recovery. Message flow-control
    is triggered when hardware message matching resources become exhausted.
    Messages may be dropped and retransmitted in order to recover. This impacts
    performance significantly.

    Programs should be careful to avoid using large numbers of unmatched
    receive operations and unexpected messages to prevent message flow-control.
    To help avoid this condition, increase Overflow buffer space using
    environment variables *FI_CXI_OFLOW_\**.

    Flow control recovery is enabled by default.

*FI_CXI_RDZV_GET_MIN*
:   Minimum rendezvous Get payload size. A Send with length less than or equal
    to *FI_CXI_RDZV_THRESHOLD* plus *FI_CXI_RDZV_GET_MIN* will be performed
    using the eager protocol. Larger Sends will be performed using the
    rendezvous protocol with *FI_CXI_RDZV_THRESHOLD* bytes of payload sent
    eagerly and the remainder of the payload read from the source using a Get.
    *FI_CXI_RDZV_THRESHOLD* plus *FI_CXI_RDZV_GET_MIN* must be less than or
    equal to *FI_CXI_OFLOW_BUF_SIZE*.

*FI_CXI_RDZV_EAGER_SIZE*
:   Eager data size for rendezvous protocol.

*FI_CXI_OFLOW_BUF_SIZE*
:   Overflow buffer size.

*FI_CXI_OFLOW_BUF_COUNT*
:   Overflow buffer count.

*FI_CXI_OPTIMIZED_MRS*
:   Enables optimized memory regions.

*FI_CXI_LLRING_MODE*
:   Set the policy for use of the low-latency command queue ring mechanism.
    This mechanism improves the latency of command processing on an idle
    command queue.  Valid values are idle, always, and never.

*FI_CXI_CQ_POLICY*
:   Experimental. Set Command Queue write-back policy. Valid values are always,
    high_empty, low_empty, and low. "always", "high", and "low" refer to the
    frequency of write-backs. "empty" refers to whether a write-back is
    performed when the queue becomes empty.

*FI_CXI_DEFAULT_VNI*
:   Default VNI value (masked to 16 bits).

*FI_CXI_EQ_ACK_BATCH_SIZE*
:   Number of EQ events to process before writing an acknowledgement to HW.
    Batching ACKs amortizes the cost of event acknowledgement over multiple
    network operations.

*FI_CXI_MSG_OFFLOAD*
:   Enable or disable message matching offload. If disabled, the provider will
    perform the message matching. All incoming unmatched messages are written
    into a request buffer. The environment variables FI_CXI_REQ_BUF_SIZE and
    FI_CXI_REQ_BUF_COUNT are used to control the size and number of request
    buffers posted to handle incoming unmatched messages.

*FI_CXI_REQ_BUF_SIZE*
:   Size of request buffers. Increasing the request buffer size allows for more
    unmatched messages to be sent into a single request buffer.

*FI_CXI_REQ_BUF_COUNT*
:   Number of request buffers. Dynamically increasing and decreasing request
    buffer count is not currently supported.

*FI_CXI_FC_RETRY_USEC_DELAY*
:   Number of micro-seconds to sleep before retrying a dropped side-band, flow
    control message. Setting to zero will disable any sleep.

Note: Use the fi_info utility to query provider environment variables:
<code>fi_info -p cxi -e</code>

# CXI EXTENSIONS

The CXI provider supports various fabric-specific extensions. Extensions are
accessed using the fi_open_ops function. Currently, extensions are only
supported for CXI domains.

CXI domain extensions have been named *FI_CXI_DOM_OPS_1*. The flags parameter
is ignored. The fi_open_ops function takes a `struct fi_cxi_dom_ops`. See an
example of usage below:

```c
struct fi_cxi_dom_ops *dom_ops;

ret = fi_open_ops(&domain->fid, FI_CXI_DOM_OPS_1, 0, (void **)&dom_ops, NULL);
```

The following domain extensions are defined:

```c
struct fi_cxi_dom_ops {
	int (*cntr_read)(struct fid *fid, unsigned int cntr, uint64_t *value,
		      struct timespec *ts);
};
```

The cntr_read extension is used to read hardware counter values. Valid values
of the cntr argument are found in the Cassini-specific header file
cassini_cntr_defs.h. Note that Counter accesses by applications may be
rate-limited to 1HZ.

# FABTESTS

The CXI provider does not currently support fabtests which depend on IP
addressing.

fabtest RDM benchmarks are supported, like:

```c
# Start server by specifying source PID and interface
./fabtests/benchmarks/fi_rdm_tagged_pingpong -B 10 -s cxi0

# Read server NIC address
CXI0_ADDR=$(cat /sys/class/cxi/cxi0/device/properties/nic_addr)

# Start client by specifying server PID and NIC address
./fabtests/benchmarks/fi_rdm_tagged_pingpong -P 10 $CXI0_ADDR

# The client may be bound to a specific interface, like:
./fabtests/benchmarks/fi_rdm_tagged_pingpong -B 10 -s cxi1 -P 10 $CXI0_ADDR
```

Some functional fabtests are supported (including fi_bw). Others use IP sockets
and are not yet supported.

multinode fabtests are not yet supported.

ubertest is supported for test configs matching the provider's current
capabilities.

unit tests are supported where the test feature set matches the CXI provider's
current capabilities.

# ERRATA

* Fetch and compare type AMOs with FI_DELIVERY_COMPLETE or FI_MATCH_COMPLETE
  completion semantics are not supported.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
