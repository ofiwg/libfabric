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
*FI_MR_ENDPOINT*. *FI_MR_ALLOCATED* is required if ODP in not enabled or not
desired. Client specified 32-bit MR keys are the default unless *FI_MR_PROV_KEY*
is specified. For *FI_MR_PROV_KEY* provider generated 64-bit MR keys are used.
An RMA initiator can work concurrently with client and provider generated keys.

## Data transfer operations

The following data transfer interfaces are supported: *FI_ATOMIC*, *FI_MSG*,
*FI_RMA*, *FI_TAGGED*.  See DATA TRANSFER OPERATIONS below for more details.

## Completion events

The CXI provider supports all CQ event formats.

## Modes

The CXI provider does not require any operation modes.

## Progress

The CXI provider currently supports *FI_PROGRESS_MANUAL* data and control
progress modes.

## Multi-threading

The CXI provider supports FI_THREAD_SAFE and FI_THREAD_DOMAIN threading models.

## Wait Objects

The CXI provider supports FI_WAIT_FD and FI_WAIT_POLLFD CQ wait object types.
FI_WAIT_UNSPEC will default to FI_WAIT_FD. However FI_WAIT_NONE should achieve
the lowest latency and reduce interrupt overhead.

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
Traffic Classes which a libfabric user can access.

Two endpoints must use the same VNI in order to communicate. Generally, a
parallel application should be assigned to a unique VNI on the fabric in order
to achieve network traffic and address isolation. Typically a privileged
entity, like a job launcher, will allocate one or more VNIs for use by the
libfabric user.

The CXI service API is provided by libCXI. It enables a privileged entity, like
an application launcher, to control an unprivileged process's access to NIC
resources. Generally, a parallel application should be assigned to a unique CXI
service in order to control access to local resources, VNIs, and Traffic
Classes.

While a libfabric user provided authorization key is optional, it is highly
encouraged that libfabric users provide an authorization key through the domain
attribute hints during `fi_getinfo()`. How libfabric users acquire the
authorization key may vary between the users and is outside the scope of this
document.

If an authorization key is not provided by the libfabric user, the CXI provider
will attempt to generate an authorization key on behalf of the user. The
following outlines how the CXI provider will attempt to generate an
authorization key.

1. Query for the following environment variables and generate an authorization
key using them.
    * *SLINGSHOT_VNIS*: Comma separated list of VNIs. The CXI provider will only
    use the first VNI if multiple are provide. Example: `SLINGSHOT_VNIS=234`.
    * *SLINGSHOT_DEVICES*: Comma separated list of device names. Each device index
    will use the same index to lookup the service ID in *SLINGSHOT_SVC_IDS*.
    Example: `SLINGSHOT_DEVICES=cxi0,cxi1`.
    * *SLINGSHOT_SVC_IDS*: Comma separated list of pre-configured CXI service IDs.
    Each service ID index will use the same index to lookup the CXI device in
    *SLINGSHOT_DEVICES*. Example: `SLINGSHOT_SVC_IDS=5,6`.

    **Note:** How valid VNIs and device services are configured is outside
    the responsibility of the CXI provider.

2. Query pre-configured device services and find first entry with same UID as
the libfabric user.

3. Query pre-configured device services and find first entry with same GID as
the libfabric user.

4. Query pre-configured device services and find first entry which does not
restrict member access. If enabled, the default service is an example of an
unrestricted service.

    **Note:** There is a security concern with such services since it allows
    for multiple independent libfabric users to use the same service.

**Note:** For above entries 2-4, it is possible the found device service does
not restrict VNI access. For such cases, the CXI provider will query
*FI_CXI_DEFAULT_VNI* to assign a VNI.

During Domain allocation, if the domain auth_key attribute is NULL, the CXI
provider will attempt to generate a valid authorization key. If the domain
auth_key attribute is valid (i.e. not NULL and encoded authorization key has
been verified), the CXI provider will use the encoded VNI and service ID.
Failure to generate a valid authorization key will result in Domain allocation
failure.

During Endpoint allocation, if the endpoint auth_key attribute is NULL, the
Endpoint with inherit the parent Domain's VNI and service ID. If the Endpoint
auth_key attribute is valid, the encoded VNI and service ID must match the
parent Domain's VNI and service ID. Allocating an Endpoint with a different VNI
and service from the parent Domain is not supported.

The following is the expected parallel application launch workflow with
CXI integrated launcher and CXI authorization key aware libfabric user:

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

Scalable Endpoints (SEPs) support is not enabled in the CXI provider. Future
releases of the provider will re-introduce SEP support.

## Messaging

The CXI provider supports both tagged (*FI_TAGGED*) and untagged (*FI_MSG*)
two-sided messaging interfaces. In the normal case, message matching is
performed by hardware. In certain low resource conditions, the responsibility to
perform message matching may be transferred to software. Specification
of the receive message matching mode in the environment (*FI_CXI_RX_MATCH_MODE*)
controls the initial matching mode and whether hardware matching can
transparently transition matching to software where a hybrid of hardware
and software receive matching is done.

If a Send operation arrives at a node where there is no matching Receive
operation posted, it is considered unexpected. Unexpected messages are
supported. The provider manages buffers to hold unexpected message data.

Unexpected message handling is transparent to clients. Despite that, clients
should take care to avoid excessive use of unexpected messages by pre-posting
Receive operations. An unexpected message ties up hardware and memory resources
until it is matched with a user buffer.

The CXI provider implements several message protocols internally. A message
protocol is selected based on payload length. Short messages are transferred
using the eager protocol. In the eager protocol, the entire message payload is
sent along with the message header. If an eager message arrives unexpectedly,
the entire message is buffered at the target until it is matched to a Receive
operation.

Long messages are transferred using a rendezvous protocol. The threshold at
which the rendezvous protocol is used is controlled with the
*FI_CXI_RDZV_THRESHOLD* and *FI_CXI_RDZV_GET_MIN* environment variables.

In the rendezvous protocol, a portion of the message payload is sent
along with the message header. Once the header is matched to a Receive
operation, the remainder of the payload is pulled from the source using an RDMA
Get operation. If the message arrives unexpectedly, the eager portion of the
payload is buffered at the target until it is matched to a Receive operation.
In the normal case, the Get is performed by hardware and the operation
completes without software progress.

Unexpected rendezvous protocol messages can not complete and release source side
buffer resources until a matching receive is posted at the destination and the
non-eager data is read from the source with a rendezvous get DMA. The number of
rendezvous messages that may be outstanding is limited by the minimum of the
hints->tx_attr->size value specified and the number of rendezvous operation ID
mappings available. FI_TAGGED rendezvous messages have 32K-256 ID mappings,
FI_MSG rendezvous messages are limited to 256 ID mappings. While this
works well with MPI, care should be taken that this minimum is large enough to
ensure applications written in a manner that assumes unlimited resources and
use FI_MSG rendezvous messaging do not induce a software deadlock. If FI_MSG
rendezvous messaging is done in a unexpected manner that may exceed the FI_MSG
ID mappings available, it may be sufficient to reduce the number of rendezvous
operations by increasing the rendezvous threshold. See *FI_CXI_RDZV_THRESHOLD*
for information.

Message flow-control is triggered when hardware message matching resources
become exhausted. Messages may be dropped and retransmitted in order to
recover; impacting performance significantly. Programs should be careful to avoid
posting large numbers of unmatched receive operations and to minimize the
number of outstanding unexpected messages to prevent message flow-control.
If the RX message matching mode is configured to support hybrid mode, when
resources are exhausted, hardware will transition to hybrid operation where
hardware and software share matching responsibility.

To help avoid this condition, increase Overflow buffer space using environment
variables *FI_CXI_OFLOW_\**, and for software and hybrid RX match modes
increase Request buffer space using the variables *FI_CXI_REQ_\**.

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

## Heterogenous Memory (HMEM) Supported Interfaces

The CXI provider supports the following OFI iface types: FI_HMEM_CUDA, FI_HMEM_ROCR, and FI_HMEM_ZE.

### FI_HMEM_ZE Limitations

The CXI provider only supports GPU direct RDMA with ZE device buffers if implicit scaling
is disabled. The following ZE environment variables disable implicit scaling:
EnableImplicitScaling=0 NEOReadDebugKeys=1.

For testing purposes only, the implicit scaling check can be disabled by setting the
following environment variable: FI_CXI_FORCE_ZE_HMEM_SUPPORT=1. This may need to be
combined with the following environment variable to get CXI provider memory registration
to work: FI_CXI_DISABLE_HMEM_DEV_REGISTER=1.

## Collectives (accelerated)

The CXI provider supports a limited set of collective operations specifically
intended to support use of the hardware-accelerated reduction features of the
CXI-supported NIC and fabric hardware.

These features are implemented using the (experimental) OFI collectives API. The
implementation supports the following collective functions:

* **fi_query_collective**()
* **fi_join_collective**()
* **fi_barrier**()
* **fi_broadcast**()
* **fi_reduce**()
* **fi_allreduce**()

### **fi_query_collective**()

Standard implementation that exposes the features described below.

### **fi_join_collective**()

The **fi_join_collective**() implementation is provider-managed. However, the
*coll_addr* parameter is not useful to the implementation, and must be
specified as FI_ADDR_NOTAVAIL. The *set* parameter must contain fi_addr_t
values that resolve to meaningful CXI addresses in the endpoint *fi_av*
structure. **fi_join_collective**() must be called for every address in the
*set* list, and must be progressed until the join operation is complete. There
is no inherent limit on join concurrency.

The join will create a multicast tree in the fabric to manage the collective
operations. This operation requires access to a secure Fabric Manager REST API
that constructs this tree, so any application that attempts to use accelerated
collectives will bind to libcurl and associated security libraries, which must
be available on the system.

There are hard limits to the number of multicast addresses available on a
system, and administrators may impose additional limits on the number of
multicast addresses available to any given collective job.

### fi_reduction operations

Payloads are limited to 32-byte data structures, and because they all use the
same underlying hardware model, they are all synchronizing calls. Specifically,
the supported functions are all variants of fi_allreduce().

* **fi_barrier** is **fi_allreduce** using an optimized no-data operator.
* **fi_broadcast** is **fi_allreduce** using FI_BOR, with data forced to zero for all but the root rank.
* **fi_reduce** is **fi_allreduce** with a result pointer ignored by all but the root rank.

All functions must be progressed to completion on all ranks participating in
the collective group. There is a hard limit of eight concurrent reductions on
each collective group, and attempts to launch more operations will return
<nobr>-FI_EAGAIN.</nobr>

**allreduce** supports the following hardware-accelerated reduction operators:

| Operator | Supported Datatypes |
| -------- | --------- |
| FI_BOR   | FI_UINT8, FI_UINT16, FI_UINT32, FI_UINT64 |
| FI_BAND  | FI_UINT8, FI_UINT16, FI_UINT32, FI_UINT64 |
| FI_BXOR  | FI_UINT8, FI_UINT16, FI_UINT32, FI_UINT64 |
| FI_MIN   | FI_INT64, FI_DOUBLE |
| FI_MAX   | FI_INT64, FI_DOUBLE |
| FI_SUM   | FI_INT64, FI_DOUBLE |
| FI_CXI_MINMAXLOC      | FI_INT64, FI_DOUBLE |
| FI_CXI_MINNUM         | FI_DOUBLE |
| FI_CXI_MAXNUM         | FI_DOUBLE |
| FI_CXI_MINMAXNUMLOC   | FI_DOUBLE |
| FI_CXI_REPSUM         | FI_DOUBLE |

Data space is limited to 32 bytes in all cases except REPSUM, which supports
only a single FI_DOUBLE.

Only unsigned bitwise operators are supported.

Only signed integer arithmetic operations are are supported.

The MINMAXLOC operators are a mixed data representation consisting of two
values, and two indices. Each rank reports its minimum value and rank index,
and its maximum value and rank index. The collective result is the global
minimum value and rank index, and the global maximum value and rank index. Data
structures for these functions can be found int the fi_cxi_ext.h file. The
*datatype* should represent the type of the minimum/maximum values, and the
*count* must be 1.

The double-precision operators provide an associative (NUM) variant for MIN,
MAX, and MINMAXLOC. Default IEEE behavior is to treat any operation with NaN as
invalid, including comparison, which has the interesting property of causing:

    MIN(NaN, value) => NaN
    MAX(NaN, value) => NaN

This means that if NaN creeps into a MIN/MAX reduction in any rank, it tends to
poison the entire result. The associative variants instead effectively ignore
the NaN, such that:

    MIN(NaN, value) => value
    MAX(NaN, value) => value

The REPSUM operator implements a reproducible (associative) sum of
double-precision values. The payload can accommodate only a single
double-precision value per reduction, so *count* must be 1.

See: [Berkeley reproducible sum algorithm](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.pdf)
https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.pdf

### double precision rounding

C99 defines four rounding modes for double-precision SUM, and some systems may
support a "flush-to-zero" mode for each of these, resulting in a total of eight
different modes for double-precision sum.

The fabric hardware supports all eight modes transparently.

Although the rounding modes have thread scope, all threads, processes, and
nodes should use the same rounding mode for any single reduction.

### reduction flags

The reduction operations supports two flags:

* **FI_MORE**
* **FI_CXI_PRE_REDUCED** (overloads **FI_SOURCE**)

The **FI_MORE** flag advises that the *result* data pointer represents an
opaque, local reduction accumulator, and will be used as the destination of the
reduction. This operation can be repeated any number of times to accumulate
results locally, and spans the full set of all supported reduction operators.
The *op*, *count*, and *datatype* values must be consistent for all calls. The
operation ignores all global or static variables &mdash; it can be treated as a
*pure* function call &mdash; and returns immediately. The caller is responsible
for protecting the accumulator memory if it is used by multiple threads or
processes on a compute node.

If **FI_MORE** is omitted, the destination is the fabric, and this will
initiate a fabric reduction through the associated endpoint. The reduction must
be progressed, and upon successful completion, the *result* data pointer will
be filled with the final reduction result of *count* elements of type
*datatype*.

The **FI_CXI_PRE_REDUCED** flag advises that the source data pointer represents
an opaque reduction accumulator containing pre-reduced data. The *count* and
*datatype* arguments are ignored.

if **FI_CXI_PRE_REDUCED** is omitted, the source is taken to be user data with
*count* elements of type *datatype*.

The opaque reduction accumulator is exposed as **struct cxip_coll_accumulator**
in the fi_cxi_ext.h file.

**Note**: The opaque reduction accumulator provides extra space for the
expanded form of the reproducible sum, which carries the extra data required to
make the operation reproducible in software.

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

When FI_MR_PROV_KEY mr_mode is specified caching of remote access MRs is enabled,
which can improve registration/de-registration performance in RPC type applications,
that wrap RMA operations within a message RPC protocol. Optimized MRs will be
preferred, but will fallback to standard MRs if insufficient hardware resources are
available.

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

*FI_CXI_RDZV_THRESHOLD*
:   Message size threshold for rendezvous protocol.

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

*FI_CXI_DISABLE_NON_INJECT_MSG_IDC*
:   Experimental option to disable favoring IDC for transmit of small messages
    when FI_INJECT is not specified. This can be useful with GPU source buffers
    to avoid the host copy in cases a performant copy can not be used. The default
    is to use IDC for all messages less than IDC size.

*FI_CXI_DISABLE_HOST_REGISTER*
:   Disable registration of host buffers (overflow and request) with GPU. There
    are scenarios where using a large number of processes per GPU results in page
    locking excessive amounts of memory degrading performance and/or restricting
    process counts. The default is to register buffers with the GPU.

*FI_CXI_OFLOW_BUF_SIZE*
:   Size of overflow buffers. Increasing the overflow buffer size allows for
    more unexpected message eager data to be held in single overflow buffer.
    The default size is 2MB.

*FI_CXI_OFLOW_BUF_MIN_POSTED/FI_CXI_OFLOW_BUF_COUNT*
:   The minimum number of overflow buffers that should be posted. The default
    minimum posted count is 3. Buffers will grow unbounded to support
    outstanding unexpected messages. Care should be taken to size appropriately
    based on job scale, size of eager data, and the amount of unexpected
    message traffic to reduce the need for flow control.

*FI_CXI_OFLOW_BUF_MAX_CACHED*
:   The maximum number of overflow buffers that will be cached. The default
    maximum count is 3 * FI_CXI_OFLOW_BUF_MIN_POSTED. A value of zero indicates
    that once a overflow buffer is allocated it will be cached and used as
    needed. A non-zero value can be used with bursty traffic to shrink the
    number of allocated buffers to the maximum count when they are no longer
    needed.

*FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD
:   Defines the maximum CPU memcpy size for HMEM device memory that is
    accessible by the CPU with load/store operations.

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
:   Default VNI value used only for service IDs where the VNI is not restricted.

*FI_CXI_EQ_ACK_BATCH_SIZE*
:   Number of EQ events to process before writing an acknowledgement to HW.
    Batching ACKs amortizes the cost of event acknowledgement over multiple
    network operations.

*FI_CXI_RX_MATCH_MODE*
:   Specify the receive message matching mode to be utilized.
    *FI_CXI_RX_MATCH_MODE=*hardware | software | hybrid

    *hardware* - Message matching is fully offloaded, if resources become
    exhausted flow control will be performed and existing unexpected message
    headers will be onloaded to free resources.

    *software* - Message matching is fully onloaded.

    *hybrid* - Message matching begins fully offloaded, if resources become
    exhuasted hardware will transition message matching to a hybrid of
    hardware and software matching.

    For both *"hybrid"* and *"software"* modes and care should be taken to
    minimize the threshold for rendezvous processing
    (i.e. *FI_CXI_RDZV_THRESHOLD* + *FI_CXI_RDZV_GET_MIN*). When running in
    software endpoint mode the environment variables *FI_CXI_REQ_BUF_SIZE*
    and *FI_CXI_REQ_BUF_MIN_POSTED* are used to control the size and number
    of the eager request buffers posted to handle incoming unmatched messages.

*FI_CXI_HYBRID_PREEMPTIVE*
:   When in hybrid mode, this variable can be used to enable preemptive
    transitions to software matching. This is useful at scale for poorly
    written applications with a large number of unexpected messages
    where reserved resources may be insufficient to prevent to prevent
    starvation of software request list match entries. Default is 0, disabled.

*FI_CXI_HYBRID_RECV_PREEMPTIVE*
:   When in hybrid mode, this variable can be used to enable preemptive
    transitions to software matching. This is useful at scale for poorly
    written applications with a large number of unmatched posted receives
    where reserved resources may be insufficient to prevent starvation of
    software request list match entries. Default is 0, disabled.

*FI_CXI_REQ_BUF_SIZE*
:   Size of request buffers. Increasing the request buffer size allows for more
    unmatched messages to be sent into a single request buffer. The default
    size is 2MB.

*FI_CXI_REQ_BUF_MIN_POSTED*
:   The minimum number of request buffers that should be posted. The default
    minimum posted count is 4. The number of buffers will grow unbounded to
    support outstanding unexpected messages. Care should be taken to size
    appropriately based on job scale and the size of eager data to reduce
    the need for flow control.

*FI_CXI_REQ_BUF_MAX_CACHED/FI_CXI_REQ_BUF_MAX_COUNT*
:   The maximum number of request buffers that will be cached. The default
    maximum count is 0. A value of zero indicates that once a request buffer
    is allocated it will be cached and used as needed. A non-zero value can
    be used with bursty traffic to shrink the number of allocated buffers to
    a maximum count when they are no longer needed.

*FI_CXI_MSG_LOSSLESS*
:   Enable or disable lossless receive matching. If hardware resources are
    exhausted, hardware will pause the associated traffic class until a
    overflow buffer (hardware match mode) or request buffer (software match
    mode or hybrid match mode) is posted. This is considered experimental and
    defaults to disabled.

*FI_CXI_FC_RETRY_USEC_DELAY*
:   Number of micro-seconds to sleep before retrying a dropped side-band, flow
    control message. Setting to zero will disable any sleep.

*FI_UNIVERSE_SIZE*
:   Defines the maximum number of processes that will be used by distribute
    OFI application. Note that this value is used in setting the default
    control EQ size, see FI_CXI_CTRL_RX_EQ_MAX_SIZE.

*FI_CXI_CTRL_RX_EQ_MAX_SIZE*
:   Max size of the receive event queue used for side-band/control messages.
    Default receive event queue size is based on FI_UNIVERSE_SIZE. Increasing the
    receive event queue size can help prevent side-band/control messages from
    being dropped and retried but at the cost of additional memory usage. Size is
    always aligned up to a 4KiB boundary.

*FI_CXI_DEFAULT_CQ_SIZE*
:   Change the provider default completion queue size expressed in entries. This
    may be useful for applications which rely on middleware, and middleware defaults
    the completion queue size to the provider default. To avoid flow-control due
    to the associated event queue being full, care should be taken to adequately
    progress the CQ and to size it appropriately. Note that unexpected messages
    hold reservations against the queue and reduce the amount of space available
    at any given time. The sizing is application specific and based on job scale,
    but should minimally meet the following:

    entries = Num Receives Posted + Num Unexpected Messages * 6 + Num Sends Posted +
              FI_CXI_EQ_ACK_BATCH_SIZE + Num Overflow Buffers + Num Request Buffers

    For instance if 1K of each RX, Unexpected, TX messages can be outstanding, then
    the size should be set to at least something > 8K. See FI_CXI_CQ_FILL_PERCENT
    which can be set to change the percentage at which the CQ should indicate that
    it has become saturated and force pushback to the application to ensure progress.

*FI_CXI_DISABLE_EQ_HUGETLB/FI_CXI_DISABLE_CQ_HUGETLB*
:   By default, the provider will attempt to allocate 2 MiB hugetlb pages for
    provider event queues. Disabling hugetlb support will cause the provider
    to fallback to memory allocators using host page sizes.
    FI_CXI_DISABLE_EQ_HUGETLB replaces FI_CXI_DISABLE_CQ_HUGETLB, however use
    of either is still supported.

*FI_CXI_DEFAULT_TX_SIZE*
:   Set the default tx_attr.size field to be used by the provider if the size
    in not specified in the user provided fi_info hints.

*FI_CXI_SW_RX_TX_INIT_MAX*
:   Debug control to override the number of TX operations that can be
    outstanding that are initiated by software RX processing. It has no impact
    on hardware initiated RX rendezvous gets.

*FI_CXI_DEVICE_NAME*
:   Restrict CXI provider to specific CXI devices. Format is a comma separated
    list of CXI devices (e.g. cxi0,cxi1).

*FI_CXI_TELEMETRY*
:   Perform a telemetry delta between fi_domain open and close. Format is a
    comma separated list of telemetry files as defined in
    /sys/class/cxi/cxi*/device/telemetry/. The ALL-in-binary file in this
    directory is invalid. Note that these are per CXI interface counters and not
    per CXI process per interface counters.

*FI_CXI_TELEMETRY_RGID*
:   Resource group ID (RGID) to restrict the telemetry collection to. Value less
    than 0 is no restrictions.

*FI_CXI_CQ_FILL_PERCENT*
:   Fill percent of underlying hardware event queue used to determine when
    completion queue is saturated. A saturated completion queue results in the
    provider returning -FI_EAGAIN for data transfer and other related libfabric
    operations.

*FI_CXI_COMPAT*
:   Temporary compatibility to allow use of pre-upstream values for FI_ADDR_CXI and
    FI_PROTO_CXI. Compatibility can be disabled to verify operation with upstream
    constant values and to enable access to conflicting provider values. The default
    setting of 1 specifies both old and new constants are supported. A setting of 0
    disables support for old constants and can be used to test that an application is
    compatible with the upstream values. A setting of 2 is a safety fallback that if
    used the provider will only export fi_info with old constants and will be incompatible
    with libfabric clients that been recompiled.

*FI_CXI_COLL_FABRIC_MGR_URL*
:   **accelerated collectives:** Specify the HTTPS address of the fabric manager REST API
    used to create specialized multicast trees for accelerated collectives. This parameter
    is **REQUIRED** for accelerated collectives, and is a fixed, system-dependent value.

*FI_CXI_COLL_TIMEOUT_USEC*
:   **accelerated collectives:** Specify the reduction engine timeout. This should be
    larger than the maximum expected compute cycle in repeated reductions, or acceleration
    can create incast congestion in the switches. The relative performance benefit of
    acceleration declines with increasing compute cycle time, dropping below one percent at
    32 msec (32000). Using acceleration with compute cycles larger than 32 msec is not
    recommended except for experimental purposes. Default is 32 msec (32000), maximum is
    20 sec (20000000).

*FI_CXI_COLL_USE_DMA_PUT*
:   **accelerated collectives:** Use DMA for collective packet put. This uses DMA to
    inject reduction packets rather than IDC, and is considered experimental. Default
    is false.

*FI_CXI_DISABLE_HMEM_DEV_REGISTER*
:   Disable registering HMEM device buffer for load/store access. Some HMEM devices
    (e.g. AMD, Nvidia, and Intel GPUs) support backing the device memory by the PCIe BAR.
    This enables software to perform load/stores to the device memory via the BAR instead
    of using device DMA engines. Direct load/store access may improve performance.

*FI_CXI_FORCE_ZE_HMEM_SUPPORT*
:   Force the enablement of ZE HMEM support. By default, the CXI provider will only
    support ZE memory registration if implicit scaling is disabled (i.e. the environment
    variables EnableImplicitScaling=0 NEOReadDebugKeys=1 are set). Set
    FI_CXI_FORCE_ZE_HMEM_SUPPORT to 1 will cause the CXI provider to skip the implicit
    scaling checks. GPU direct RDMA may or may not work in this case.

Note: Use the fi_info utility to query provider environment variables:
<code>fi_info -p cxi -e</code>

# CXI EXTENSIONS

The CXI provider supports various fabric-specific extensions. Extensions are
accessed using the fi_open_ops function.

## CXI Domain Extensions

CXI domain extensions have been named *FI_CXI_DOM_OPS_4*. The flags parameter
is ignored. The fi_open_ops function takes a `struct fi_cxi_dom_ops`. See an
example of usage below:

```c
struct fi_cxi_dom_ops *dom_ops;

ret = fi_open_ops(&domain->fid, FI_CXI_DOM_OPS_4, 0, (void **)&dom_ops, NULL);
```

The following domain extensions are defined:

```c
struct fi_cxi_dom_ops {
	int (*cntr_read)(struct fid *fid, unsigned int cntr, uint64_t *value,
		      struct timespec *ts);
	int (*topology)(struct fid *fid, unsigned int *group_id,
	              unsigned int *switch_id, unsigned int *port_id);
	int (*enable_hybrid_mr_desc)(struct fid *fid, bool enable);
	size_t (*ep_get_unexp_msgs)(struct fid_ep *fid_ep,
                                    struct fi_cq_tagged_entry *entry,
                                    size_t count, fi_addr_t *src_addr,
                                    size_t *ux_count);
};
```

The cntr_read extension is used to read hardware counter values. Valid values
of the cntr argument are found in the Cassini-specific header file
cassini_cntr_defs.h. Note that Counter accesses by applications may be
rate-limited to 1HZ.

The topology extension is used to return CXI NIC address topology information
for the domain. Currently only a dragonfly fabric topology is reported.

The enablement of hybrid MR descriptor mode allows for libfabric users
to optionally pass in a valid MR desc for local communications operations.

The get unexpected message function is used to obtain a list of
unexpected messages associated with an endpoint. The list is returned
as an array of CQ tagged entries set in the following manner:

```
struct fi_cq_tagged_entry {
	.op_context = NULL,
	.flags = any of [FI_TAGGED | FI_MSG | FI_REMOTE_CQ_DATA],
	.len = message length,
	.buf = NULL,
	.data = CQ data if FI_REMOTE_CQ_DATA set
	.tag = tag if FI_TAGGED set
};
```

If the src_addr or entry array is NULL, only the ux_count of
available unexpected list entries will be returned. The parameter
count specifies the size of the array provided, if it is 0 then only
the ux_count will be returned. The function returns the number of
entries written to the array or a negative errno. On successful return,
ux_count will always be set to the total number of unexpected messages available.


## CXI Counter Extensions

CXI counter extensions have been named *FI_CXI_COUNTER_OPS*. The flags parameter
is ignored. The fi_open_ops function takes a `struct fi_cxi_cntr_ops`. See an
example of usage below.

```c
struct fi_cxi_cntr_ops *cntr_ops;

ret = fi_open_ops(&cntr->fid, FI_CXI_COUNTER_OPS, 0, (void **)&cntr_ops, NULL);
```

The following domain extensions are defined:

```c
struct fi_cxi_cntr_ops {
	/* Set the counter writeback address to a client provided address. */
	int (*set_wb_buffer)(struct fid *fid, const void *buf, size_t len);

	/* Get the counter MMIO region. */
	int (*get_mmio_addr)(struct fid *fid, void **addr, size_t *len);
};
```

## CXI Counter Writeback Flag

If a client is using the CXI counter extensions to define a counter writeback
buffer, the CXI provider will not update the writeback buffer success or
failure values for each hardware counter success or failure update. This can
especially create issues when clients expect the completion of a deferred
workqueue operation to generate a counter writeback. To support this, the flag
*FI_CXI_CNTR_WB* can be used in conjunction with a deferred workqueue operation
to force a writeback at the completion of the deferred workqueue operation. See
an example of usage below.

```c
struct fi_op_rma rma = {
  /* Signal to the provider the completion of the RMA should trigger a
   * writeback.
   */
  .flags = FI_CXI_CNTR_WB,
};

struct fi_deferred_work rma_work = {
  .op_type = FI_OP_READ,
  .triggering_counter = cntr,
  .completion_cntr = cntr,
  .threshold = 1,
  .op.rma = &rma,
};

ret = fi_control(&domain->fid, FI_QUEUE_WORK, &rma_work);
```

**Note:** Using *FI_CXI_CNTR_WB* will lead to additional hardware usage. To
conserve hardware resources, it is recommended to only use the *FI_CXI_CNTR_WB*
when a counter writeback is absolutely required.

## CXI Alias EP Overrides

A transmit alias endpoint can be created and configured to utilize
a different traffic class than the original endpoint. This provides a
lightweight mechanism to utilize multiple traffic classes within a process.
Message order between the original endpoint and the alias endpoint is
not defined/guaranteed. See example usage below for setting the traffic
class of a transmit alias endpoint.

```c
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cxi_ext.h>     // Ultimately fi_ext.h

struct fid_ep *ep;
. . .

struct fid_ep *alias_ep = NULL;
uint32_t tclass = FI_TC_LOW_LATENCY;
uint64_t op_flags = FI_TRANSMIT | desired data operation flags;

ret = fi_ep_alias(ep, &alias_ep, op_flags);
if (ret)
    error;

ret = fi_set_val(&alias_ep->fid, FI_OPT_CXI_SET_TCLASS, (void *)&tlcass);
if (ret)
    error;
```

In addition, the alias endpoint message order may be modified to override
the default endpoint message order. Message order between the modified
alias endpoint and the original endpoint is not guaranteed. See example
usage below for setting the traffic class of a transmit alias endpoint.

```c
uint64_t msg_order = FI_ORDER_RMA_WAW;

ret = fi_set_val(&alias_ep->fid, FI_OPT_CXI_SET_MSG_ORDER,
                 (void *)&msg_order);
if (ret)
    error;
```

When an endpoint does not support FI_FENCE (e.g. optimized MR), a provider
specific transmit flag, FI_CXI_WEAK_FENCE, may be specified on an alias EP
to issue a FENCE operation to create a data ordering point for the alias.
This is supported for one-sided operations only.

Alias EP must be closed prior to closing the original EP.

## PCIe Atomics
The CXI provider has the ability to issue a given libfabric atomic memory
operation as a PCIe operation as compared to a NIC operation. The CXI
provider extension flag FI_CXI_PCIE_AMO is used to signify this.

Since not all libfabric atomic memory operations can be executed as a PCIe
atomic memory operation, `fi_query_atomic()` could be used to query if a
given libfabric atomic memory operation could be executed as PCIe atomic
memory operation.

The following is a query to see if a given libfabric operation can be a
PCIe atomic operation.
```c
int ret;
struct fi_atomic_attr out_attrs;

/* Query if non-fetching PCIe atomic is supported. */
ret = fi_query_atomic(domain, FI_UINT32, FI_SUM, &out_attrs, FI_CXI_PCIE_AMO);

/* Query if fetching PCIe atomic is supported. */
ret = fi_query_atomic(domain, FI_UINT32, FI_SUM, &out_attrs,
                      FI_FETCH_ATOMIC | FI_CXI_PCIE_AMO);
```

The following is how to issue a PCIe atomic operation.
```c
ssize_t ret;
struct fi_msg_atomic msg;
struct fi_ioc resultv;
void *result_desc;
size_t result_count;

ret = fi_fetch_atomicmsg(ep, &msg, &resultv, &result_desc, result_count,
                         FI_CXI_PCIE_AMO);
```

**Note:** The CXI provider only supports PCIe fetch add for UINT32_T, INT32_t,
UINT64_T, and INT64_t. This support requires enablement of PCIe fetch add in
the CXI driver, and it comes at the cost of losing NIC atomic support for another
libfabric atomic operation.

**Note:** Ordering between PCIe atomic operations and NIC atomic/RMA operations is
undefined.

To enable PCIe fetch add for libfabric, the following CXI driver kernel module
parameter must be set to non-zero.

```
/sys/module/cxi_core/parameters/amo_remap_to_pcie_fadd
```

The following are the possible values for this kernel module and the impact of
each value:
- -1: Disable PCIe fetch add support. FI_CXI_PCIE_AMO is not supported.
- 0: Enable PCIe fetch add support. FI_MIN is not supported.
- 1: Enable PCIe fetch add support. FI_MAX is not supported.
- 2: Enable PCIe fetch add support. FI_SUM is not supported.
- 4: Enable PCIe fetch add support. FI_LOR is not supported.
- 5: Enable PCIe fetch add support. FI_LAND is not supported.
- 6: Enable PCIe fetch add support. FI_BOR is not supported.
- 7: Enable PCIe fetch add support. FI_BAND is not supported.
- 8: Enable PCIe fetch add support. FI_LXOR is not supported.
- 9: Enable PCIe fetch add support. FI_BXOR is not supported.
- 10: Enable PCIe fetch add support. No loss of default CXI provider AMO
functionality.

Guidance is to default amo_remap_to_pcie_fadd to 10.

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
  completion semantics are not supported with FI_RMA_EVENT.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
