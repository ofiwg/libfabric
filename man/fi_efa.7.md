---
layout: page
title: fi_efa(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_efa \- The Amazon Elastic Fabric Adapter (EFA) Provider

# OVERVIEW

The EFA provider supports the Elastic Fabric Adapter (EFA) device on
Amazon EC2.  EFA provides reliable and unreliable datagram send/receive
with direct hardware access from userspace (OS bypass). For reliable
datagram (RDM) EP type, it supports two fabric names: `efa` and `efa-direct`.
The `efa` fabric implements a set of
[wire protocols](https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md)
to support more capabilities and features beyond the EFA device capabilities.
The `efa-direct` fabric, on the contrary, offloads all libfabric data plane calls to
the device directly without wire protocols. Compared to the `efa` fabric, the `efa-direct`
fabric supports fewer capabilities and has more mode requirements for applications.
But it provides a fast path to hand off application requests to the device.
To use `efa-direct`, set the name field in `fi_fabric_attr` to `efa-direct`.
More details and difference between the two fabrics will be presented below.


# SUPPORTED FEATURES

The following features are supported:

*Endpoint types*
: The provider supports endpoint type *FI_EP_DGRAM*, and *FI_EP_RDM* on a new
  Scalable (unordered) Reliable Datagram protocol (SRD). SRD provides support
  for reliable datagrams and more complete error handling than typically seen
  with other Reliable Datagram (RD) implementations.

*RDM Endpoint capabilities*
: 
  For the `efa` fabric, the following data transfer interfaces are supported:
  *FI_MSG*, *FI_TAGGED*, *FI_SEND*, *FI_RECV*,  *FI_RMA*, *FI_WRITE*, *FI_READ*,
  *FI_ATOMIC*, *FI_DIRECTED_RECV*, *FI_MULTI_RECV*, and *FI_SOURCE*.
  It provides SAS guarantees for data operations, and
  does not have a maximum message size (for all operations).
  For the `efa-direct` fabric, it supports *FI_MSG*, *FI_SEND*, *FI_RECV*, *FI_RMA*,
  *FI_WRITE*, *FI_READ*, and *FI_SOURCE*. As mentioned earlier, it doesn't provide
  SAS guarantees, and has different maximum message sizes for different operations.
  For MSG operations, the maximum message size is the MTU size of the efa device
  (approximately 8KiB). For RMA operations, the maximum message size is the maximum
  RDMA size of the EFA device. The exact values of these sizes can be queried by the
  `fi_getopt` API with option names `FI_OPT_MAX_MSG_SIZE` and `FI_OPT_MAX_RMA_SIZE`


*DGRAM Endpoint capabilities*
: The DGRAM endpoint only supports *FI_MSG* capability with a maximum
  message size of the MTU of the underlying hardware (approximately 8 KiB).

*Address vectors*
: The provider supports *FI_AV_TABLE*. *FI_AV_MAP* was deprecated in Libfabric 2.x.
  Applications can still use *FI_AV_MAP* to create an address vector. But the EFA
  provider implementation will print a warning and switch to *FI_AV_TABLE*.
  *FI_EVENT* is unsupported.

*Completion events*
: The provider supports *FI_CQ_FORMAT_CONTEXT*, *FI_CQ_FORMAT_MSG*, and
  *FI_CQ_FORMAT_DATA*. *FI_CQ_FORMAT_TAGGED* is supported on the `efa` fabric
  of RDM endpoint. Wait objects are not currently supported.

*Modes*
: The provider requires the use of *FI_MSG_PREFIX* when running over
  the DGRAM endpoint. And it requires the use of *FI_CONTEXT2* mode
  for DGRAM endpoint and the `efa-direct` fabric of RDM endpoint.
  The `efa` fabric of RDM endpoint doesn't have these requirements.

*Memory registration modes*
: The `efa` fabric of RDM endpoint does not require memory registration for send and receive
  operations, i.e. it does not require *FI_MR_LOCAL*. Applications may specify
  *FI_MR_LOCAL* in the MR mode flags in order to use descriptors provided by the
  application. The `efa-direct` fabric of *FI_EP_RDM* endpint and the *FI_EP_DGRAM* endpoint only supports *FI_MR_LOCAL*.

*Progress*
: RDM and DGRAM endpoints support *FI_PROGRESS_MANUAL*.
  EFA erroneously claims the support for *FI_PROGRESS_AUTO*, despite not properly
  supporting automatic progress. Unfortunately, some Libfabric consumers also ask
  for *FI_PROGRESS_AUTO* when they only require *FI_PROGRESS_MANUAL*, and fixing
  this bug would break those applications. This will be fixed in a future version
  of the EFA provider by adding proper support for *FI_PROGRESS_AUTO*.

*Threading*
: Both RDM and DGRAM endpoints supports *FI_THREAD_SAFE*.

# LIMITATIONS

## Completion events
- Synchronous CQ read is not supported.
- Wait objects are not currently supported.

## RMA operations
- Completion events for RMA targets (*FI_RMA_EVENT*) is not supported.
- For the `efa-direct` fabric, the target side of RMA operation must
  insert the initiator side's address into AV before the RMA operation
  is kicked off, due to a current device limitation. The same limitation
  applies to the `efa` fabric when the `FI_OPT_EFA_HOMOGENEOUS_PEERS` option
  is set as `true`.

## [Zero-copy receive mode](https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#48-user-receive-qp-feature--request-and-zero-copy-receive)
 - Zero-copy receive mode can be enabled only if SHM transfer is disabled.
 - Unless the application explicitly disables P2P, e.g. via FI_HMEM_P2P_DISABLED,
  zero-copy receive can be enabled only if available FI_HMEM devices all have
  P2P support.
  
## `fi_cancel` support
 - `fi_cancel` is only supported in the non-zero-copy-receive mode of the `efa` fabric.
 It's not supported in `efa-direct`, DGRAM endpoint, and the zero-copy receive mode of
 the `efa` fabric in RDM endpoint.

When using FI_HMEM for AWS Neuron or Habana SynapseAI buffers, the provider
requires peer to peer transaction support between the EFA and the FI_HMEM
device. Therefore, the FI_HMEM_P2P_DISABLED option is not supported by the EFA
provider for AWS Neuron or Habana SynapseAI.

# PROVIDER SPECIFIC ENDPOINT LEVEL OPTION

*FI_OPT_EFA_RNR_RETRY*
: Defines the number of RNR retry. The application can use it to reset RNR retry
  counter via the call to fi_setopt. Note that this option must be set before
  the endpoint is enabled. Otherwise, the call will fail. Also note that this
  option only applies to RDM endpoint.

*FI_OPT_EFA_EMULATED_READ, FI_OPT_EFA_EMULATED_WRITE, FI_OPT_EFA_EMULATED_ATOMICS - bool*
: These options only apply to the fi_getopt() call.
  They are used to query the EFA provider to determine if the endpoint is
  emulating Read, Write, and Atomic operations (return value is true), or if
  these operations are assisted by hardware support (return value is false).

*FI_OPT_EFA_USE_DEVICE_RDMA - bool*
: This option only applies to the fi_setopt() call.
  Only available if the application selects a libfabric API version >= 1.18.
  This option allows an application to change libfabric's behavior
  with respect to RDMA transfers.  Note that there is also an environment
  variable FI_EFA_USE_DEVICE_RDMA which the user may set as well.  If the
  environment variable and the argument provided with this variable are in
  conflict, then fi_setopt will return -FI_EINVAL, and the environment variable
  will be respected.  If the hardware does not support RDMA and the argument
  is true, then fi_setopt will return -FI_EOPNOTSUPP.  If the application uses
  API version < 1.18, the argument is ignored and fi_setopt returns
  -FI_ENOPROTOOPT.
  The default behavior for RDMA transfers depends on API version.  For
  API >= 1.18 RDMA is enabled by default on any hardware which supports it.
  For API<1.18, RDMA is enabled by default only on certain newer hardware
  revisions.

*FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES - bool*
: This option only applies to the fi_setopt() call.
  It is used to force the endpoint to use in-order send/recv operation for each 128 bytes
  aligned block. Enabling the option will guarantee data inside each 128 bytes
  aligned block being sent and received in order, it will also guarantee data
  to be delivered to the receive buffer only once. If endpoint is not able to
  support this feature, it will return -FI_EOPNOTSUPP for the call to fi_setopt().


*FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES - bool*
: This option only applies to the fi_setopt() call.
  It is used to set the endpoint to use in-order RDMA write operation for each 128 bytes
  aligned block. Enabling the option will guarantee data inside each 128 bytes
  aligned block being written in order, it will also guarantee data to be
  delivered to the target buffer only once. If endpoint is not able to support
  this feature, it will return -FI_EOPNOTSUPP for the call to fi_setopt().

*FI_OPT_EFA_HOMOGENEOUS_PEERS - bool*
: This option only applies to the fi_setopt() call for RDM endpoints on efa fabric. 
  RDM endpoints on efa-direct fabric are unaffected by this option. 
  When set to true, it indicates all peers are homogeneous, meaning they run on the 
  same platform, use the same software versions, and share identical capabilities.
  It accelerates the initial communication setup as interoperability between peers
  is guaranteed. When set to true, the target side of a RMA operation must
  insert the initiator side's address into AV before the RMA operation
  is kicked off, due to a current device limitation.
  The default value is false.

# PROVIDER SPECIFIC DOMAIN OPS
The efa provider exports extensions for operations
that are not provided by the standard libfabric interface. These extensions
are available via the "`fi_ext_efa.h`" header file.

## Domain Operation Extension

Domain operation extension is obtained by calling `fi_open_ops`
(see [`fi_domain(3)`](fi_domain.3.html))
```c
int fi_open_ops(struct fid *domain, const char *name, uint64_t flags,
    void **ops, void *context);
```

Requesting `FI_EFA_DOMAIN_OPS` in `name` returns `ops` as
the pointer to the function table `fi_efa_ops_domain` defined as follows:

```c
struct fi_efa_ops_domain {
	int (*query_mr)(struct fid_mr *mr, struct fi_efa_mr_attr *mr_attr);
};
```

### query_mr
This op queries an existing memory registration as input, and outputs the efa
specific mr attribute which is defined as follows

```c
struct fi_efa_mr_attr {
    uint16_t ic_id_validity;
    uint16_t recv_ic_id;
    uint16_t rdma_read_ic_id;
    uint16_t rdma_recv_ic_id;
};
```

*ic_id_validity*
:	Validity mask of interconnect id fields. Currently the following bits are supported in the mask:

	FI_EFA_MR_ATTR_RECV_IC_ID:
		recv_ic_id has a valid value.

	FI_EFA_MR_ATTR_RDMA_READ_IC_ID:
		rdma_read_ic_id has a valid value.

	FI_EFA_MR_ATTR_RDMA_RECV_IC_ID:
		rdma_recv_ic_id has a valid value.

*recv_ic_id*
:	Physical interconnect used by the device to reach the MR for receive operation. It is only valid when `ic_id_validity` has the `FI_EFA_MR_ATTR_RECV_IC_ID` bit.

*rdma_read_ic_id*
:	Physical interconnect used by the device to reach the MR for RDMA read operation. It is only valid when `ic_id_validity` has the `FI_EFA_MR_ATTR_RDMA_READ_IC_ID` bit.

*rdma_recv_ic_id*
:	Physical interconnect used by the device to reach the MR for RDMA write receive. It is only valid when `ic_id_validity` has the `FI_EFA_MR_ATTR_RDMA_RECV_IC_ID` bit.

#### Return value
**query_mr()** returns 0 on success, or the value of errno on failure
(which indicates the failure reason).


To enable GPU Direct Async (GDA), which allows the GPU to interact directly with the NIC, 
request `FI_EFA_GDA_OPS` in the `name` parameter with efa-direct fabirc.
This returns `ops` as a pointer to the function table `fi_efa_ops_gda` defined as follows:

```c
struct fi_efa_ops_gda {
	int (*query_addr)(struct fid_ep *ep_fid, fi_addr_t addr, uint16_t *ahn,
			  uint16_t *remote_qpn, uint32_t *remote_qkey);
	int (*query_qp_wqs)(struct fid_ep *ep_fid, struct fi_efa_wq_attr *sq_attr, struct fi_efa_wq_attr *rq_attr);
	int (*query_cq)(struct fid_cq *cq_fid, struct fi_efa_cq_attr *cq_attr);
	int (*cq_open_ext)(struct fid_domain *domain_fid,
			   struct fi_cq_attr *attr,
			   struct fi_efa_cq_init_attr *efa_cq_init_attr,
			   struct fid_cq **cq_fid, void *context);
	uint64_t (*get_mr_lkey)(struct fid_mr *mr);
};
```

### query_addr
This op queries the following address information for a given endpoint and destination address.

*ahn*
:	Address handle number.

*remote_qpn*
:	Remote queue pair Number.

*remote_qkey*
:	qkey for the remote queue pair.

#### Return value
**query_addr()** returns FI_SUCCESS on success, or -FI_EINVAL on failure.

### query_qp_wqs
This op queries EFA specific Queue Pair work queue attributes for a given endpoint.
It retrieves the send queue attributes in sq_attr and receive queue attributes in rq_attr, which is defined as follows.

```c
struct fi_efa_wq_attr {
    uint8_t *buffer;
    uint32_t entry_size;
    uint32_t num_entries;
    uint32_t *doorbell;
    uint32_t max_batch;
};
```

*buffer*
:	Queue buffer.

*entry_size*
:	Size of each entry in the queue.

*num_entries*
:	Maximal number of entries in the queue.

*doorbell*
:	Queue doorbell.

*max_batch*
:	Maximum batch size for queue submissions.

#### Return value
**query_qp_wqs()** returns 0 on success, or the value of errno on failure
(which indicates the failure reason).

### query_cq
This op queries EFA specific Completion Queue attributes for a given cq.

```c
struct fi_efa_cq_attr {
    uint8_t *buffer;
    uint32_t entry_size;
    uint32_t num_entries;
};
```

*buffer*
:	Completion queue buffer.

*entry_size*
:	Size of each completion queue entry.

*num_entries*
:	Maximal number of entries in the completion queue.

#### Return value
**query_cq()** returns 0 on success, or the value of errno on failure
(which indicates the failure reason).

### cq_open_ext
This op creates a completion queue with external memory provided via dmabuf.
The memory can be passed by supplying the following struct.

```c
struct fi_efa_cq_init_attr {
	uint64_t flags;
	struct {
		uint8_t *buffer;
		uint64_t length;
		uint64_t offset;
		uint32_t fd;
	} ext_mem_dmabuf;
};
```

*flags*
:	A bitwise OR of the various values described below.

	FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF:
		create CQ with external memory provided via dmabuf.

*ext_mem_dmabuf*
:	Structure containing information about external memory when using
	FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF flag.

	*buffer*
	:	Pointer to the memory mapped in the process's virtual address space. 
		The field is optional, but if not provided, the use of CQ poll interfaces should be avoided.

	*length*
	:	Length of the memory region to use.

	*offset*
	:	Offset within the dmabuf.

	*fd*
	:	File descriptor of the dmabuf.

#### Return value
**cq_open_ext()** returns 0 on success, or the value of errno on failure
(which indicates the failure reason).

### get_mr_lkey
Returns the local memory translation key associated with a MR. The memory registration must have completed successfully before invoking this.

*lkey*
:	local memory translation key used by TX/RX buffer descriptor.

#### Return value
**get_mr_lkey()** returns lkey on success, or FI_KEY_NOTAVAIL if the registration has not completed.


# Traffic Class (tclass) in EFA
To prioritize the messages from a given endpoint, user can specify `fi_info->tx_attr->tclass = FI_TC_LOW_LATENCY` in the fi_endpoint() call to set the service level in rdma-core. All other tclass values will be ignored.

# RUNTIME PARAMETERS

*FI_EFA_IFACE*
: A comma-delimited list of EFA device, i.e. NIC, names that should be visible to
  the application. This paramater can be used to include/exclude NICs to enforce
  process affinity based on the hardware topology. The default value is "all" which
  allows all available NICs to be discovered.

*FI_EFA_TX_SIZE*
: Maximum number of transmit operations before the provider returns -FI_EAGAIN.
  For only the RDM endpoint, this parameter will cause transmit operations to
  be queued when this value is set higher than the default and the transmit queue
  is full.

*FI_EFA_RX_SIZE*
: Maximum number of receive operations before the provider returns -FI_EAGAIN.

# RUNTIME PARAMETERS SPECIFIC TO RDM ENDPOINT

These OFI runtime parameters apply only to the RDM endpoint.

*FI_EFA_RX_WINDOW_SIZE*
: Maximum number of MTU-sized messages that can be in flight from any
  single endpoint as part of long message data transfer.

*FI_EFA_TX_QUEUE_SIZE*
: Depth of transmit queue opened with the NIC. This may not be set to a value
  greater than what the NIC supports.

*FI_EFA_RECVWIN_SIZE*
: Size of out of order reorder buffer (in messages).  Messages
  received out of this window will result in an error.

*FI_EFA_CQ_SIZE*
: Size of any cq created, in number of entries.

*FI_EFA_MR_CACHE_ENABLE*
: Enables using the mr cache and in-line registration instead of a bounce
  buffer for iov's larger than max_memcpy_size. Defaults to true. When
  disabled, only uses a bounce buffer

*FI_EFA_MR_MAX_CACHED_COUNT*
: Sets the maximum number of memory registrations that can be cached at
  any time.

*FI_EFA_MR_MAX_CACHED_SIZE*
: Sets the maximum amount of memory that cached memory registrations can
  hold onto at any time.

*FI_EFA_MAX_MEMCPY_SIZE*
: Threshold size switch between using memory copy into a pre-registered
  bounce buffer and memory registration on the user buffer.

*FI_EFA_MTU_SIZE*
: Overrides the default MTU size of the device.

*FI_EFA_RX_COPY_UNEXP*
: Enables the use of a separate pool of bounce-buffers to copy unexpected
  messages out of the pre-posted receive buffers.

*FI_EFA_RX_COPY_OOO*
: Enables the use of a separate pool of bounce-buffers to copy out-of-order RTS
  packets out of the pre-posted receive buffers.

*FI_EFA_MAX_TIMEOUT*
: Maximum timeout (us) for backoff to a peer after a receiver not ready error.

*FI_EFA_TIMEOUT_INTERVAL*
: Time interval (us) for the base timeout to use for exponential backoff
  to a peer after a receiver not ready error.

*FI_EFA_ENABLE_SHM_TRANSFER*
: Enable SHM provider to provide the communication across all intra-node processes.
  SHM transfer will be disabled in the case where
  [`ptrace protection`](https://wiki.ubuntu.com/SecurityTeam/Roadmap/KernelHardening#ptrace_Protection)
  is turned on. You can turn it off to enable shm transfer.

  FI_EFA_ENABLE_SHM_TRANSFER is parsed during the fi_domain call and is related to the FI_OPT_SHARED_MEMORY_PERMITTED endpoint option.
  If FI_EFA_ENABLE_SHM_TRANSFER is set to true, the FI_OPT_SHARED_MEMORY_PERMITTED endpoint
  option overrides FI_EFA_ENABLE_SHM_TRANSFER. If FI_EFA_ENABLE_SHM_TRANSFER is set to false,
  but the FI_OPT_SHARED_MEMORY_PERMITTED is set to true, the FI_OPT_SHARED_MEMORY_PERMITTED
  setopt call will fail with -FI_EINVAL.

*FI_EFA_SHM_AV_SIZE*
: Defines the maximum number of entries in SHM provider's address vector.

*FI_EFA_SHM_MAX_MEDIUM_SIZE*
: Defines the switch point between small/medium message and large message. The message
  larger than this switch point will be transferred with large message protocol.
  NOTE: This parameter is now deprecated.

*FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE*
: The maximum size for inter EFA messages to be sent by using medium message protocol. Messages which can fit in one packet will be sent as eager message. Messages whose sizes are smaller than this value will be sent using medium message protocol. Other messages will be sent using CTS based long message protocol.

*FI_EFA_FORK_SAFE*
: Enable fork() support. This may have a small performance impact and should only be set when required. Applications that require to register regions backed by huge pages and also require fork support are not supported.

*FI_EFA_RUNT_SIZE*
: The maximum number of bytes that will be eagerly sent by inflight messages uses runting read message protocol (Default 307200).

*FI_EFA_INTER_MIN_READ_MESSAGE_SIZE*
: The minimum message size in bytes for inter EFA read message protocol. If instance support RDMA read, messages whose size is larger than this value will be sent by read message protocol. (Default 1048576).

*FI_EFA_INTER_MIN_READ_WRITE_SIZE*
: The mimimum message size for emulated inter EFA write to use read write protocol. If firmware support RDMA read, and FI_EFA_USE_DEVICE_RDMA is 1, write requests whose size is larger than this value will use the read write protocol (Default 65536). If the firmware supports RDMA write, device RDMA write will always be used.

*FI_EFA_USE_DEVICE_RDMA*
: Specify whether to require or ignore RDMA features of the EFA device.
- When set to 1/true/yes/on, all RDMA features of the EFA device are used. But if EFA device does not support RDMA and FI_EFA_USE_DEVICE_RDMA is set to 1/true/yes/on, user's application is aborted and a warning message is printed.
- When set to 0/false/no/off, libfabric will emulate all fi_rma operations instead of offloading them to the EFA network device. Libfabric will not use device RDMA to implement send/receive operations.
- If not set, RDMA operations will occur when available based on RDMA device ID/version.

*FI_EFA_USE_HUGE_PAGE*
: Specify Whether EFA provider can use huge page memory for internal buffer.
Using huge page memory has a small performance advantage, but can
cause system to run out of huge page memory. By default, EFA provider
will use huge page unless FI_EFA_FORK_SAFE is set to 1/on/true.

*FI_EFA_USE_ZCPY_RX*
: Enables the use of application's receive buffers in place of bounce-buffers when feasible.
(Default: 1). Setting this environment variable to 0 can disable this feature.
Explicitly setting this variable to 1 does not guarantee this feature
due to other requirements. See
https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md
for details.

*FI_EFA_USE_UNSOLICITED_WRITE_RECV*
: Use device's unsolicited write recv functionality when it's available. (Default: 1).
Setting this environment variable to 0 can disable this feature.

*FI_EFA_INTERNAL_RX_REFILL_THRESHOLD*
: The threshold that EFA provider will refill the internal rx pkt pool. (Default: 8).
When the number of internal rx pkts to post is lower than this threshold,
the refill will be skipped.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
