---
layout: page
title: fi_mr(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_mr \- Memory region operations

fi_mr_reg / fi_mr_regv / fi_mr_regattr
: Register local memory buffers for direct fabric access

fi_close
: Deregister registered memory buffers.

fi_mr_desc
: Return a local descriptor associated with a registered memory region

fi_mr_key
: Return the remote key needed to access a registered memory region

fi_mr_raw_attr
: Return raw memory region attributes.

fi_mr_map_raw
: Converts a raw memory region key into a key that is usable for data
  transfer operations.

fi_mr_unmap_key
: Releases a previously mapped raw memory region key.

fi_mr_bind
: Associate a registered memory region with a completion counter.

fi_mr_refresh
: Updates the memory pages associated with a memory region.

fi_mr_enable
: Enables a memory region for use.

# SYNOPSIS

```c
#include <rdma/fi_domain.h>

int fi_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
    uint64_t access, uint64_t offset, uint64_t requested_key,
    uint64_t flags, struct fid_mr **mr, void *context);

int fi_mr_regv(struct fid_domain *domain, const struct iovec * iov,
    size_t count, uint64_t access, uint64_t offset, uint64_t requested_key,
    uint64_t flags, struct fid_mr **mr, void *context);

int fi_mr_regattr(struct fid_domain *domain, const struct fi_mr_attr *attr,
    uint64_t flags, struct fid_mr **mr);

int fi_close(struct fid *mr);

void * fi_mr_desc(struct fid_mr *mr);

uint64_t fi_mr_key(struct fid_mr *mr);

int fi_mr_raw_attr(struct fid_mr *mr, uint64_t *base_addr,
    uint8_t *raw_key, size_t *key_size, uint64_t flags);

int fi_mr_map_raw(struct fid_domain *domain, uint64_t base_addr,
    uint8_t *raw_key, size_t key_size, uint64_t *key, uint64_t flags);

int fi_mr_unmap_key(struct fid_domain *domain, uint64_t key);

int fi_mr_bind(struct fid_mr *mr, struct fid *bfid, uint64_t flags);

int fi_mr_refresh(struct fid_mr *mr, const struct iovec *iov, size, count,
    uint64_t flags)

int fi_mr_enable(struct fid_mr *mr);
```

# ARGUMENTS

*domain*
: Resource domain

*mr*
: Memory region

*bfid*
: Fabric identifier of an associated resource.

*context*
: User specified context associated with the memory region.

*buf*
: Memory buffer to register with the fabric hardware

*len*
: Length of memory buffer to register

*iov*
: Vectored memory buffer.

*count*
: Count of vectored buffer entries.

*access*
: Memory access permissions associated with registration

*offset*
: Optional specified offset for accessing specified registered buffers.
  This parameter is reserved for future use and must be 0.

*requested_key*
: Optional requested remote key associated with registered buffers.

*attr*
: Memory region attributes

*flags*
: Additional flags to apply to the operation.

# DESCRIPTION

Registered memory regions associate memory buffers with permissions
granted for access by fabric resources.  A memory buffer must be
registered with a resource domain before it can be used as the target
of a remote RMA or atomic data transfer.  Additionally, a fabric
provider may require that data buffers be registered before being used
in local transfers.  Memory registration restrictions are controlled
using a separate set of mode bits, specified through the domain
attributes (mr_mode field).

The following apply to memory registration.

*Scalable Memory Registration*
: By default, memory registration is considered scalable.  (For library versions
  1.4 and earlier, this is indicated by setting mr_mode to FI_MR_SCALABLE,
  with the fi_info mode bit FI_LOCAL_MR set to 0).  For versions 1.5 and later,
  scalable is implied by the lack of any mr_mode bits being set.  The setting
  of mr_mode bits therefore adjusts application behavior as described below.
  Default, scalable registration has several properties.

  In scalable mode, registration occurs on memory address ranges.
  Because registration refers to memory regions, versus data buffers, the
  address ranges given for a registration request do not need to map to
  data buffers allocated by the application at the time the registration
  call is made.  That is, an application can register any
  range of addresses in their virtual address space, whether or not those
  addresses are backed by physical pages or have been allocated.

  The resulting memory regions are accessible by peers starting at a base
  address of 0.  That is, the target address that is specified is a byte
  offset into the registered region.

  The application also selects the access key associated with the MR.  The
  key size is restricted to a maximum of 8 bytes.

  With scalable registration, locally accessed data buffers are not registered.
  This includes source buffers for all transmit operations -- sends,
  tagged sends, RMA, and atomics -- as well as buffers posted for receive
  and tagged receive operations.

*FI_MR_LOCAL*
: When the FI_MR_LOCAL mode bit is set, applications must register all
  data buffers that will be accessed by the local hardware and provide
  a valid mem_desc parameter into applicable data transfer operations.
  When FI_MR_LOCAL is zero, applications are not required to register
  data buffers before using them for local operations (e.g. send and
  receive data buffers), and the mem_desc parameter into data transfer
  operations is ignored.

  A provider may hide local registration requirements from applications
  by making use of an internal registration cache or similar mechanisms.
  Such mechanisms, however, may negatively impact performance for some
  applications, notably those which manage their own network buffers.
  In order to support as broad range of applications as possible,
  without unduly affecting their performance, applications that wish to
  manage their own local memory registrations may do so by using the
  memory registration calls.

  Note: the FI_MR_LOCAL mr_mode bit replaces the FI_LOCAL_MR fi_info mode bit.
  When FI_MR_LOCAL is set, FI_LOCAL_MR is ignored.

*FI_MR_RAW*
: Raw memory regions are used to support providers with keys larger than
  64-bits or require setup at the peer.  When the FI_MR_RAW bit is set,
  applications must use fi_mr_raw_attr() locally and fi_mr_map_raw()
  at the peer before targeting a memory region as part of any data transfer
  request.

*FI_MR_VIRT_ADDR*
: The FI_MR_VIRT_ADDR bit indicates that the provider references memory
  regions by virtual address, rather than a 0-based offset.  Peers that
  target memory regions registered with FI_MR_VIRT_ADDR specify the
  destination memory buffer using the target's virtual address, with any
  offset into the region specified as virtual address + offset.  Support
  of this bit typically implies that peers must exchange addressing data
  prior to initiating any RMA or atomic operation.

*FI_MR_ALLOCATED*
: When set, all registered memory regions must be backed by physical
  memory pages at the time the registration call is made.

*FI_MR_PROV_KEY*
: This memory region mode indicates that the provider does not support
  application requested MR keys.  MR keys are returned by the provider.
  Applications that support FI_MR_PROV_KEY can obtain the provider key
  using fi_mr_key(), unless FI_MR_RAW is also set. The returned key should
  then be exchanged with peers prior to initiating an RMA or atomic
  operation.

*FI_MR_MMU_NOTIFY*
: FI_MR_MMU_NOTIFY is typically set by providers that support memory
  registration against memory regions that are not necessarily backed by
  allocated physical pages at the time the memory registration occurs.
  (That is, FI_MR_ALLOCATED is typically 0).  However, such providers
  require that applications notify the provider prior to the MR being
  accessed as part of a data transfer operation.  This notification
  informs the provider that all necessary physical pages now back the
  region.  The notification is necessary for providers that cannot
  hook directly into the operating system page tables or memory management
  unit.  See fi_mr_refresh() for notification details.

*FI_MR_RMA_EVENT*
: This mode bit indicates that the provider must configure memory
  regions that are associated with RMA events prior to their use.  This
  includes all memory regions that are associated with completion counters.
  When set, applications must indicate if a memory region will be
  associated with a completion counter as part of the region's creation.
  This is done by passing in the FI_RMA_EVENT flag to the memory
  registration call.

  Such memory regions will be created in a disabled state and must be
  associated with all completion counters prior to being enabled.  To
  enable a memory region, the application must call fi_mr_enable().
  After calling fi_mr_enable(), no further resource bindings may be
  made to the memory region.

*FI_MR_ENDPOINT*
: This mode bit indicates that the provider associates memory regions
  with endpoints rather than domains.  Memory regions that are
  registered with the provider are created in a disabled state and
  must be bound to an endpoint prior to being enabled.  To bind the
  MR with an endpoint, the application must use fi_mr_bind().  To
  enable the memory region, the application must call fi_mr_enable().

*Basic Memory Registration*
: Basic memory registration is indicated by the FI_MR_BASIC mr_mode bit.
  FI_MR_BASIC is maintained for backwards compatibility (libfabric version
  1.4 or earlier).  The behavior of basic registration is equivalent
  to setting the following mr_mode bits to one: FI_MR_VIRT_ADDR,
  FI_MR_ALLOCATED, and FI_MR_PROV_KEY.  Additionally, providers that
  support basic registration usually required FI_MR_LOCAL.  FI_MR_BASIC 
  must either be set alone, or in conjunction with FI_MR_LOCAL.  Other
  mr_mode bit pairings are invalid.  Unlike other mr_mode bits, if
  FI_MR_BASIC is set on input to fi_getinfo(),  it will not be cleared
  by the provider.  That is, setting FI_MR_BASIC
  to one requests basic registration.

The registrations functions -- fi_mr_reg, fi_mr_regv, and
fi_mr_regattr -- are used to register one or more memory regions with
fabric resources.  The main difference between registration functions
are the number and type of parameters that they accept as input.
Otherwise, they perform the same general function.

By default, memory registration completes synchronously.  I.e. the
registration call will not return until the registration has
completed.  Memory registration can complete asynchronous by binding
the resource domain to an event queue using the FI_REG_MR flag.  See
fi_domain_bind.  When memory registration is asynchronous, in order to
avoid a race condition between the registration call returning and the
corresponding reading of the event from the EQ, the mr output
parameter will be written before any event associated with the
operation may be read by the application.  An asynchronous event will
not be generated unless the registration call returns success (0).

## fi_mr_reg

The fi_mr_reg call registers the user-specified memory buffer with
the resource domain.  The buffer is enabled for access by the fabric
hardware based on the provided access permissions.  See the access
field description for memory region attributes below.

Registered memory is associated with a local memory descriptor and,
optionally, a remote memory key.  A memory descriptor is a provider
specific identifier associated with registered memory.  Memory
descriptors often map to hardware specific indices or keys associated
with the memory region.  Remote memory keys provide limited protection
against unwanted access by a remote node.  Remote accesses to a memory
region must provide the key associated with the registration.

Because MR keys must be provided by a remote process, an application
can use the requested_key parameter to indicate that a specific key
value be returned.  Support for user requested keys is provider
specific and is determined by the mr_mode domain attribute.

Remote RMA and atomic operations indicate the location within a
registered memory region by specifying an address.  The location
is referenced by adding the offset to either the base virtual address
of the buffer or to 0, depending on the mr_mode.

The offset parameter is reserved for future use and must be 0.

For asynchronous memory registration requests, the result will be
reported to the user through an event queue associated with the
resource domain.  If successful, the allocated memory region structure
will be returned to the user through the mr parameter.  The mr address
must remain valid until the registration operation completes.  The
context specified with the registration request is returned with the
completion event.

## fi_mr_regv

The fi_mr_regv call adds support for a scatter-gather list to
fi_mr_reg.  Multiple memory buffers are registered as a single memory
region.  Otherwise, the operation is the same.

## fi_mr_regattr

The fi_mr_regattr call is a more generic, extensible registration call
that allows the user to specify the registration request using a
struct fi_mr_attr (defined below).

## fi_close

Fi_close is used to release all resources associated with a
registering a memory region.  Once unregistered, further access to the
registered memory is not guaranteed.  Active or queued operations that
reference a memory region being closed may fail or result in accesses
to invalid memory.  Applications are responsible for ensuring that a
MR is no longer needed prior to closing it.  Note that accesses to a
closed MR from a remote peer will result in an error at the peer.  The
state of the local endpoint will be unaffected.

When closing the MR, there must be no opened endpoints or counters associated
with the MR.  If resources are still associated with the MR when attempting to
close, the call will return -FI_EBUSY.

## fi_mr_desc

Obtains the local memory descriptor associated with a MR.
The memory registration must have completed successfully before invoking
this call.

## fi_mr_key

Returns the remote protection key associated with a MR.  The memory
registration must have completed successfully before invoking this.  The
returned key may be used in data transfer operations at a peer.  If the
FI_RAW_MR mode bit has been set for the domain, then the memory key must
be obtained using the fi_mr_raw_key function instead.  A return value of
FI_KEY_NOTAVAIL will be returned if the registration has not completed
or a raw memory key is required.

## fi_mr_raw_attr

Returns the raw, remote protection key and base address associated with a MR.
The memory registration must have completed successfully before invoking
this routine.  Use of this call is required if the FI_RAW_MR mode bit has
been set by the provider; however, it is safe to use this call with any
memory region.

A raw key must be mapped by a peer before it can be used in data transfer
operations.  See fi_mr_map_raw below.

## fi_mr_map_raw

Raw protection keys must be mapped to a usable key value before they
can be used for data transfer operations.  The mapping is done by the
peer that initiates the RMA or atomic operation.  The mapping function
takes as input the raw key and its size, and returns the mapped key.
Use of the fi_mr_map_raw function is required if the peer has the
FI_RAW_MR mode bit set, but this routine may be called on any valid
key.  All mapped keys must be freed by calling fi_mr_unmap_key when
access to the peer memory region is no longer necessary.

## fi_mr_unmap_key

This call releases any resources that may have been allocated as part of
mapping a raw memory key.  All mapped keys must be freed before the
corresponding domain is closed.

## fi_mr_bind

The fi_mr_bind function associates a memory region with a
counter or endpoint.  Counter bindings are needed by providers
that support the generation of completions based on fabric
operations.  Endpoint bindings are needed if the provider
associates memory regions with endpoints (see FI_MR_ENDPOINT).

When binding with a counter, the type of events tracked against the
memory region is based on the bitwise OR of the following flags.

*FI_REMOTE_WRITE*
: Generates an event whenever a remote RMA write or atomic operation
  modifies the memory region.  Use of this flag requires that the endpoint
  through which the MR is accessed be created with the FI_RMA_EVENT
  capability.

When binding the memory region to an endpoint, flags should be 0.

## fi_mr_refresh

The use of this call is required to notify the provider of any change
to the physical pages backing a registered memory region if the
FI_MR_MMU_NOTIFY mode bit has been set.  This call informs the provider
that the page table entries associated with the region may have been
modified, and the provider should verify and update the registered
region accordingly.  The iov parameter is optional and may be used
to specify which portions of the registered region requires updating.
Providers are only guaranteed to update the specified address ranges.

The refresh operation has the effect of disabling and re-enabling
access to the registered region.  Any operations from peers that attempt
to access the region will fail while the refresh is occurring.
Additionally, attempts to access the region by the local process
through libfabric APIs may result in a page fault or other fatal operation.

The fi_mr_refresh call is only needed if the physical pages might have
been updated after the memory region was created.

## fi_mr_enable

The enable call is used with memory registration associated with the
FI_MR_RMA_EVENT mode bit.  Memory regions created in the disabled state
must be explicitly enabled after being fully configured by the
application.  Any resource bindings to the MR must be done prior
to enabling the MR.

# MEMORY REGION ATTRIBUTES

Memory regions are created using the following attributes.  The struct
fi_mr_attr is passed into fi_mr_regattr, but individual fields also
apply to other memory registration calls, with the fields passed directly
into calls as function parameters.

```c
struct fi_mr_attr {
	const struct iovec *mr_iov;
	size_t             iov_count;
	uint64_t           access;
	uint64_t           requested_key;
	void               *context;
	size_t             auth_key_size;
	uint8_t            *auth_key;
};
```
## mr_iov

This is an IO vector of addresses that will represent a single memory
region.  The number of entries in the iovec is specified by iov_count.

## iov_count

The number of entries in the mr_iov array.  The maximum number of memory
buffers that may be associated with a single memory region is specified
as the mr_iov_limit domain attribute.  See `fi_domain(3)`.

## access

Indicates the type of access that the local or a peer endpoint has to
the registered memory region.  Supported access permissions are the
bitwise OR of the following flags:

*FI_SEND*
: The memory buffer may be used in outgoing message data transfers.  This
  includes fi_msg and fi_tagged operations.

*FI_RECV*
: The memory buffer may be used to receive inbound message transfers.
  This includes fi_msg and fi_tagged operations.

*FI_READ*
: The memory buffer may be used as the result buffer for RMA read
  and atomic operations on the initiator side.

*FI_WRITE*
: The memory buffer may be used as the source buffer for RMA write
  and atomic operations on the initiator side.

*FI_REMOTE_READ*
: The memory buffer may be used as the source buffer of an RMA read
  operation on the target side.

*FI_REMOTE_WRITE*
: The memory buffer may be used as the target buffer of an RMA write
  or atomic operation.

## requested_key

An application specified access key associated with the memory region.
The MR key must be provided by a remote process when performing RMA
or atomic operations to a memory region.  Applications
can use the requested_key field to indicate that a specific key be
used by the provider.  This allows applications to use well known key
values, which can avoid applications needing to exchange and store keys.
Support for user requested keys is provider specific and is determined
by the mr_mode domain attribute.

## context

Application context associated with asynchronous memory registration
operations.  This value is returned as part of any asynchronous event
associated with the registration.  This field is ignored for synchronous
registration calls.

## auth_key_size

The size of key referenced by the auth_key field in bytes, or 0 if no authorization
key is given.  This field is ignored unless the fabric is opened with API
version 1.5 or greater.

## auth_key

Indicates the key to associate with this memory registration.  Authorization
keys are used to limit communication between endpoints.  Only peer endpoints
that are programmed to use the same authorization key may access the memory
region.  The domain authorization key will be used if the auth_key_size 
provided is 0.  This field is ignored unless the fabric is opened with API 
version 1.5 or greater.

# NOTES

Direct access to an application's memory by a remote peer requires that
the application register the targeted memory buffer(s).  This is typically
done by calling one of the fi_mr_reg* routines.  For FI_MR_PROV_KEY, the
provider will return a key that must be used by the peer when accessing
the memory region.  The application is responsible for transferring this
key to the peer.  If FI_MR_RAW mode has been set, the key must be retrieved
using the fi_mr_raw_attr function.

FI_RAW_MR allows support for providers that require more than 8-bytes for
their protection keys or need additional setup before a key can
be used for transfers.  After a raw key has been retrieved, it must be
exchanged with the remote peer.  The peer must use fi_mr_map_raw to convert
the raw key into a usable 64-bit key.  The mapping must be done even if
the raw key is 64-bits or smaller.

The raw key support functions are usable with all registered memory
regions, even if FI_MR_RAW has not been set.  It is recommended that
portable applications target using those interfaces; however, their use
does carry extra message and memory footprint overhead, making it less
desirable for highly scalable apps.

# FLAGS

The follow flag may be specified to any memory registration call.

*FI_RMA_EVENT*
: This flag indicates that the specified memory region will be
  associated with a completion counter used to count RMA operations
  that access the MR.

*FI_RMA_PMEM*
: This flag indicates that the underlying memory region is backed by
  persistent memory and will be used in RMA operations.  It must be
  specified if persistent completion semantics or persistent data transfers
  are required when accessing the registered region.

# RETURN VALUES

Returns 0 on success.  On error, a negative value corresponding to
fabric errno is returned.

Fabric errno values are defined in
`rdma/fi_errno.h`.

# ERRORS

*-FI_ENOKEY*
: The requested_key is already in use.

*-FI_EKEYREJECTED*
: The requested_key is not available.  They key may be out of the
  range supported by the provider, or the provider may not support
  user-requested memory registration keys.

*-FI_ENOSYS*
: Returned by fi_mr_bind if the provider does not support reporting
  events based on access to registered memory regions.

*-FI_EBADFLAGS*
: Returned if the specified flags are not supported by the provider.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_rma`(3)](fi_rma.3.html),
[`fi_msg`(3)](fi_msg.3.html),
[`fi_atomic`(3)](fi_atomic.3.html)
