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

fi_mr_bind
: Associate a registered memory region with a completion counter.

# SYNOPSIS

{% highlight c %}
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

int fi_mr_bind(struct fid_mr *mr, struct fid *bfid, uint64_t flags);
{% endhighlight %}

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
in local transfers.

A provider may hide local registration requirements from applications
by making use of an internal registration cache or similar mechanisms.
Such mechanisms, however, may negatively impact performance for some
applications, notably those which manage their own network buffers.
In order to support as broad range of applications as possible,
without unduly affecting their performance, applications that wish to
manage their own local memory registrations may do so by using the
memory registration calls.  Applications may use the FI_LOCAL_MR
domain mode bit as a guide.

When the FI_LOCAL_MR mode bit is set, applications must register all
data buffers that will be accessed by the local hardware and provide
a valid mem_desc parameter into applicable data transfer operations.
When FI_LOCAL_MR is zero, applications are not required to register
data buffers before using them for local operations (e.g. send and
receive data buffers), and the mem_desc parameter into data transfer
operations is ignored.

Further behavior of memory registration operations is
controlled based on the mr_mode field in the domain attribute.

*Basic Memory Registration Mode*
: If the mr_mode field is set to FI_MR_BASIC, then memory registration
  operations are set to basic mode.  In basic mode, registration occurs
  on allocated data buffers, and the MR attributes are selected by
  the provider.

  Basic mode uses provider assigned attributes for the registered buffers.
  The local memory descriptor and remote memory key are selected by
  the provider.  The address used to access a buffer as the target of an
  RMA or atomic operation is the same as the virtual address of the buffer.

  Applications that support the basic registration mode will need to
  exchange MR parameters with remote peers for RMA and atomic
  operations.  The exchanged data should include both the address of
  the memory region as well as the MR key.

*Scalable Memory Registration Mode*
: If the mr_mode field is set to FI_MR_SCALABLE, then memory registration
  operations are set to scalable mode.  In scalable mode, registration
  occurs on memory address ranges, and the MR attributes are selected by
  the user.

  Memory regions registered as the target of RMA and atomic operations are
  associated with a MR key selected by the application.  If local
  registrations are required (see FI_LOCAL_MR mode), the local descriptor
  will be the same as the remote key.  The resulting memory
  region will be accessible by remote peers starting at a base address of 0.
  Because scalable registration mode refers to memory regions, versus data
  buffers, the address ranges given for a registration request do not need
  to map to data buffers allocated by the application at the time the
  registration call is made.  That is, an application can register any
  range of addresses in their virtual address space, whether or not those
  addresses are backed by physical pages or have been allocated.

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
hardware based on the provided access permissions.  Supported access
permissions are the bitwise OR of the following:

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
struct fi_mr_attr.

{% highlight c %}
struct fi_mr_attr {
	const struct iovec *mr_iov;       /* scatter-gather array */
	size_t             iov_count;     /* # elements in mr_iov */
	uint64_t           access;        /* access permission flags */
	uint64_t           requested_key; /* requested remote key */
	void               *context;      /* user-defined context */
};
{% endhighlight %}

## fi_close

Fi_close is used to release all resources associated with a
registering a memory region.  Once unregistered, further access to the
registered memory is not guaranteed.

When closing the MR, there must be no opened endpoints or counters associated
with the MR.  If resources are still associated with the MR when attempting to
close, the call will return -FI_EBUSY.

## fi_mr_desc / fi_mr_key

The local memory descriptor and remote protection key associated with
a MR may be obtained by calling fi_mr_desc and fi_mr_key,
respectively.  The memory registration must have completed
successfully before invoking these calls.

## fi_mr_bind

The fi_mr_bind function associates a memory region with a
counter, for providers that support the generation of completions
based on fabric operations.  The type of events tracked against the
memory region is based on the bitwise OR of the following flags.

*FI_REMOTE_WRITE*
: Generates an event whenever a remote RMA write or atomic operation
  modify the memory region.

# FLAGS

Flags are reserved for future use and must be 0.

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
