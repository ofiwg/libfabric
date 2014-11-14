---
layout: page
title: fi_rma(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_rma - Remote memory access operations

fi_read / fi_readv  
fi_readfrom / fi_readmsg
:   Initiates a read from remote memory

fi_write / fi_writev  
fi_writeto / fi_writemsg  
fi_inject_write / fi_inject_writeto  
fi_writedata / fi_writedatato
:   Initiate a write to remote memory

# SYNOPSIS

{% highlight c %}
#include <rdma/fi_rma.h>

ssize_t fi_read(struct fid_ep *ep, void *buf, size_t len, void *desc,
	uint64_t addr, uint64_t key, void *context);

ssize_t fi_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	size_t count, uint64_t addr, uint64_t key, void *context);

ssize_t fi_readfrom(struct fid_ep *ep, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t addr, uint64_t key,
	void *context);

ssize_t fi_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
	uint64_t flags);

ssize_t fi_write(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, uint64_t addr, uint64_t key, void *context);

ssize_t fi_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
	size_t count, uint64_t addr, uint64_t key, void *context);

ssize_t fi_writeto(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
	void *context);

ssize_t fi_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
	uint64_t flags);

ssize_t fi_inject_write(struct fid_ep *ep, const void *buf, size_t len,
	uint64_t addr, uint64_t key);

ssize_t fi_inject_writeto(struct fid_ep *ep, const void *buf, size_t len,
	fi_addr_t dest_addr, uint64_t addr, uint64_t key);

ssize_t fi_writedata(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, uint64_t data, uint64_t addr, uint64_t key,
	void *context);

ssize_t fi_writedatato(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, uint64_t data, fi_addr_t dest_addr, uint64_t addr,
	uint64_t key, void *context);
{% endhighlight %}

# ARGUMENTS

*ep*
: Fabric endpoint on which to initiate read or write operation.

*buf*
: Local data buffer to read into (read target) or write from (write
  source)

*len*
: Length of data to read or write, specified in bytes.  Valid
  transfers are from 0 bytes up to the endpoint's max_msg_size.

*iov*
: Vectored data buffer.

*count*
: Count of vectored data entries.

*addr*
: Address of remote memory to access.

*key*
: Protection key associated with the remote memory.

*desc*
: Descriptor associated with the local data buffer

*data*
: Remote CQ data to transfer with the operation.

*dest_addr*
: Destination address for connectionless write transfers

*src_addr*
: Source address to read from for connectionless transfers

*msg*
: Message descriptor for read and write operations.

*flags*
: Additional flags to apply for the read or write operation.

*context*
: User specified pointer to associate with the operation.

# DESCRIPTION

RMA (remote memory access) operations are used to transfer data
directly between a local data buffer and a remote data buffer.  RMA
transfers occur on a byte level granularity, and no message boundaries
are maintained.

The write functions -- fi_write, fi_writev, fi_writeto, fi_writemsg,
fi_inject_write, fi_inject_writeto, fi_writedata, and fi_writedatato
-- are used to transmit data into a remote memory buffer.  The main
difference between write functions are the number and type of
parameters that they accept as input.  Otherwise, they perform the
same general function.

The read functions -- fi_read, fi_readv, fi_readfrom, fi_readmsg --
are used to transfer data from a remote memory region into local data
buffer(s).  Similar to the write operations, read operations operate
asynchronously.  Users should not touch the posted data buffer(s)
until the read operation has completed.

Completed RMA operations are reported to the user through one or more
event collectors associated with the endpoint.  Users provide context
which are associated with each operation, and is returned to the user
as part of the event completion.  See fi_eq for completion event
details.

By default, the remote endpoint does not generate an event or notify
the user when a memory region has been accessed by an RMA read or
write operation.  However, immediate data may be associated with an
RMA write operation.  RMA writes with immediate data will generate a
completion entry at the remote endpoint, so that the immediate data
may be delivered.

## fi_write

The call fi_write transfers the data contained in the user-specified
data buffer to a remote memory region.  The local endpoint must be
connected to a remote endpoint or destination before fi_write is
called.  Unless the endpoint has been configured differently, the data
buffer passed into fi_write must not be touched by the application
until the fi_write call completes asynchronously.

## fi_writev

The fi_writev call adds support for a scatter-gather list to fi_write
and/or fi_writemem.  The fi_writev transfers the set of data buffers
referenced by the iov parameter to the remote memory region.  The
format of iov parameter is specified by the user when the endpoint is
created.  See fi_getinfo for more details on iov formats.

## fi_writeto

The fi_writeto function is equivalent to fi_write for unconnected
endpoints.

## fi_writemsg

The fi_writemsg call supports data transfers over both connected and
unconnected endpoints, with the ability to control the write operation
per call through the use of flags.  The fi_writemsg function takes a
struct fi_msg_rma as input.

{% highlight c %}
struct fi_msg_rma {
	const struct iovec *msg_iov;     /* local scatter-gather array */
	void               **desc;       /* operation descriptor */
	size_t             iov_count;    /* # elements in msg_iov */
	const void         *addr;        /* optional endpoint address */
	const struct fi_rma_iov rma_iov; /* remote SGL */
	size_t             rma_iov_count;/* # elements in rma_iov */
	void               *context;     /* user-defined context */
	uint64_t           data;         /* optional immediate data */
};

struct fi_rma_iov {
    uint64_t           addr;         /* target RMA address */
    size_t             len;          /* size of target buffer */
    uint64_t           key;          /* access key */
};
{% endhighlight %}

## fi_inject_write

The write inject call is an optimized version of fi_write.  The
fi_inject_write function behaves as if the FI_INJECT transfer flag
were set, and FI_COMPLETION were not.  That is, the data buffer is
available for reuse immediately on returning from from
fi_inject_write, and no completion event will be generated for this
write.  The completion event will be suppressed even if the endpoint
has not been configured with FI_COMPLETION.  See the flags discussion
below for more details.

## fi_inject_writeto

This call is similar to fi_inject_write, but for unconnected
endpoints.

## fi_writedata

The write data call is similar to fi_write, but allows for the sending
of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the
transfer.

## fi_writedatato

This call is similar to fi_writedata, but for unconnected endpoints.

## fi_read

The fi_read call requests that the remote endpoint transfer data from
the remote memory region into the local data buffer.  The local
endpoint must be connected to a remote endpoint or destination before
fi_read is called.

## fi_readfrom

The fi_readfrom call is equivalent to fi_read for unconnected endpoints.

## fi_readmsg

The fi_readmsg call supports data transfers over both connected and
unconnected endpoints, with the ability to control the read operation
per call through the use of flags.  The fi_readmsg function takes a
struct fi_msg_rma as input.

# FLAGS

The fi_readmsg and fi_writemsg calls allow the user to specify flags
which can change the default data transfer operation.  Flags specified
with fi_readmsg / fi_writemsg override most flags previously
configured with the endpoint, except where noted (see fi_endpoint).
The following list of flags are usable with fi_readmsg and/or
fi_writemsg.

*FI_REMOTE_CQ_DATA*
: Applies to fi_writemsg, fi_writedata, and fi_writedatato.  Indicates
  that remote CQ data is available and should be sent as part of the
  request.  See fi_getinfo for additional details on
  FI_REMOTE_CQ_DATA.

*FI_COMPLETION*
: Indicates that a completion entry should be generated for the
  specified operation.  The endpoint must be bound to an event queue
  with FI_COMPLETION that corresponds to the specified operation, or
  this flag is ignored.

*FI_MORE*
: Indicates that the user has additional requests that will
  immediately be posted after the current call returns.  Use of this
  flag may improve performance by enabling the provider to optimize
  its access to the fabric hardware.

*FI_REMOTE_SIGNAL*
: Indicates that a completion event at the target process should be
  generated for the given operation.  The remote endpoint must be
  configured with FI_REMOTE_SIGNAL, or this flag will be ignored by
  the target.

*FI_INJECT*
: Applies to fi_writemsg.  Indicates that the outbound data buffer
   should be returned to user immediately after the write call
   returns, even if the operation is handled asynchronously.  This may
   require that the underlying provider implementation copy the data
   into a local buffer and transfer out of that buffer.

*FI_REMOTE_COMPLETE*
: Applies to fi_writemsg.  Indicates that a completion should not be
  generated until the operation has completed on the remote side.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# ERRORS

*-FI_EAGAIN*
: Indicates that the underlying provider currently lacks the resources
  needed to initiate the requested operation.  This may be the result
  of insufficient internal buffering, in the case of FI_SEND_BUFFERED,
  or processing queues are full.  The operation may be retried after
  additional provider resources become available, usually through the
  completion of currently outstanding operations.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_eq`(3)](fi_eq.3.html)
