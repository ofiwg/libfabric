---
layout: page
title: fi_wr(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_wr \- Work request data transfer operations

fi_wr_prepare
:   Format a work request for a data transfer operation

fi_wr_modify_addr / fi_wr_modify_iov / fi_wr_modify_rma_iov
fi_wr_modify_tag / fi_wr_modify_data / fi_wr_modify_flags
:   Update a prepared work request in place

fi_wr_queue_tx / fi_wr_queue_rx / fi_wr_queue_trx
:   Queue a prepared work request on an endpoint

# SYNOPSIS

```c
#include <rdma/fi_wr.h>

int fi_wr_prepare(struct fid_ep *ep, const struct fi_wr_attr *attr,
	fi_wr wr, size_t *wr_len);

int fi_wr_modify_addr(struct fid_ep *ep, fi_wr wr,
	fi_addr_t addr);

int fi_wr_modify_iov(struct fid_ep *ep, fi_wr wr,
	const struct iovec *iov, void **desc, size_t count);

int fi_wr_modify_rma_iov(struct fid_ep *ep, fi_wr wr,
	const struct fi_rma_iov *rma_iov, size_t count);

int fi_wr_modify_tag(struct fid_ep *ep, fi_wr wr,
	uint64_t tag, uint64_t ignore);

int fi_wr_modify_data(struct fid_ep *ep, fi_wr wr,
	uint64_t data);

int fi_wr_modify_flags(struct fid_ep *ep, fi_wr wr,
	uint64_t flags);

int fi_wr_queue_tx(struct fid_ep *ep, const fi_wr wr,
	void *context);

int fi_wr_queue_rx(struct fid_ep *ep, const fi_wr wr,
	void *context);

int fi_wr_queue_trx(struct fid_ep *ep, const fi_wr wr,
	void *context);
```

# ARGUMENTS

*ep*
: Fabric endpoint on which to prepare, modify, or queue a work request.

*attr*
: Description of the operation to format into a work request.

*wr*
: Application allocated work request, of size fi_ep_attr::max_tx_wr_size or
  fi_ep_attr::max_rx_wr_size.

*wr_len*
: On input, the number of bytes available in *wr*.  On output, the size of
  the formatted work request, which may be smaller than the input value.

*addr*
: Peer address: the destination for a transmit work request, or the source
  to match for a receive work request.

*iov*
: Vectored local data buffer.

*desc*
: Descriptors associated with the local data buffers.  See
  [`fi_mr`(3)](fi_mr.3.html).

*count*
: Count of vectored data entries.

*rma_iov*
: Vectored remote memory buffer, specified as an address and key per entry.

*tag*
: Tag associated with the message.

*ignore*
: Mask of bits to ignore when matching the tag of an incoming message.
  Applies to tagged receive work requests only.

*data*
: Remote CQ data to transfer with the message.

*context*
: User specified pointer to associate with the queued operation, returned to
  the user as part of the event completion.

# DESCRIPTION

The standard data transfer post operations -- fi_send, fi_write, fi_read,
and their variants -- are monolithic: a single call formats a work request,
hands it to the provider, and initiates it on the fabric.  The work request
operations decompose that lifecycle into discrete stages -- prepare, modify,
queue, and flush -- so that formatting a work request is separated in time
and place from submitting it.

An application that knows its transfers ahead of time formats its work
requests once, at initialization, and repeats only the queue and flush steps
in its inner loop; a work request may be queued any number of times.  An
application that repeats a transfer with only a few fields changed, for
example a different destination address or remote buffer, updates those
fields in place with the fi_wr_modify_* calls rather than formatting a new
work request.  The fi_tx_flush, fi_rx_flush, and fi_trx_flush calls then give a
well-defined point at which queued work is initiated, including operations the
provider deferred under the FI_MORE flag.

Prepare is a control path operation; modify, queue, and flush are data path
operations.  The transport behavior of a work request is identical to that of
the corresponding fi_*msg call, so a prepare-queue-flush sequence and the
standard post calls are interchangeable.

Completed operations are reported to the user through one or more completion
queues associated with the endpoint, exactly as they are for the standard post
calls.  The context reported in a completion is the one supplied to the queue
call, not one carried in the work request.  See [`fi_cq`(3)](fi_cq.3.html) for
completion event details.

## Capability

Work request support is advertised through the FI_WR endpoint capability.  An
application requests it in the `caps` field of the fi_info hints passed to
fi_getinfo, and a provider that supports work requests reports FI_WR in the
returned fi_info.  A provider that reports FI_WR implements the full work
request interface -- fi_wr_prepare, all of the fi_wr_modify_* calls, and the
fi_queue_*_wr calls -- and reports the required work request
sizes in fi_ep_attr::max_tx_wr_size and fi_ep_attr::max_rx_wr_size.  Queued
work requests are initiated with the endpoint flush calls fi_tx_flush,
fi_rx_flush, and fi_trx_flush, which are available independently of FI_WR.

## Work Request

A work request is described on input by a `struct fi_wr_attr` and formatted
into an `fi_wr`.

```c
struct fi_wr_attr {
	enum fi_op_type                 op_type;

	union {
		struct fi_op_msg            *msg;
		struct fi_op_tagged         *tagged;
		struct fi_op_rma            *rma;
		struct fi_op_atomic         *atomic;
		struct fi_op_fetch_atomic   *fetch_atomic;
		struct fi_op_compare_atomic *compare_atomic;
	} op;
};

typedef void *fi_wr;
```

The op_type field selects which member of the op union is valid.  Each
`fi_op_*` descriptor carries the target endpoint, the operation's message
descriptor, and its flags; these are the same descriptors used by
`struct fi_deferred_work` (see [`fi_trigger`(3)](fi_trigger.3.html)).  The
fi_wr_attr, and everything it references, is input only and is not accessed
after fi_wr_prepare returns.

Because the operation descriptors are shared with the deferred work queue
interface, two of their fields do not apply here.  The endpoint they reference
must match the endpoint the call is issued on; a mismatch returns -FI_EINVAL.
Their per operation context, for example fi_msg::context, is ignored, as the
context of a queued operation is supplied by the queue call instead.

`fi_wr` is an opaque handle, defined as `void *`.  Its contents are provider
defined and its buffer size is reported by the provider.  The application
allocates the work request, backing it with a buffer of at least
fi_ep_attr::max_tx_wr_size bytes for a transmit operation or
fi_ep_attr::max_rx_wr_size bytes for a receive operation, and owns it for its
entire lifetime.

A single work request may be queued any number of
times, including concurrently from multiple threads or device work items, each
supplying its own context.  If the provider requires the FI_CONTEXT or
FI_CONTEXT2 mode, the context passed to the queue call must be a
struct fi_context or struct fi_context2 respectively, following the usual rules
for those modes.

## fi_wr_prepare

The fi_wr_prepare call validates its inputs and formats a work request for the
operation described by *attr* into *wr*, updating *wr_len* to the size of the
formatted request.  Prepare does not queue the request, does not consume
endpoint transmit resources, and does not access the data buffers referenced by
the operation.

On output *wr_len* is the size the provider actually used, which may be less
than fi_ep_attr::max_tx_wr_size or fi_ep_attr::max_rx_wr_size.  The required size can
depend on the operation and on its flags, so the endpoint attribute reports the
largest work request the endpoint can produce while *wr_len* reports the size
of this one.

## fi_wr_queue_tx / fi_wr_queue_rx / fi_wr_queue_trx

The queue calls place a prepared work request on an endpoint queue,
associating it with *context*.  The work request is taken by const pointer and
is not retained by the provider: once the call returns successfully, the
application may modify or free *wr* without affecting the queued operation.

Queueing is not guaranteed to initiate the operation. To guarantee that
previously queued work requests have been initiated, an application must flush
the queue they were placed on: fi_tx_flush for work queued with fi_wr_queue_tx,
fi_rx_flush for fi_wr_queue_rx, and fi_trx_flush for fi_wr_queue_trx.  A
provider is also allowed to initiate queued work on its own to make forward
progress, for example when its internal batch limit is reached.

The three calls differ in the queue the work request is placed on, not in the
operation it performs.  After prepare, the operation is part of the formatted
work request, so every transmit work request -- FI_OP_SEND, FI_OP_TSEND,
FI_OP_READ, FI_OP_WRITE, and the atomic operations -- is queued with
fi_wr_queue_tx.  FI_OP_RECV work requests are queued with fi_wr_queue_rx and
FI_OP_TRECV work requests with fi_wr_queue_trx, because a provider may place
untagged and tagged receives on different structures, for example a hardware
receive queue versus a posted tag matching list.  Queuing a work request with a
call that does not match the direction and operation it was prepared for
returns -FI_EINVAL.

Operation ordering, as given by the endpoint's msg_order attribute, follows the
order in which work requests are queued, not the order in which they were
prepared.


## Modifying a Work Request

Each fi_wr_modify_* call updates an already prepared work request,
leaving the rest of it formatted.  The following applies to all of them.

A field may be modified while operations queued from the same work request are
still outstanding; the change applies only to subsequent queue calls.


## fi_wr_modify_addr

The fi_wr_modify_addr call sets the peer address: the destination for a
transmit work request, or the source to match for a receive work request.
*addr* is an address inserted in the endpoint's address vector, or
FI_ADDR_UNSPEC on a receive work request to accept any source.

## fi_wr_modify_iov

The fi_wr_modify_iov call sets the local buffers and their memory descriptors,
and with them the length of the transfer.  For a fetching atomic work request
this sets the operand buffers, not the result buffer.

Length is a single property of the work request rather than one field of each
side of the transfer, and this is the call that owns it; see
fi_wr_modify_rma_iov below.

*count* may not exceed the count used at prepare, since fi_wr_prepare may have
sized the work request for that count; a larger value returns -FI_EINVAL.  A
smaller value is always valid, so an application whose transfers vary in
segment count prepares once at its maximum and reduces the count per transfer.
*count* is passed even though it cannot be increased, because it bounds the
array the provider reads.

## fi_wr_modify_rma_iov

The fi_wr_modify_rma_iov call sets the remote buffers -- their addresses and
keys.  It applies only to RMA and atomic work requests, and returns -FI_EINVAL
otherwise.

The `len` field of each fi_rma_iov is not used to change the transfer length.
A value that does not agree with the current local length returns -FI_EINVAL.

*count* is bounded exactly as it is for fi_wr_modify_iov.

## fi_wr_modify_tag

The fi_wr_modify_tag call sets the tag, and for a tagged receive work request
the ignore bits.  It applies only to tagged work requests, and returns
-FI_EINVAL otherwise.  *ignore* is unused for FI_OP_TSEND and must be 0.

## fi_wr_modify_data

The fi_wr_modify_data call sets the remote CQ data delivered with the
operation.  It applies only to a work request prepared with FI_REMOTE_CQ_DATA
in its operation flags, since that flag determines whether the formatted work
request carries a data field at all, and returns -FI_EINVAL otherwise.

## fi_wr_modify_flags

The fi_wr_modify_flags call replaces the operation flags of a prepared work
request with *flags*.

The flags that select the layout or size of the formatted work request (e.g.
FI_REMOTE_CQ_DATA and FI_INJECT) cannot be changed, because doing so would
require reformatting the work request.  An application that needs such a change
should prepare a new work request.

The operation type is fixed at prepare and is not a flag.  fi_wr_modify_flags
cannot change it.

# FLAGS

The work request calls take no flags argument of their own.  The flags that
apply to an operation are those of the `fi_op_*` descriptor passed to
fi_wr_prepare, and they carry the same meaning as the flags of the
corresponding fi_*msg call; see [`fi_msg`(3)](fi_msg.3.html),
[`fi_tagged`(3)](fi_tagged.3.html), [`fi_rma`(3)](fi_rma.3.html), and
[`fi_atomic`(3)](fi_atomic.3.html) for the flags usable with each operation
type.

FI_MORE retains its meaning that further requests will follow, and the
deferred requests are initiated by the flush call for their queue
(fi_tx_flush, fi_rx_flush, or fi_trx_flush).

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# NOTES

The work request calls described here are a host-side interface.  A device-side
extension is described in [`fi_xpu`(3)](fi_xpu.3.html).

# EXAMPLE

The following prepares an RMA write once, then repeats it to a set of peers,
changing only the destination address:

```c
size_t wr_len = info->ep_attr->max_tx_wr_size;
fi_wr wr = malloc(wr_len);

struct fi_rma_iov rma_iov = { .addr = raddr, .len = length, .key = rkey };
struct iovec iov = { .iov_base = local_buf, .iov_len = length };
struct fi_op_rma rma = {
	.ep    = ep,
	.msg   = { .msg_iov = &iov, .iov_count = 1, .desc = &desc,
	           .addr = dest_addr, .rma_iov = &rma_iov,
	           .rma_iov_count = 1 },
	.flags = 0,
};
struct fi_wr_attr attr = { .op_type = FI_OP_WRITE, .op.rma = &rma };

ret = fi_wr_prepare(ep, &attr, wr, &wr_len);

for (i = 0; i < n; i++) {
	fi_wr_modify_addr(ep, wr, peers[i]);
	fi_wr_queue_tx(ep, wr, &contexts[i]);
}
fi_tx_flush(ep, 0);
```

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_trigger`(3)](fi_trigger.3.html),
[`fi_msg`(3)](fi_msg.3.html),
[`fi_rma`(3)](fi_rma.3.html),
[`fi_cq`(3)](fi_cq.3.html),
[`fi_xpu`(3)](fi_xpu.3.html)
