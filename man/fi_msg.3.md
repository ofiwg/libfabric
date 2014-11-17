---
layout: page
title: fi_msg(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_msg - Message data transfer operations

fi_recv / fi_recvv / fi_recvmsg
:   Post a buffer to receive an incoming message

fi_send / fi_sendv / fi_sendmsg  
fi_inject / fi_senddata
:   Initiate an operation to send a message

# SYNOPSIS

{% highlight c %}
#include <rdma/fi_endpoint.h>

ssize_t fi_recv(struct fid_ep *ep, void * buf, size_t len,
	void *desc, fi_addr_t src_addr, void *context);

ssize_t fi_recvv(struct fid_ep *ep, const struct iovec *iov, void *desc,
	size_t count, fi_addr_t src_addr, void *context);

ssize_t fi_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
	uint64_t flags);

ssize_t fi_send(struct fid_ep *ep, void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, void *context);

ssize_t fi_sendv(struct fid_ep *ep, const void *iov, void *desc,
	size_t count, fi_addr_t dest_addr, void *context);

ssize_t fi_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
	uint64_t flags);

ssize_t fi_inject(struct fid_ep *ep, void *buf, size_t len,
	fi_addr_t dest_addr);

ssize_t fi_senddata(struct fid_ep *ep, void *buf, size_t len,
	void *desc, uint64_t data, fi_addr_t dest_addr, void *context);
{% endhighlight %}

# ARGUMENTS

*ep*
: Fabric endpoint on which to initiate send or post receive buffer.

*buf*
: Data buffer to send or receive.

*len*
: Length of data buffer to send or receive, specified in bytes.  Valid
  transfers are from 0 bytes up to the endpoint's max_msg_size.

*iov*
: Vectored data buffer.

*count*
: Count of vectored data entries.

*desc*
: Descriptor associated with the data buffer

*data*
: Remote CQ data to transfer with the sent message.

*dest_addr*
: Destination address for connectionless transfers.  Ignored for
  connected endpoints.

*src_addr*
: Source address to receive from for connectionless transfers.  Ignored
  for connected endpoints.

*msg*
: Message descriptor for send and receive operations.

*flags*
: Additional flags to apply for the send or receive operation.

*context*
: User specified pointer to associate with the operation.

# DESCRIPTION

The send functions -- fi_send, fi_sendv, fi_sendmsg,
fi_inject, and fi_senddata -- are used to
transmit a message from one endpoint to another endpoint.  The main
difference between send functions are the number and type of
parameters that they accept as input.  Otherwise, they perform the
same general function.  Messages sent using fi_msg operations are
received by a remote endpoint into a buffer posted to receive such
messages.

The receive functions -- fi_recv, fi_recvv, fi_recvmsg --
post a data buffer to an endpoint to receive inbound messages.
Similar to the send operations, receive operations operate
asynchronously.  Users should not touch the posted data buffer(s)
until the receive operation has completed.

Completed message operations are reported to the user through one or
more event collectors associated with the endpoint.  Users provide
context which are associated with each operation, and is returned to
the user as part of the event completion.  See fi_eq for completion
event details.

## fi_send

The call fi_send transfers the data contained in the user-specified
data buffer to a remote endpoint, with message boundaries being
maintained.  The local endpoint must be connected to a remote endpoint
or destination before fi_send is called.  Unless the endpoint has been
configured differently, the data buffer passed into fi_send must not
be touched by the application until the fi_send call completes
asynchronously.

## fi_sendv

The fi_sendv call adds support for a scatter-gather list to fi_send.
The fi_sendv transfers the set of data buffers
referenced by the iov parameter to a remote endpoint as a single
message.

## fi_sendmsg

The fi_sendmsg call supports data transfers over both connected and
unconnected endpoints, with the ability to control the send operation
per call through the use of flags.  The fi_sendmsg function takes a
`struct fi_msg` as input.

{% highlight c %}
struct fi_msg {
	const struct iovec *msg_iov;/* scatter-gather array */
	void               **desc;  /* local request descriptor */
	size_t             iov_count;/* # elements in iov */
	const void         *addr;   /* optional endpoint address */
	void               *context;/* user-defined context */
	uint64_t           data;    /* optional message data */
};
{% endhighlight %}

## fi_inject

The send inject call is an optimized version of fi_send.  The
fi_inject function behaves as if the FI_INJECT transfer flag were
set, and FI_COMPLETION were not.  That is, the data buffer is
available for reuse immediately on returning from from fi_inject, and
no completion event will be generated for this send.  The completion
event will be suppressed even if the endpoint has not been configured
with FI_COMPLETION.  See the flags discussion below for more details.

## fi_senddata

The send data call is similar to fi_send, but allows for the sending
of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the
transfer.

## fi_recv

The fi_recv call posts a data buffer to the receive queue of the
corresponding endpoint.  Posted receives are searched in the order in
which they were posted in order to match sends.
Message boundaries are maintained.  The order in which
the receives complete is dependent on
the endpoint type and protocol.  For unconnected endpoints, the
src_addr parameter can be used to indicate that a buffer should be
posted to receive incoming data from a specific remote endpoint.

## fi_recvv

The fi_recvv call adds support for a scatter-gather list to fi_recv.
The fi_recvv posts the set of data buffers referenced by the iov
parameter to a receive incoming data.

## fi_recvmsg

The fi_recvmsg call supports posting buffers over both connected and
unconnected endpoints, with the ability to control the receive
operation per call through the use of flags.  The fi_recvmsg function
takes a struct fi_msg as input.

# FLAGS

The fi_recvmsg and fi_sendmsg calls allow the user to specify flags
which can change the default message handling of the endpoint.  Flags
specified with fi_recvmsg / fi_sendmsg override most flags previously
configured with the endpoint, except where noted (see fi_endpoint).
The following list of flags are usable with fi_recvmsg and/or
fi_sendmsg.

*FI_REMOTE_CQ_DATA*
: Applies to fi_sendmsg, fi_senddata, and fi_senddatato.  Indicates
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
: Applies to fi_sendmsg.  Indicates that the outbound data buffer
  should be returned to user immediately after the send call returns,
  even if the operation is handled asynchronously.  This may require
  that the underlying provider implementation copy the data into a
  local buffer and transfer out of that buffer.

*FI_MULTI_RECV*
: Applies to posted receive operations.  This flag allows the user to
  post a single buffer that will receive multiple incoming messages.
  Received messages will be packed into the receive buffer until the
  buffer has been consumed.  Use of this flag may cause a single
  posted receive operation to generate multiple events as messages are
  placed into the buffer.  The placement of received data into the
  buffer may be subjected to provider specific alignment restrictions.
  The buffer will be freed from the endpoint when the available buffer
  space falls below the network's MTU size (see
  FI_OPT_MIN_MULTI_RECV).

*FI_REMOTE_COMPLETE*
: Applies to fi_sendmsg.  Indicates that a completion should not be
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
