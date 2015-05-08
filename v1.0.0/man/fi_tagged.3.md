---
layout: page
title: fi_tagged(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_tagged \- Tagged data transfer operations

fi_trecv / fi_trecvv / fi_trecvmsg
:   Post a buffer to receive an incoming message

fi_tsend / fi_tsendv / fi_tsendmsg / fi_tinject / fi_tsenddata
:   Initiate an operation to send a message

# SYNOPSIS

{% highlight c %}
#include <rdma/fi_tagged.h>

ssize_t fi_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context);

ssize_t fi_trecvv(struct fid_ep *ep, const struct iovec *iov, void *desc,
	size_t count, fi_addr_t src_addr, uint64_t tag, uint4_t ignore,
	void *context);

ssize_t fi_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
	uint64_t flags);

ssize_t fi_tsend(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t tag, void *context);

ssize_t fi_tsendv(struct fid_ep *ep, const struct iovec *iov,
	void *desc, size_t count, fi_addr_t dest_addr, uint64_t tag,
	void *context);

ssize_t fi_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
	uint64_t flags);

ssize_t fi_tinject(struct fid_ep *ep, const void *buf, size_t len,
	fi_addr_t dest_addr, uint64_t tag);

ssize_t fi_tsenddata(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, uint64_t data, fi_addr_t dest_addr, uint64_t tag,
	void *context);
{% endhighlight %}

# ARGUMENTS

*fid*
: Fabric endpoint on which to initiate tagged communication operation.

*buf*
: Data buffer to send or receive.

*len*
: Length of data buffer to send or receive.

*iov*
: Vectored data buffer.

*count*
: Count of vectored data entries.

*tag*
: Tag associated with the message.

*ignore*
: Mask of bits to ignore applied to the tag for receive operations.

*desc*
: Memory descriptor associated with the data buffer

*data*
: Remote CQ data to transfer with the sent data.

*dest_addr*
: Destination address for connectionless transfers.  Ignored for
  connected endpoints.

*src_addr*
: Source address to receive from for connectionless transfers.  Applies
  only to connectionless endpoints with the FI_DIRECTED_RECV capability
  enabled, otherwise this field is ignored.  If set to FI_ADDR_UNSPEC,
  any source address may match.

*msg*
: Message descriptor for send and receive operations.

*flags*
: Additional flags to apply for the send or receive operation.

*context*
: User specified pointer to associate with the operation.

# DESCRIPTION

Tagged messages are data transfers which carry a key or tag with the
message buffer.  The tag is used at the receiving endpoint to match
the incoming message with a corresponding receive buffer.  Message
tags match when the receive buffer tag is the same as the send buffer
tag with the ignored bits masked out.  This can be stated as:

{% highlight c %}
send_tag & ~ignore == recv_tag & ~ignore
{% endhighlight %}

In general, message tags are checked against receive buffers in the
order in which messages have been posted to the endpoint.  See the
ordering discussion below for more details.

The send functions -- fi_tsend, fi_tsendv, fi_tsendmsg,
fi_tinject, and fi_tsenddata -- are used
to transmit a tagged message from one endpoint to another endpoint.
The main difference between send functions are the number and type of
parameters that they accept as input.  Otherwise, they perform the
same general function.

The receive functions -- fi_trecv, fi_trecvv, fi_recvmsg
-- post a data buffer to an endpoint to receive inbound tagged
messages.  Similar to the send operations, receive operations operate
asynchronously.  Users should not touch the posted data buffer(s)
until the receive operation has completed.  Posted receive buffers are
matched with inbound send messages based on the tags associated with
the send and receive buffers.

Completed message operations are reported to the user through one or
more event collectors associated with the endpoint.  Users provide
context which are associated with each operation, and is returned to
the user as part of the event completion.  See fi_cq for completion
event details.

## fi_tsend

The call fi_tsend transfers the data contained in the user-specified
data buffer to a remote endpoint, with message boundaries being
maintained.  The local endpoint must be connected to a remote endpoint
or destination before fi_tsend is called.  Unless the endpoint has
been configured differently, the data buffer passed into fi_tsend must
not be touched by the application until the fi_tsend call completes
asynchronously.

## fi_tsendv

The fi_tsendv call adds support for a scatter-gather list to fi_tsend.
The fi_sendv transfers the set of data buffers
referenced by the iov parameter to a remote endpoint as a single
message.

## fi_tsendmsg

The fi_tsendmsg call supports data transfers over both connected and
unconnected endpoints, with the ability to control the send operation
per call through the use of flags.  The fi_tsendmsg function takes a
struct fi_msg_tagged as input.

{% highlight c %}
struct fi_msg_tagged {
	const struct iovec *msg_iov; /* scatter-gather array */
	void               *desc;    /* data descriptor */
	size_t             iov_count;/* # elements in msg_iov *
	const void         *addr;    /* optional endpoint address */
	uint64_t           tag;      /* tag associated with message */
	uint64_t           ignore;   /* mask applied to tag for receives */
	void               *context; /* user-defined context */
	uint64_t           data;     /* optional immediate data */
};
{% endhighlight %}

## fi_tinject

The tagged inject call is an optimized version of fi_tsend.  The
fi_tinject function behaves as if the FI_INJECT transfer flag were
set, and FI_COMPLETION were not.  That is, the data buffer is
available for reuse immediately on returning from from fi_tinject, and
no completion event will be generated for this send.  The completion
event will be suppressed even if the endpoint has not been configured
with FI_COMPLETION.  See the flags discussion below for more details.

## fi_tsenddata

The tagged send data call is similar to fi_tsend, but allows for the
sending of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the
transfer.

## fi_trecv

The fi_trecv call posts a data buffer to the receive queue of the
corresponding endpoint.  Posted receives are searched in the order in
which they were posted in order to match sends.  Message boundaries are
maintained.  The order in which the receives complete is dependent on
the endpoint type and protocol.

## fi_trecvv

The fi_trecvv call adds support for a scatter-gather list to fi_trecv.
The fi_trecvv posts the set of data buffers referenced by the iov
parameter to a receive incoming data.

## fi_trecvmsg

The fi_trecvmsg call supports posting buffers over both connected and
unconnected endpoints, with the ability to control the receive
operation per call through the use of flags.  The fi_trecvmsg function
takes a struct fi_msg_tagged as input.

# FLAGS

The fi_trecvmsg and fi_tsendmsg calls allow the user to specify flags
which can change the default message handling of the endpoint.  Flags
specified with fi_trecvmsg / fi_tsendmsg override most flags
previously configured with the endpoint, except where noted (see
fi_endpoint).  The following list of flags are usable with fi_trecvmsg
and/or fi_tsendmsg.

*FI_REMOTE_CQ_DATA*
: Applies to fi_tsendmsg and fi_tsenddata.  Indicates
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

*FI_INJECT*
: Applies to fi_tsendmsg.  Indicates that the outbound data buffer
  should be returned to user immediately after the send call returns,
  even if the operation is handled asynchronously.  This may require
  that the underlying provider implementation copy the data into a
  local buffer and transfer out of that buffer.

*FI_INJECT_COMPLETE*
: Applies to fi_tsendmsg.  Indicates that a completion should be
  generated when the source buffer(s) may be reused.
  
*FI_TRANSMIT_COMPLETE*
: Applies to fi_tsendmsg.  Indicates that a completion should not be
  generated until the operation has been successfully transmitted and
  is no longer being tracked by the provider.

*FI_FENCE*
: Applies to transmits.  Indicates that the requested operation, also
  known as the fenced operation, be deferred until all previous operations
  targeting the same target endpoint have completed.

The following flags may be used with fi_trecvmsg.

*FI_PEEK*
: The peek flag may be used to see if a specified message has arrived.
  A peek request is often useful on endpoints that have provider
  allocated buffering enabled (see fi_rx_attr total_buffered_recv).
  Unlike standard receive operations, a receive operation with the FI_PEEK
  flag set does not remain queued with the provider until the peek completes
  successfully.  If no data is available, the FI_PEEK receive will complete
  with a status of FI_ENOMSG.

  If a peek request locates a matching message, the operation will complete
  successfully.  The returned completion data will indicate the meta-data
  associated with the message, such as the message length, completion flags,
  available CQ data, tag, and source address.  The data available is subject to
  the completion entry format (e.g. struct fi_cq_tagged_entry).

  An application may supply a buffer as part of the peek operation.  If
  given, the provider may return a copy of the message data.  The returned data
  is limited to the size of the input buffer(s) or the message size, if
  smaller.  A provider indicates if data is available by setting the buf
  field of the CQ entry to the user's first input buffer.  If buf is NULL, no
  data was available to return.  A provider may return NULL even if the
  peek operation completes successfully.  Note that the CQ entry len field
  will reference the size of the message, not necessarily the size of the
  returned data.

*FI_CLAIM*
: If this flag is used in conjunction with FI_PEEK, it indicates if the
  peek request completes successfully -- indicating that a matching message
  was located -- the message is claimed by caller.  Claimed messages can only
  be retrieved using a subsequent, paired receive operation with the FI_CLAIM
  flag set.  A receive operation with the FI_CLAIM flag set, but FI_PEEK not
  set is used to retrieve a previously claimed message.

  In order to use the FI_CLAIM flag, an application must supply a struct
  fi_context structure as the context for the receive operation.  The same
  fi_context structure used for an FI_PEEK + FI_CLAIM operation must be used
  by the paired FI_CLAIM request.

*FI_DISCARD*
: This flag must be used in conjunction with either FI_PEEK or FI_CLAIM.
  If this flag is used in conjunction with FI_PEEK, it indicates if the
  peek request completes successfully -- indicating that a matching message
  was located -- the message is discarded by the provider, as the data is not
  needed by the application.  This flag may also be used in conjunction with
  FI_CLAIM in order to retrieve and discard a message previously claimed
  using an FI_PEEK + FI_CLAIM request.

  If this flag is set, the input buffer(s) and length parameters.

# RETURN VALUE

The tagged send and receive calls return 0 on success.  On error, a
negative value corresponding to fabric _errno _ is returned. Fabric
errno values are defined in `fi_errno.h`.

# ERRORS

*-FI_EAGAIN*
: Indicates that the underlying provider currently lacks the resources
  needed to initiate the requested operation.  This may be the result
  of insufficient internal buffering, in the case of FI_INJECT,
  or processing queues are full.  The operation may be retried after
  additional provider resources become available, usually through the
  completion of currently outstanding operations.

*-FI_EINVAL*
: Indicates that an invalid argument was supplied by the user.

*-FI_EOTHER*
: Indicates that an unspecified error occurred.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_cq`(3)](fi_cq.3.html)
