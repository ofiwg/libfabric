---
layout: page
title: fi_rpc(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_rpc - RPC style data transfer operations

fi_rpc / fi_rpcv / fi_rpcdata / fi_rpcmsg
:   Initiate an RPC request

fi_rpc_resp / fi_rpc_respv / fi_rpc_respdata / fi_rpc_respmsg
:   Respond to an RPC request

fi_rpc_discard
:   Discard an RPC request

# SYNOPSIS

```c
#include <rdma/fi_rpc.h>

ssize_t fi_rpc(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, void *resp_buf, size_t resp_len, void *resp_desc,
	fi_addr_t dest_addr, int timeout, void *context);

ssize_t fi_rpcv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	size_t count, struct iovec *resp_buf, void **resp_desc,
	size_t resp_count, fi_addr_t dest_addr, int timeout, void *context);

ssize_t fi_rpcdata(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, void *resp_buf, size_t resp_len, void *resp_desc,
	uint64_t data, fi_addr_t dest_addr, int timeout, void *context);

ssize_t fi_rpcmsg(struct fid_ep *ep, const struct fi_msg_rpc *msg,
	uint64_t flags);

ssize_t fi_rpc_resp(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t rpc_id, void *context);

ssize_t fi_rpc_respv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t dest_addr, uint64_t rpc_id, void *context);

ssize_t fi_rpc_respdata(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, uint64_t data, fi_addr_t dest_addr, uint64_t rpc_id,
	void *context);

ssize_t fi_rpc_respmsg(struct fid_ep *ep, const struct fi_msg_rpc_resp *msg,
	uint64_t flags);

ssize_t fi_rpc_discard(struct fid_ep *ep, uint64_t rpc_id, void *context);
```

# ARGUMENTS

*ep*
: Fabric endpoint on which to send RPC request or response.

*buf*
: Data buffer to send RPC request or response.

*resp_buf*
: Data buffer to receive RPC response.  The content of this buffer is only
  valid if the operation completes successfully.

*len*
: Length of data buffer to send RPC request or response, specified in bytes.  Valid
  transfers are from 0 bytes up to the endpoint's max_msg_size.

*resp_len*
: Length of data buffer to receive RPC response, specified in bytes.  Valid
  transfers are from 0 bytes up to the endpoint's max_msg_size.

*iov*
: Vectored data buffer.

*resp_iov*
: Vectored data buffer to receive RPC response.

*count*
: Count of vectored data entries.

*resp_count*
: Count of vectored data entries to receive RPC response.

*desc*
: Descriptor associated with the data buffer.  See [`fi_mr`(3)](fi_mr.3.html).

*resp_desc*
: Descriptor associated with the data buffer to receive RPC response.  See
  [`fi_mr`(3)](fi_mr.3.html).

*dest_addr*
: Destination address for connectionless transfers.  Ignored for
  connected endpoints.

*msg*
: Message descriptor for send RPC request or response.

*flags*
: Additional flags to apply for the operation.

*timeout*
: Timeout value for the RPC request.  The value is specified in microseconds.
  A negative value indicates no timeout.

*rpc_id*
: RPC identifier for the request.  This value is used to match the request
  with the response.

*context*
: User specified pointer to associate with the operation.  This parameter is
  ignored if the operation will not generate a successful completion, unless
  an op flag specifies the context parameter be used for required input.

# DESCRIPTION

RPC style data transfer involves a client sending a request to a server
and the server sending back a response. The buffer for receiving the response
is supplied with the request.  The content of the request and response is
defined by the application.

An RPC request is delivered into a data buffer posted by message receive
functions -- fi_recv, fi_recvv, fi_recvmsg.  The completion of the receive
operation has FI_RPC set in the `flags` field.  The full completion entry
is defined as `struct fi_cq_rpc_entry`.  The server should set the CQ format
to FI_CQ_FORMAT_RPC in order to get the necessary information for correct
handling of the RPC request.

The RPC ID associated with the request is reported as part of the completion
entry.  The RPC ID uniquely identifies the request and is used to match the
response with the request.  The server must use the RPC ID to send the
response back to the client.

The RPC request functions -- fi_rpc, fi_rpcv, fi_rpcdata, fi_rpcmsg -- post
to an endpoint a data buffer containing the request and a data buffer to
receive the response.  The main difference between these functions are the
number and type of parameters that they accept as input.  Otherwise, they
perform the same general function.  The RPC request operations operate
asynchronously.  Users should not touch the posted data buffer(s) until the
operation has completed.

The RPC response functions -- fi_rpc_resp, fi_rpc_respv, fi_rpc_respdata,
fi_rpc_respmsg -- post a data buffer containing the response for a previously
received RPC request.  Similar to the RPC request functions, the variations
of RPC response functions perform the same general function, but accept
different parameters.  The RPC response operations operate asynchronously.
Users should not touch the posted data buffer until the operation has
completed.

An endpoint must be enabled before an application can post RPC request
or response operations to it.

Completed RPC operations are reported to the user through one or
more event collectors associated with the endpoint.  Users provide
context which are associated with each operation, and is returned to
the user as part of the event completion.  See fi_cq for completion
event details.

## fi_rpc

The call fi_rpc post a user-specified reponse buffer for receiving the
response and transfers the data contained in the user-specified data buffer
to a remote endpoint, with message boundaries being maintained.

## fi_rpcv

The fi_rpcv call adds support for a scatter-gather list to fi_rpc.
The fi_rpcv transfers the set of data buffers referenced by the iov parameter
to a remote endpoint as a single message and posts the set of data buffers
referenced by the resp_iov parameter to receive the response.

## fi_rpcdata

The fi_rpcdata call is similar to fi_rpc, but allows for the sending of
remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the transfer.

## fi_rpcmsg

The fi_rpcmsg call supports RPC request over both connected and
connectionless endpoints, with the ability to control the send operation
per call through the use of flags.  The fi_rpcmsg function takes a
`struct fi_msg_rpc` as input.

```c
struct fi_msg_rpc {
	const struct iovec *req_iov;      /* scatter-gather array for request */
	void               **req_desc;    /* local request descriptors */
	size_t             req_iov_count; /* # elements in req_iov */
	const struct iovec *resp_iov;     /* scatter-gather array for response */
	void               **resp_desc;   /* local response descriptors */
	size_t             resp_iov_count;/* # elements in resp_iov */
	fi_addr_t          addr;          /* optional endpoint address */
	void               *context;      /* user-defined context */
	int		   timeout;       /* optional timeout value */
	uint64_t	   data;          /* optional immediate data */
};
```

## fi_rpc_resp

The fi_rpc_resp call transfers a user-sepicified data buffer to the
original sender of the RPC request identified by the rpc_id parameter.
The response message is delivered to the response buffer supplied with
the original fi_rpc/fi_rpcv/fi_rpcmsg call.

## fi_rpc_respv

The fi_rpc_respv call adds support for a scatter-gather list to fi_rpc_resp.
The fi_rpc_respv call transfers the set of data buffers referenced by the
iov parameter to the orignal sender of the RPC request identified by rpc_id
parameter.

## fi_rpc_respdata

The fi_rpc_respdata call is similar to fi_rpc_resp, but allows for the
sending of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the
transfer.

## fi_rpc_respmsg

The fi_rpcmsg call supports send RPC resposne over both connected and
connectionless endpoints, with the ability to control the operation per
call through the use of flags.  The fi_rpc_respmsg function takes a
struct fi_msg_rpc_resp as input.

```c
struct fi_msg_rpc_resp {
	const struct iovec *iov;     /* scatter-gather array */
	void               **desc;   /* local response descriptors */
	size_t             iov_count;/* # elements in iov */
	fi_addr_t          addr;     /* optional endpoint address */
	fi_addr_t          rpc_id;   /* RPC ID */
	void               *context; /* user-defined context */
	uint64_t	   data;     /* optional immediate data */
};
```

## fi_rpc_discard

The fi_discard call is used to discard a previously received RPC request.
It allows the provider to free any resources associated with the request.
No new fi_rpc_resp/fi_rpc_respv/fi_rpc_respdata/fi_rpc_respmsg calls can
be made for the discarded request but the handling of existing uncompleted
response calls is provider specific.


# FLAGS

The fi_rpcmsg and fi_rpc_respmsg calls allow the user to specify flags
which can change the default message handling of the endpoint.
The following list of flags are usable with fi_rpcmsg and fi_rpc_respmsg.

*FI_REMOTE_CQ_DATA*
: Indicates that remote CQ data is available and should be sent as part of
  the request.  See fi_getinfo for additional details on FI_REMOTE_CQ_DATA.

*FI_COMPLETION*
: Indicates that a completion entry should be generated for the
  specified operation.  The endpoint must be bound to a completion
  queue with FI_SELECTIVE_COMPLETION that corresponds to the
  specified operation, or this flag is ignored.

# NOTES

If an endpoint has been configured with FI_MSG_PREFIX, the application
must include buffer space of size msg_prefix_size, as specified by the
endpoint attributes.  The prefix buffer must occur at the start of the
data referenced by the buf parameter, or be referenced by the first IO vector.
Message prefix space cannot be split between multiple IO vectors.  The size
of the prefix buffer should be included as part of the total buffer length.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

See the discussion below for details handling FI_EAGAIN.

# ERRORS

*-FI_EAGAIN*
: Indicates that the underlying provider currently lacks the resources
  needed to initiate the requested operation.  The reasons for a provider
  returning FI_EAGAIN are varied.  However, common reasons include
  insufficient internal buffering or full processing queues.

  Full processing queues may be a temporary state related to local
  processing (for example, a large message is being transferred), or may be
  the result of flow control.  In the latter case, the queues may remain
  blocked until additional resources are made available at the remote side
  of the transfer.

  In all cases, the operation may be retried after additional resources become
  available.  When using FI_PROGRESS_MANUAL, the application must check for
  transmit and receive completions after receiving FI_EAGAIN as a return value,
  independent of the operation which failed.  This is also strongly recommended
  when using FI_PROGRESS_AUTO, as acknowledgements or flow control messages may
  need to be processed in order to resume execution.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_cq`(3)](fi_cq.3.html)
