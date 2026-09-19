---
layout: page
title: fi_wr(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_wr \- OFI Work Request (WR) API

# SYNOPSIS

```c
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>

int fi_prepare_wr(struct fid_ep *ep, struct fi_wr_attr *work);

int fi_queue_wr(struct fid_ep *ep, struct fi_wr_attr *work);

int fi_flush_wr(struct fid_ep *ep);
```

# OVERVIEW

The standard data transfer post operations (fi_send, fi_write, fi_read, etc.)
are monolithic: a single call constructs the work request, hands it to the
provider, and initiates it on the fabric. The Work Request (WR) API decomposes
that lifecycle into three discrete stages -- *prepare*, *queue*, and *flush* --
so that work request construction, provider-specific metadata injection, and
submission can be separated in time and place. This enables:

- **Staged posting** -- a work request can be constructed at one time or place
  and submitted (staged with the provider, then committed) at another. This
  supports data paths that prepare work requests ahead of time and submit them
  later, including submission driven from a device rather than the host CPU.

- **Provider-specific work request metadata** -- extra per-request fields
  beyond the standard parameters (addr, desc, rkey, data) can be injected into
  a prepared work request through provider-specific setters before it is
  submitted.

- **Explicit submission control** -- the flush stage gives an application a
  single, well-defined point at which to commit staged work to the provider,
  including operations a provider deferred under the existing FI_MORE flag.

The WR API reuses the operation descriptors and control commands of the
Deferred Work Queue interface (see [`fi_trigger`(3)](fi_trigger.3.html)): the
`struct fi_op_*` descriptors discriminated by `enum fi_op_type`, and the
`FI_QUEUE_WORK` and `FI_FLUSH_WORK` `fi_control` commands. When these commands
are issued on an endpoint fid they drive the WR API described here; on a domain
fid they retain their existing Deferred Work Queue meaning.

# CAPABILITY

WR API support is advertised through the `FI_WR` endpoint capability. An
application requests it in the `caps` field of the `fi_info` hints passed to
fi_getinfo, and a provider that supports the WR API reports `FI_WR` in the
returned `fi_info`. A provider that reports `FI_WR` supports the full WR API
(fi_prepare_wr, fi_queue_wr, fi_flush_wr) and the FI_OPT_TX_REQ_SIZE /
FI_OPT_RX_REQ_SIZE endpoint options used to size request buffers.

```c
hints->caps |= FI_WR;
fi_getinfo(version, node, service, 0, hints, &info);
if (info->caps & FI_WR) {
	/* WR API is supported */
}
```

# WORK REQUEST DESCRIPTOR

A work request is described by `struct fi_wr_attr`, defined in
`rdma/fi_trigger.h`:

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

	void   *wr_buf;    /* caller-allocated work request buffer */
	size_t  wr_len;    /* [in] buffer capacity; [out] size after prepare */
};
```

The `op_type` field selects which member of the `op` union is used, and each
`fi_op_*` descriptor carries the target endpoint, the operation's message
descriptor, and its flags (the same descriptors used by
`struct fi_deferred_work`; see [`fi_trigger`(3)](fi_trigger.3.html)). The
endpoint referenced by the operation descriptor must match the endpoint the
command is issued on.

`wr_buf` is a caller-allocated buffer that the provider fills with the
formatted work request during prepare; `wr_len` is the buffer capacity on
input and the actual work request size on output. The required buffer size is
queried with fi_getopt using the FI_OPT_TX_REQ_SIZE option (FI_OPT_RX_REQ_SIZE
for receive-side requests); the work request format is specific to the
endpoint, so the size is an endpoint property.

# OPERATIONS

## fi_prepare_wr

Validates the inputs and writes a formatted work request into `work.wr_buf`,
updating `work.wr_len` to the actual size. The buffer content is opaque to the
application but may be passed to provider-specific setters before queuing. The
buffer remains valid until the application reuses or frees it. Prepare does not
submit the request or consume endpoint transmit resources, and referenced data
buffers are not accessed at prepare time.

## fi_queue_wr

Stages the prepared work request (`work.wr_buf`, `work.wr_len`) with the
endpoint for transmission. It does not initiate the request -- the application
does that with fi_flush_wr -- except when the provider's batch limit is
reached, in which case the provider automatically flushes the pending requests
first. If the endpoint's transmit resources are exhausted, the provider
flushes any pending requests before returning -FI_EAGAIN, and `work.wr_buf`
remains valid for retry.

## fi_flush_wr

Commits all queued-but-not-yet-flushed work requests on the endpoint, after
which the provider begins processing (transmitting) them. If nothing is pending
it is a no-op. fi_flush_wr also commits operations deferred earlier under the
FI_MORE flag, providing a single point to initiate both WR API requests and
FI_MORE-batched transfers.

# PROVIDER-SPECIFIC SETTERS

Between fi_prepare_wr and fi_queue_wr, an application may inject
provider-specific metadata into the prepared work request using
provider-specific functions that operate directly on `wr_buf`. Such setters
are defined in provider extension headers and are outside the scope of this
page. Single-value hints do not need a setter; they are carried in the flags
field of the operation descriptor supplied to prepare. An application must not
call setters after the request has been queued.

# BATCHING

The provider manages batching internally, tracking the number of pending
(queued but not yet flushed) requests per endpoint. fi_queue_wr never
initiates transmission on its own; the application accumulates requests and
commits them once with fi_flush_wr. The one exception is the provider's batch
limit: when the pending count reaches that limit, the provider automatically
flushes before staging the next request. This auto-flush is transparent and
never surfaces as an error. The batch limit is not exposed as a general API;
an application may issue any number of fi_queue_wr calls followed by a single
fi_flush_wr.

# RETURN VALUE

All three calls return 0 on success and a negative fabric errno on failure.

fi_prepare_wr returns -FI_EINVAL for invalid inputs, -FI_ETOOSMALL if the
supplied `wr_buf`/`wr_len` is smaller than the work request the provider must
write, and -FI_ENOSYS or -FI_EOPNOTSUPP if the operation is not supported.

fi_queue_wr returns -FI_EAGAIN if the endpoint's transmit resources are
exhausted; this is a transient, retryable condition (the provider has already
flushed pending requests before returning), and the application should progress
the endpoint to reap completions and retry the same request. Any other negative
value is fatal.

fi_flush_wr returns 0 on success, including the no-op case where nothing is
pending.

A return of -FI_ENOSYS from any call indicates the provider does not implement
the WR API; the application should fall back to fi_write / fi_send / etc.

# EXAMPLE

```c
size_t req_size, len = sizeof(req_size);
fi_getopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_TX_REQ_SIZE, &req_size, &len);
void *buf = malloc(req_size);

struct fi_rma_iov rma_iov = { .addr = raddr, .len = length, .key = rkey };
struct iovec iov = { .iov_base = local_buf, .iov_len = length };
struct fi_op_rma rma = {
	.ep    = ep,
	.msg   = { .msg_iov = &iov, .iov_count = 1, .desc = &desc,
	           .addr = dest_addr, .rma_iov = &rma_iov, .rma_iov_count = 1,
	           .context = context },
	.flags = 0,
};
struct fi_wr_attr work = {
	.op_type = FI_OP_WRITE,
	.op.rma  = &rma,
	.wr_buf = buf,
	.wr_len = req_size,
};

fi_prepare_wr(ep, &work);
/* optionally inject provider-specific metadata into work.wr_buf here */
fi_queue_wr(ep, &work);
fi_flush_wr(ep);
```

# NOTES

The WR API is a host-side interface. A device-side extension that lets an
accelerator kernel prepare, queue, and flush work requests is a possible future
addition; see [`fi_xpu`(3)](fi_xpu.3.html).

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_trigger`(3)](fi_trigger.3.html),
[`fi_mr`(3)](fi_mr.3.html),
[`fi_xpu`(3)](fi_xpu.3.html)
