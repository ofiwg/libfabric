---
layout: page
title: fi_cm(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_cm - Connection management operations

fi_connect / fi_listen / fi_accept / fi_reject / fi_shutdown
: Manage endpoint connection state.

fi_getname / fi_getpeer
: Return local or peer endpoint address

# SYNOPSIS

{% highlight c %}
#include <rdma/fi_cm.h>

int fi_connect(struct fid_ep *ep, const void *addr,
    const void *param, size_t paramlen);

int fi_listen(struct fid_pep *pep);

int fi_accept(struct fid_ep *ep, const void *param, size_t paramlen);

int fi_reject(struct fid_pep *pep, fi_connreq_t connreq,
    const void *param, size_t paramlen);

int fi_shutdown(struct fid_ep *ep, uint64_t flags);

int fi_getname(fid_t fid, void *addr, size_t *addrlen);

int fi_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen);
{% endhighlight %}

# ARGUMENTS

*ep / pep*
: Fabric endpoint on which to change connection state.

*addr*
: Buffer to store queried address (get), or address to
  connect.  The address must be in the same format as that
  specified using fi_info: addr_format when the endpoint was created.

*addrlen*
: On input, specifies size of addr buffer.  On output, stores number
  of bytes written to addr buffer.

*param*
: User-specified data exchanged as part of the connection exchange.

*paramlen*
: Size of param buffer.

*info*
: Fabric information associated with a connection request.

*flags*
: Additional flags for controlling connection operation.

*context*
: User context associated with the request.

# DESCRIPTION

Connection management functions are used to connect an 
connection-oriented endpoint to a peer endpoint.

## fi_listen

The fi_listen call indicates that the specified endpoint should be
transitioned into a passive connection state, allowing it to accept
incoming connection requests.  Connection requests against a listening
endpoint are reported asynchronously to the user through a bound CM
event queue using the FI_CONNREQ event type.  The number of outstanding
connection requests that can be queued at an endpoint is limited by the
listening endpoint's backlog parameter.  The backlog is initialized
based on administrative configuration values, but may be adjusted
through the fi_control call.

## fi_connect

The fi_connect call initiates a connection request on a
connection-oriented endpoint to the destination address.

## fi_accept / fi_reject

The fi_accept and fi_reject calls are used on the passive (listening)
side of a connection to accept or reject a connection request,
respectively.  To accept a connection, the listening application first
waits for a connection request event (FI_CONNREQ).
After receiving such an event, the application
allocates a new endpoint to accept the connection.  This endpoint must
be allocated using an fi_info structure referencing the connreq from this
FI_CONNREQ event.  fi_accept is then invoked
with the newly allocated endpoint.  If
the listening application wishes to reject a connection request, it calls
fi_reject with the listening endpoint and
a reference to the connection request.

A successfully accepted connection request will result in the active
(connecting) endpoint seeing an FI_CONNECTED event on its associated
event queue.  A rejected or failed connection request will generate an
error event.  The error entry will provide additional details describing
the reason for the failed attempt.

An FI_CONNECTED event will also be generated on the passive side for the
accepting endpoint once the connection has been properly established.
The fid of the FI_CONNECTED event will be that of the endpoint passed to
fi_accept as opposed to the listening passive endpoint.
Outbound data transfers cannot be initiated on a connection-oriented
endpoint until an FI_CONNECTED event has been generated.  However, receive
buffers may be associated with an endpoint anytime.

For connection-oriented endpoints, the param buffer will be sent as
part of the connection request or response, subject to the constraints of
the underlying connection protocol.  Applications may use fi_control
to determine the size of application data that may be exchanged as
part of a connection request or response.  The fi_connect, fi_accept, and
fi_reject calls will silently truncate any application data which cannot
fit into underlying protocol messages.

## fi_shutdown

The fi_shutdown call is used to gracefully disconnect an endpoint from
its peer.  If shutdown flags are 0, the endpoint is fully disconnected,
and no additional data transfers will be possible.  Flags may also be
used to indicate that only outbound (FI_WRITE) or inbound (FI_READ) data
transfers should be disconnected.  Regardless of the shutdown option
selected, any queued completions associated with asynchronous operations
may still be retrieved from the corresponding event queues.

An FI_SHUTDOWN event will be generated for an endpoint when the remote
peer issues a disconnect using fi_shutdown or abruptly closes the endpoint.

## fi_getname / fi_getpeer

The fi_getname and fi_getpeer calls may be used to retrieve the local or
peer endpoint address, respectively.  On input, the addrlen parameter should
indicate the size of the addr buffer.  If the actual address is larger than
what can fit into the buffer, it will be truncated.  On output, addrlen
is set to the size of the buffer needed to store the address, which may
be larger than the input value.

# FLAGS

Flag values are reserved and must be 0.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# ERRORS


# NOTES


# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_eq`(3)](fi_eq.3.html)
