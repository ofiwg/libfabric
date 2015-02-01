---
layout: page
title: fi_domain(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_domain \- Open a fabric access domain

# SYNOPSIS

{% highlight c %}
#include <rdma/fabric.h>

#include <rdma/fi_domain.h>

int fi_domain(struct fid_fabric *fabric, struct fi_info *info,
    struct fid_domain **domain, void *context);

int fi_close(struct fid *domain);

int fi_domain_bind(struct fid_domain *domain, struct fid *eq,
    uint64_t flags);

int fi_open_ops(struct fid *domain, const char *name, uint64_t flags,
    void **ops, void *context);
{% endhighlight %}

# ARGUMENTS

*fabric*
: Fabric domain

*info*
: Fabric information, including domain capabilities and attributes.

*domain*
: An opened access domain.

*context*
: User specified context associated with the domain.  This context is
  returned as part of any asynchronous event associated with the
  domain.

*eq*
: Event queue for asynchronous operations initiated on the domain.

*name*
: Name associated with an interface.

*ops*
: Fabric interface operations.

# DESCRIPTION

An access domain typically refers to a physical or virtual NIC or
hardware port; however, a domain may span across multiple hardware
components for fail-over or data striping purposes.  A domain defines
the boundary for associating different resources together.  Fabric
resources belonging to the same domain may share resources.

## fi_domain

Opens a fabric access domain, also referred to as a resource domain.
Fabric domains are identified by a name.  The properties of the opened
domain are specified using the info parameter.

## fi_open_ops

fi_open_ops is used to open provider specific interfaces.  Provider
interfaces may be used to access low-level resources and operations
that are specific to the opened resource domain.  The details of
domain interfaces are outside the scope of this documentation.

## fi_domain_bind

Associates an event queue with the domain.  An event queue bound to a
domain will be the default EQ associated with asynchronous control
events that occur on the domain or active endpoints allocated on a
domain.  This includes CM events.  Endpoints may direct their control
events to alternate EQs by binding directly with the EQ.

Binding an event queue to a domain with the FI_REG_MR flag indicates
that the provider should perform all memory registration operations
asynchronously, with the completion reported through the event queue.
If an event queue is not bound to the domain with the FI_REG_MR flag,
then memory registration requests complete synchronously.

## fi_close

The fi_close call is used to release all resources associated with a domain or
interface.  All objects associated with the opened domain must be released
prior to calling fi_close, otherwise the call will return -FI_EBUSY.

# DOMAIN ATTRIBUTES

The `fi_domain_attr` structure defines the set of attributes associated
with a domain.

{% highlight c %}
struct fi_domain_attr {
	struct fid_domain     *domain;
	char                  *name;
	enum fi_threading     threading;
	enum fi_progress      control_progress;
	enum fi_progress      data_progress;
	enum fi_resource_mgmt resource_mgmt;
	size_t                mr_key_size;
	size_t                cq_data_size;
	size_t                cq_cnt;
	size_t                ep_cnt;
	size_t                tx_ctx_cnt;
	size_t                rx_ctx_cnt;
	size_t                max_ep_tx_ctx;
	size_t                max_ep_rx_ctx;
};
{% endhighlight %}

## domain

On input to fi_getinfo, a user may set this to an opened domain
instance to restrict output to the given domain.  On output from
fi_getinfo, if no domain was specified, but the user has an opened
instance of the named domain, this will reference the first opened
instance.  If no instance has been opened, this field will be NULL.

## Name

The name of the access domain.

## Multi-threading Support (threading)

The threading model specifies the level of serialization required of
an application when using the libfabric data transfer interfaces.
Control interfaces are always considered thread safe, and may be
accessed by multiple threads.  Applications which can guarantee
serialization in their access of provider allocated resources and
interfaces enables a provider to eliminate lower-level locks.

*FI_THREAD_UNSPEC*
: This value indicates that no threading model has been defined.  It
  may be used on input hints to the fi_getinfo call.  When specified,
  providers will return a threading model that allows for the greatest
  level of parallelism.

*FI_THREAD_SAFE*
: A thread safe serialization model allows a multi-threaded
  application to access any allocated resources through any interface
  without restriction.  All providers are required to support
  FI_THREAD_SAFE.

*FI_THREAD_FID*
: A fabric descriptor (FID) serialization model requires applications
  to serialize access to individual fabric resources associated with
  data transfer operations and completions.  Multiple threads must
  be serialized when accessing the same endpoint, transmit context,
  receive context, completion queue, counter, wait set, or poll set.
  Serialization is required only by threads accessing the same object.

  For example, one thread may be initiating a data transfer on an
  endpoint, while another thread reads from a completion queue
  associated with the endpoint.
  
  Serialization to endpoint access is only required when accessing
  the same endpoint data flow.  Multiple threads may initiate transfers
  on different transmit contexts of the same endpoint without serializing,
  and no serialization is required between the submission of data
  transmit requests and data receive operations.
  
  In general, FI_THREAD_FID allows the provider to be implemented
  without needing internal locking when handling data transfers.
  Conceptually, FI_THREAD_FID maps well to providers that implement
  fabric services in hardware and provide separate command queues to
  different data flows.
  
*FI_THREAD_ENDPOINT*
: The endpoint threading model is similar to FI_THREAD_FID, but with
  the added restriction that serialization is required when accessing
  the same endpoint, even if multiple transmit and receive contexts are
  used.  Conceptualy, FI_THREAD_ENDPOINT maps well to providers that
  implement fabric services in hardware but use a single command
  queue to access different data flows.

*FI_THREAD_COMPLETION*
  The completion threading model is intended for providers that make use
  of manual progress.  Applications must serialize access to all objects
  that are associated through the use of having a shared completion
  structure.  This includes endpoint, completion queue, counter, wait set,
  and poll set objects.
  
  For example, threads must serialize access to an endpoint and its
  bound completion queue(s) and/or counters.  Access to endpoints that
  share the same completion queue must also be serialized.
  
  The use of FI_THREAD_COMPLETION can increase parallelism over
  FI_THREAD_SAFE, but requires the use of isolated resources.

*FI_THREAD_DOMAIN*
: A domain serialization model requires applications to serialize
  access to all objects belonging to a domain.

## Progress Models (control_progress / data_progress)

Progress is the ability of the underlying implementation to complete
processing of an asynchronous request.  In many cases, the processing
of an asynchronous request requires the use of the host processor.
For example, a received message may need to be matched with the
correct buffer, or a timed out request may need to be retransmitted.
For performance reasons, it may be undesirable for the provider to
allocate a thread for this purpose, which will compete with the
application threads.

Control progress indicates the method that the provider uses to make
progress on asynchronous control operations.  Control operations are
functions which do not directly involve the transfer of application
data between endpoints.  They include address vector, memory
registration, and connection management routines.

Data progress indicates the method that the provider uses to make
progress on data transfer operations.  This includes message queue,
RMA, tagged messaging, and atomic operations, along with their
completion processing.

To balance between performance and ease of use, two progress models
are defined.

*FI_PROGRESS_UNSPEC*
: This value indicates that no progress model has been defined.  It
  may be used on input hints to the fi_getinfo call.

*FI_PROGRESS_AUTO*
: This progress model indicates that the provider will make forward
  progress on an asynchronous operation without further intervention
  by the application.  When FI_PROGRESS_AUTO is provided as output to
  fi_getinfo in the absence of any progress hints, it often indicates
  that the desired functionality is implemented by the provider
  hardware or is a standard service of the operating system.

  All providers are required to support FI_PROGRESS_AUTO.  However, if
  a provider does not natively support automatic progress, forcing the
  use of FI_PROGRESS_AUTO may result in threads being allocated below
  the fabric interfaces.

*FI_PROGRESS_MANUAL*
: This progress model indicates that the provider requires the use of
  an application thread to complete an asynchronous request.  When
  manual progress is set, the provider will attempt to advance an
  asynchronous operation forward when the application attempts to
  wait on or read an event queue, completion queue, or counter
  where the completed operation will be reported.  Progress also
  occurs when the application processes a poll or wait set that
  has been associated with the event or completion queue.

  Only wait operations defined by the fabric interface will result in
  an operation progressing.  Operating system or external wait
  functions, such as select, poll, or pthread routines, cannot.

## Resource Management (resource_mgmt)

Resource management (RM) is provider and protocol support to protect
against overrunning local and remote resources.  This includes
local and remote transmit contexts, receive contexts, completion
queues, and source and target data buffers.

When enabled, applications are given some level of protection against
overrunning provider queues and local and remote data buffers.  Such
support may be built directly into the hardware and/or network
protocol, but may also require that checks be enabled in the provider
software.  By disabling resource management, an application assumes
all responsibility for preventing queue and buffer overruns, but doing
so may allow a provider to eliminate internal synchronization calls,
such as atomic variables or locks.

It should be noted that even if resource management is disabled, the
provider implementation and protocol may still provide some level of
protection against overruns.  However, such protection is not guaranteed.
The following values for resource management are defined.

*FI_RM_UNSPEC*
: This value indicates that no resource management model has been defined.
  It may be used on input hints to the fi_getinfo call.

*FI_RM_DISABLED*
: The provider is free to select an implementation and protocol that does
  not protect against resource overruns.  The application is responsible
  for resource protection.

*FI_RM_ENABLED*
: Resource management is enabled for this provider domain.

The behavior of the various resource management options depends on whether
the endpoint is reliable or unreliable, as well as provider and protocol
specific implementation details, as shown in the following tables.

| Resource | Unrel EP-RM Disabled| Unrel EP-RM Enabled | Rel EP-RM Disabled | Rel EP-RM Enabled |
|:--------:|:-------------------:|:-------------------:|:------------------:|:-----------------:|
| Tx             | error            | EAGAIN           | error             | EAGAIN             |
| Rx             | error            | EAGAIN           | error             | EAGAIN             |
| Tx CQ          | error            | EAGAIN           | error             | EAGAIN             |
| Rx CQ          | error            | EAGAIN or drop   | error             | EAGAIN or retry    |
| Unmatched Recv | buffered or drop | buffered or drop | buffered or error | buffered or retry  |
| Recv Overrun   | truncate or drop | truncate or drop | truncate or error | truncate or error  |
| Unmatched RMA  | not applicable   | not applicable   | error             | error              |
| RMA Overrun    | not applicable   | not applicable   | error             | error              |

The resource column indicates the resource being accessed by a data
transfer operation. Tx refers to the transmit context when a data
transfer operation posted.  Rx refers to the receive context when
receive data buffers are posted.  When RM is enabled, the
provider will ensure that space is available to accept the operation.
If space is not available, the operation will fail with -FI_EAGAIN.
If resource management is disabled, the application is responsible for
ensuring that there is space available before attempting to queue an
operation.

Tx CQ and Rx CQ refer to the completion queues associated with the
transmit and receive contexts, respectively.  When RM is disabled,
applications must take care to ensure that completion queues do not
get overrun.  This can be accomplished by sizing the CQs appropriately
or by deferring the posting of a data transfer operation unless CQ space
is available to store its completion.  When RM is enabled, providers
may use different mechanisms to prevent CQ overruns.  This includes
failing (returning -FI_EAGAIN) the posting of operations that could
result in CQ overruns, dropping received messages, or forcing requests
to be retried.

Unmatched receives and receive overruns deal with the processing of
messages that consume a receive buffers.  Unmatched receives references
incoming messages that are received by an endpoint, but do not have an
application data buffer to consume.  No buffers may be available at the
receive side, or buffers may available, but restricted from accepting
the received message (such as being associated with different tags).
Unmatched receives may be handled by protocol flow control, resulting
in the message being retried.  For unreliable endpoints, unmatched
messages are usually dropped, unless the provider can internally buffer
the data.  An error will usually occur on a reliable endpoint if received
data cannot be placed if RM is disabled, or the data cannot be received
with RM enabled after retries have been exhausted.

In some cases, buffering on the receive side may be available, but
insufficient space may have been provided to receive the full message
that was sent.  This is considered an error, however, rather than
failing the operation, a provider may instead truncate the message and
report the truncation to the app.

Unmatched RMA and RMA overruns deal with the processing of RMA and
atomic operations that access registered memory buffers directly.
RMA operations are not defined for unreliable endpoints.  For reliable
endpoints, unmatched RMA and RMA overruns are both treated as errors.

When a resource management error occurs on an endpoint, the endpoint is
transitioned into a disabled state.  Any operations which have not
already completed will fail and be discarded.  For unconnected endpoints,
the endpoint must be re-enabled before it will accept new data transfer
operations.  For connected endpoints, the connection is torn down and
must be re-established.

## MR Key Size

Size of the memory region remote access key, in bytes.  Applications
that request their own MR key must select a value within the range
specified by this value.

## CQ Data Size

The number of bytes that the provider supports for remote CQ data.
See the FI_REMOTE_CQ_DATA flag (fi_getinfo) for the use of remote CQ
data.

## Completion Queue Count (cq_cnt)

The total number of completion queues supported by the domain, relative
to any specified or default CQ attributes.  The cq_cnt value may be a
fixed value of the maximum number of CQs supported by the
underlying provider, or may be a dynamic value, based on the default
attributes of an allocated CQ, such as the CQ size and data format.

## Endpoint Count (ep_cnt)

The total number of endpoints supported by the domain, relative to any
specified or default endpoint attributes.  The ep_cnt value may be a
fixed value of the maximum number of endpoints supported by the
underlying provider, or may be a dynamic value, based on the default
attributes of an allocated endpoint, such as the endpoint capabilities
and size.  The endpoint count is the number of addressable endpoints
supported by the provider.

## Transmit Context Count (tx_ctx_cnt)

The number of outbound command queues optimally supported by the
provider.  For a low-level provider, this represents the number of
command queues to the hardware and/or the number of parallel transmit
engines effectively supported by the hardware and caches.
Applications which allocate more transmit contexts than this value
will end up sharing underlying resources.  By default, there is a
single transmit context associated with each endpoint, but in an
advanced usage model, an endpoint may be configured with multiple
transmit contexts.

## Receive Context Count (rx_ctx_cnt)

The number of inbound processing queues optimally supported by the
provider.  For a low-level provider, this represents the number
hardware queues that can be effectively utilized for processing
incoming packets.  Applications which allocate more receive contexts
than this value will end up sharing underlying resources.  By default,
a single receive context is associated with each endpoint, but in an
advanced usage model, an endpoint may be configured with multiple
receive contexts.

## Maximum Endpoint Transmit Context (max_ep_tx_ctx)

The maximum number of transmit contexts that may be associated with an
endpoint.

## Maximum Endpoint Receive Context (max_ep_rx_ctx)

The maximum number of receive contexts that may be associated with an
endpoint.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# NOTES

Users should call fi_close to release all resources allocated to the
fabric domain.

The following fabric resources are associated with access domains:
active endpoints, memory regions, completion event queues, and address
vectors.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_av`(3)](fi_av.3.html),
[`fi_eq`(3)](fi_eq.3.html),
[`fi_mr`(3)](fi_mr.3.html)
