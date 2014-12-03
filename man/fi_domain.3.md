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

The fi_close call is used to release all resources associated with a
domain or interface.  All items associated with the opened domain must
be released prior to calling fi_close.

# DOMAIN ATTRIBUTES

The `fi_domain_attr` structure defines the set of attributes associated
with a domain.

{% highlight c %}
struct fi_domain_attr {
	struct fid_domain *domain;
	char              *name;
	enum fi_threading threading;
	enum fi_progress  control_progress;
	enum fi_progress  data_progress;
	size_t            mr_key_size;
	size_t            cq_data_size;
	size_t            cq_cnt;
	size_t            ep_cnt;
	size_t            tx_ctx_cnt;
	size_t            rx_ctx_cnt;
	size_t            max_ep_tx_ctx;
	size_t            max_ep_rx_ctx;
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
  may be used on input hints to the fi_getinfo call.

*FI_THREAD_SAFE*
: A thread safe serialization model allows a multi-threaded
  application to access any allocated resources through any interface
  without restriction.  All providers are required to support
  FI_THREAD_SAFE.

*FI_THREAD_PROGRESS*
: A progress serialization model requires applications to serialize
  access to provider resources and interfaces based on the progress
  model.  For providers with automatic progress, access to each
  endpoint must be serialized, and access to each event queue,
  counter, wait or poll set must be serialized.  Serialization is
  required only by threads accessing the same object.  For example,
  one thread may be initiating a data transfer on an endpoint, while
  another thread reads from an event queue associated with the
  endpoint.  Serialization to endpoint access is further limited to
  different endpoint data flows, if available.  Multiple threads may
  initiate transfers on the same endpoint if they reference different
  data flows.

  For providers with manual progress, applications must serialize
  their access to any object that is part of a single progress domain.
  A progress domain is any set of associated endpoints, event queues,
  counters, wait sets, and poll sets.  For instance, endpoints that
  share the same event queue or poll set belong to the same progress
  domain.  Applications that can allocate endpoint resources to
  specific threads can reduce provider locking by using
  FI_THREAD_PROGRESS.

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
function which do not directly involve the transfer of application
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
  asynchronous operation forward when the application invokes any
  event queue read or wait operation where the completion will be
  reported.  Progress also occurs when the application processes a
  poll or wait set.

  Only wait operations defined by the fabric interface will result in
  an operation progressing.  Operating system or external wait
  functions, such as select, poll, or pthread routines, cannot.

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
provider.  For a low-level provider, this represents the number
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
