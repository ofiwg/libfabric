---
layout: page
title: fabric(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

Fabric Interface Library

# SYNOPSIS

{% highlight c %}
#include <rdma/fabric.h>
{% endhighlight %}

Libfabric is a high-performance fabric software library designed to
provide low-latency interfaces to fabric hardware.

# OVERVIEW

Libfabric provides 'process direct I/O' to application software communicating
across fabric software and hardware.  Process direct I/O, historically
referred to as RDMA, allows an application to directly access network
resources without operating system interventions.  Data transfers can
occur directly to and from application memory.

There are two components to the libfabric software:

*Fabric Providers*
: Conceptually, a fabric provider may be viewed as a local hardware
  NIC driver, though a provider is not limited by this definition.
  The first component of libfabric is a general purpose framework that
  is capable of handling different types of fabric hardware.  All
  fabric hardware devices and their software drivers are required to
  support this framework.  Devices and the drivers that plug into the
  libfabric framework are referred to as fabric providers, or simply
  providers.  Provider details may be found in
  [`fi_provider`(7)](fi_provider.7.html).

*Fabric Interfaces*
: The second component is a set of communication operations.
  Libfabric defines several sets of communication functions that
  providers can support.  It is not required that providers implement
  all the interfaces that are defined; however, providers clearly
  indicate which interfaces they do support.

# FABRIC INTERFACES

The fabric interfaces are designed such that they are cohesive and not
simply a union of disjoint interfaces.  The interfaces are logically
divided into two groups: control interfaces and communication
operations. The control interfaces are a common set of operations that
provide access to local communication resources, such as address
vectors and event queues.  The communication operations expose
particular models of communication and fabric functionality, such as
message queues, remote memory access, and atomic operations.
Communication operations are associated with fabric endpoints.

Applications will typically use the control interfaces to discover
local capabilities and allocate necessary resources.  They will then
allocate and configure a communication endpoint to send and receive
data, or perform other types of data transfers, with remote endpoints.

# CONTROL INTERFACES

The control interfaces APIs provide applications access to network
resources.  This involves listing all the interfaces available,
obtaining the capabilities of the interfaces and opening a provider.

*fi_getinfo - Fabric Information*
: The fi_getinfo call is the base call used to discover and request
  fabric services offered by the system.  Applications can use this
  call to indicate the type of communication that they desire.  The
  results from fi_getinfo, fi_info, are used to reserve and configure
  fabric resources.

  fi_getinfo returns a list of fi_info structures.  Each structure
  references a single fabric provider, indicating the interfaces that
  the provider supports, along with a named set of resources.  A
  fabric provider may include multiple fi_info structures in the
  returned list.

*fi_fabric - Fabric Domain*
: A fabric domain represents a collection of hardware and software
  resources that access a single physical or virtual network.  All
  network ports on a system that can communicate with each other
  through the fabric belong to the same fabric domain.  A fabric
  domain shares network addresses and can span multiple providers.
  libfabric supports systems connected to multiple fabrics.

*fi_domain - Access Domains*
: An access domain represents a single logical connection into a
  fabric.  It may map to a single physical or virtual NIC or a port.
  An access domain defines the boundary across which fabric resources
  may be associated.  Each access domain belongs to a single fabric
  domain.

*fi_endpoint - Fabric Endpoint*
: A fabric endpoint is a communication portal.  An endpoint may be
  either active or passive.  Passive endpoints are used to listen for
  connection requests.  Active endpoints can perform data transfers.
  Endpoints are configured with specific communication capabilities
  and data transfer interfaces.

*fi_eq - Event Queue*
: Event queues, are used to collect and report the completion of
  asynchronous operations and events.  Event queues report events
  that are not directly associated with data transfer operations.

*fi_cq - Completion Queue*
: Completion queues are high-performance event queues used to report
  the completion of data transfer operations.

*fi_cntr - Event Counters*
: Event counters are used to report the number of completed
  asynchronous operations.  Event counters are considered
  light-weight, in that a completion simply increments a counter,
  rather than placing an entry into an event queue.

*fi_mr - Memory Region*
: Memory regions describe application local memory buffers.  In order
  for fabric resources to access application memory, the application
  must first grant permission to the fabric provider by constructing a
  memory region.  Memory regions are required for specific types of
  data transfer operations, such as RMA transfers (see below).

*fi_av - Address Vector*
: Address vectors are used to map higher level addresses, such as IP
  addresses, which may be more natural for an application to use, into
  fabric specific addresses.  The use of address vectors allows
  providers to reduce the amount of memory required to maintain large
  address look-up tables, and eliminate expensive address resolution
  and look-up methods during data transfer operations.

# DATA TRANSFER INTERFACES

Fabric endpoints are associated with multiple data transfer
interfaces.  Each interface set is designed to support a specific
style of communication, with an endpoint allowing the different
interfaces to be used in conjunction.  The following data transfer
interfaces are defined by libfabric.

*fi_msg - Message Queue*
: Message queues expose a simple, message-based FIFO queue interface
  to the application.  Message data transfers allow applications to
  send and receive data with message boundaries being maintained.

*fi_tagged - Tagged Message Queues*
: Tagged message lists expose send/receive data transfer operations
  built on the concept of tagged messaging.  The tagged message queue
  is conceptually similar to standard message queues, but with the
  addition of 64-bit tags for each message.  Sent messages are matched
  with receive buffers that are tagged with a similar value.

*fi_rma - Remote Memory Access*
: RMA transfers are one-sided operations that read or write data
  directly to a remote memory region.  Other than defining the
  appropriate memory region, RMA operations do not require interaction
  at the target side for the data transfer to complete.

*fi_atomic - Atomic*
: Atomic operations can perform one of several operations on a remote
  memory region.  Atomic operations include well-known functionality,
  such as atomic-add and compare-and-swap, plus several other
  pre-defined calls.  Unlike other data transfer interfaces, atomic
  operations are aware of the data formatting at the target memory
  region.

# LOGGING INTERFACE

Logging is performed using the FI_ERR, FI_LOG, and FI_DEBUG macros.

## DEFINITIONS

{% highlight c %}
#define FI_ERR(prov_name, subsystem, ...)

#define FI_LOG(prov_name, prov, level, subsystem, ...)

#define FI_DEBUG(prov_name, subsystem, ...)

/* Subsystems */
enum {
	FI_FABRIC,
	FI_DOMAIN,
	FI_EP_CM,
	FI_EP_DM,
	FI_AV,
	FI_CQ,
	FI_EQ,
	FI_MR
};

/* Log Levels */
enum {
	FI_LOG_WARN = 0,
	FI_LOG_TRACE = 3,
	FI_LOG_INFO = 7
};

{% endhighlight %}

## ARGUMENTS
*prov_name*
: String representing the provider name.

*prov*
: Provider handle. In-tree providers will provide a pre-determined value in the
  first slot of the context member in the fi_provider struct returned by their
  initialization function. Dynamic providers will be assigned a handle that can
  be retrieved from the context member of their fi_provider struct. Dynamic
  providers should cleanup the returned handle in their cleanup function.

*level*
: Log level associated with log statement.

*subsystem*
: Subsystem being logged from.

## DESCRIPTION
*FI_ERR*
: Always logged.

*FI_LOG*
: Logged if the intended provider, log level, and subsystem parameters match
  the user supplied values.

*FI_DEBUG*
: Logged if configured with the --enable-debug flag.

# LOGGING CONTROL INTERFACE

Logging can be controlled using the FI_LOG_LEVEL, FI_LOG_PROV, and
FI_LOG_SUBSYSTEMS environment variables.

*FI_LOG_LEVEL*
: The FI_LOG_LEVEL environment variable has three valid values: "warn", "trace",
  and "info".

- *Warn*
: Warn is the least verbose setting and is intended for warnings. These
  will be logged regardless of the value of FI_LOG_LEVEL.

- *Trace*
: Trace is more verbose and is meant to include non-detailed output helpful to
  tracing program execution.

- *Info*
: Info is high traffic and meant for detailed output.

*FI_LOG_PROV*
: The FI_LOG_PROV environment variable enables or disables logging from
  specific providers. Providers can be enabled by listing them in a comma
  separated fashion. If the list begins with the '^' symbol, then the list will
  be negated. By default all providers are enabled.

  To enable logging from the psm and sockets provider:
	e.g. FI_LOG_PROV="psm,sockets"

  To enable logging from providers other than psm:
  	e.g. FI_LOG_PROV="^psm"

*FI_LOG_SUBSYSTEMS*
: The FI_LOG_SUBSYSTEMS environment variable enables or disables logging at the
  subsystem level. There are eight defined subsystems: "fabric", "domain",
  "ep_cm", "ep_dm", "av", "cq", "eq", and "mr". The syntax for enabling or
  disabling subsystems is the same as accepted by the FI_LOG_PROV environment
  variable.

# SEE ALSO

[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_av`(3)](fi_av.3.html),
[`fi_eq`(3)](fi_eq.3.html),
[`fi_cq`(3)](fi_cq.3.html),
[`fi_cntr`(3)](fi_cntr.3.html),
[`fi_mr`(3)](fi_mr.3.html)
