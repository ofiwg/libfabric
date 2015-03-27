---
layout: page
title: fi_provider(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

Fabric Interface Providers

# OVERVIEW

Conceptually, a fabric provider may be viewed as a local hardware NIC
driver, though a provider is not limited by this definition.  The
first component of libfabric is a general purpose framework that is
capable of handling different types of fabric hardware.  All fabric
hardware devices and their software drivers are required to support
this framework.  Devices and the drivers that plug into the libfabric
framework are referred to as fabric providers, or simply providers.

This distribution of libfabric contains the following providers
(although more may be available via run-time plugins):

*PSM*
: High-speed InfiniBand networking from Intel.  See
  [`fi_psm`(7)](fi_psm.7.html) for more information.

*Sockets*
: A general purpose provider that can be used on any network that
  supports TCP/UDP sockets.  This provider is not intended to provide
  performance improvements over regular TCP/UDP sockets, but rather to
  allow developers to write, test, and debug application code even on
  platforms that do not have high-speed networking.
  See [`fi_sockets`(7)](fi_sockets.7.html) for more information.

*usNIC*
: Ultra low latency Ethernet networking over Cisco userspace VIC
  adapters.
  See [`fi_usnic`(7)](fi_usnic.7.html) for more information.

*Verbs*
: This provider uses the Linux Verbs API for network transport.
  Application performance is, obviously expected to be similar to that
  of the native Linux Verbs API.  Analogous to the Sockets provider,
  the Verbs provider is intended to enable developers to write, test,
  and debug application code on platforms that only have Linux
  Verbs-based networking.
  See [`fi_verbs`(7)](fi_verbs.7.html) for more information.

# PROVIDER REQUIREMENTS

Libfabric provides a general framework for supporting multiple types
of fabric objects and their related interfaces.  Fabric providers have
a large amount of flexibility in selecting which components they are
able and willing to support, based on specific hardware constraints.
To assist in the development of applications, libfabric specifies the
following requirements that must be met by any fabric provider, if
requested by an application.  (Note that the instantiation of a
specific fabric object is subject to application configuration
parameters and need not meet these requirements).

* A fabric provider must support at least one endpoint type.
* All endpoints must support the message queue data transfer
  interface.
* An endpoint that advertises support for a specific endpoint
  capability must support the corresponding data transfer interface.
* Endpoints must support operations to send and receive data for any
  data transfer operations that they support.
* Connectionless endpoints must support all relevant data
  transfer routines. (send / recv / write / read / etc.)
* Connectionless endpoints must support the CM interface getname.
* Connectionless endpoints that support multicast operations must
  support the CM interfaces join and leave.
* Connection-oriented interfaces must support the CM interfaces
  getname, getpeer, connect, listen, accept, reject, and shutdown.
* All endpoints must support all relevant 'msg' data transfer
  routines.  (sendmsg / recvmsg / writemsg / readmsg / etc.)
* Access domains must support opening address vector maps and tables.
* Address vectors associated with domains that may be identified using
  IP addresses must support FI_SOCKADDR_IN and FI_SOCKADDR_IN6 input
  formats.
* Address vectors must support FI_ADDR, FI_ADDR_INDEX, and FI_AV
  output formats.
* Access domains must support opening completion queues and counters.
* Completion queues must support the FI_CQ_FORMAT_CONTEXT and
  FI_CQ_FORMAT_MSG formats.
* Event queues associated with tagged message transfers must support
  the FI_CQ_FORMAT_TAGGED format.
* A provider is expected to be forward compatible, and must be able to
  be compiled against expanded `fi_xxx_ops` structures that define new
  functions added after the provider was written.  Any unknown
  functions must be set to NULL.

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

# SEE ALSO

[`fi_psm`(7)](fi_psm.7.html),
[`fi_sockets`(7)](fi_sockets.7.html),
[`fi_usnic`(7)](fi_usnic.7.html),
[`fi_verbs`(7)](fi_verbs.7.html),
