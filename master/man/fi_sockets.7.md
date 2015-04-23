---
layout: page
title: fi_sockets(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The Sockets Fabric Provider

# OVERVIEW

The sockets provider is a general purpose provider that can be used on any
system that supports TCP sockets.  The provider is not intended to provide
performance improvements over regular TCP sockets, but rather to allow
developers to write, test, and debug application code even on platforms
that do not have high-performance fabric hardware.  The sockets provider
supports all libfabric provider requirements and interfaces.

# SUPPORTED FEATURES

The sockets provider supports all the features defined for the libfabric API. 
Key features include:

*Endpoint types*
: The provider supports all endpoint types: *FI_EP_MSG*, *FI_EP_RDM*,
  and *FI_EP_DGRAM*.

*Endpoint capabilities*
: The following data transfer interface is supported for a all endpoint
  types: *fi_msg*.  Additionally, these interfaces are supported
  for reliable endpoints (*FI_EP_MSG* and *FI_EP_RDM*): *fi_tagged*,
  *fi_atomic*, and *fi_rma*.

*Modes*
: The sockets provider supports all operational modes including
  *FI_CONTEXT* and *FI_MSG_PREFIX*.

*Progress*
: Sockets provider supports both *FI_PROGRESS_AUTO* and *FI_PROGRESS_MANUAL*,
  with a default set to auto.  When progress is set to auto, a background
  thread runs to ensure that progress is made for asynchronous requests.

# LIMITATIONS

The sockets provider attempts to emulate the entire API set, including all
defined options.  In order to support development on a wide range of systems,
it is implemented over TCP sockets.  As a result, the performance numbers
are lower compared to other providers implemented over high-speed fabric, and
lower than what an application might see implementing to sockets directly.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
