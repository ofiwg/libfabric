---
layout: page
title: fi_sockets(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The Sockets Fabric Provider

# OVERVIEW

A general purpose provider that can be used on any network that supports TCP sockets. This provider is not intended to provide performance improvements over regular TCP sockets, but rather to allow developers to write, test, and debug application code even on platforms that do not have high-speed networking. Sockets provider supports all the provider requirements and interfaces of libfabric. It can be viewed as a generic implementation of a fabric provider (for testing purpose) with the union of the interfaces of supported by libfabric even if there is no hardware to support all.


# SUPPORTED FEATURES

Sockets provider supports all the features defined in the libfabric API. Here are some key features:

Endpoint types
: Supports both connection oriented *FI_MSG* and connectionless *FI_RDM* and *FI_DGRAM*

Endpoint capabilities
: Endpoints can support any combination of primary data transfer capabilities *FI_TAGGED*, *FI_MSG*, *FI_ATOMICS*, and *FI_RMA*s, further refined by *FI_SEND*, *FI_RECV*, *FI_READ*, *FI_WRITE*, *FI_REMOTE_READ*, and *FI_REMOTE_WRITE*.*FI_MULTI_RECV* is supported for non-tagged message queue only.

Modes
: Sockets provider supports all operational modes including *FI_CONTEXT*, *FI_LOCAL_MR*, *FI_MR_PREFIX*, and *FI_PROV_MR_ATTR*.

Progress
: Sockets provider supports both *FI_PROGRESS_AUTO* and *FI_PROGRESS_MANUAL* but the default is set to auto progress. A background thread runs to ensure progress is made for asynchronous requests.

# LIMITATIONS

Sockets provider is implemented over TCP sockets and the performance numbers are lower compared to other providers which are implemented over high-speed fabric. But it is expected since the intention of sockets provider is not to have performance improvement but rather to serve as a reference implementation for testing.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
