---
layout: page
title: fi_tcp(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_tcp \- The msg sockets Fabric Provider

# OVERVIEW

The tcp provider can be used on any system that supports TCP sockets. The
provider is not intended to provide performance improvements over regular
TCP sockets, but rather to allow developers to write, test,and debug
application code even on platforms that do not have high-performance
fabric hardware.

# SUPPORTED FEATURES

The following features are supported

*Endpoint types*
: *FI_EP_MSG* is the only supported endpoint type. Reliable
datagram endpoint over TCP sockets can be achieved by layering RxM over
tcp provider.

*Endpoint capabilities*
: The tcp provider currently supports *FI_MSG*, *FI_RMA*

*Progress*
: Currently tcp provider supports only *FI_PROGRESS_MANUAL*

# LIMITATIONS

tcp provider is implemented over TCP sockets to emulate libfabric API. Hence
the performance is lower than what an application might see implementing to
sockets directly.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
