---
layout: page
title: fi_net(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_net \- The net Fabric Provider

# OVERVIEW

The net provider is a developmental fork of the tcp provider focused on
improving performance and scalability of HPC/AI applications, without
impacting the stability of the tcp provider.  It will be merged back into
the tcp provider at a later point in time.   See the
[`fi_tcp`(7)](fi_tcp.7.html) man page for additional details on the tcp
provider.

# SUPPORTED FEATURES

The following features are supported

*Endpoint types*
: *FI_EP_MSG* is supported by the net provider.  The net provider shares
  the same msg endpoint protocol as the tcp provider.

: *FI_EP_RDM* is supported directly by the net provider, unlike the tcp
  provider, which requires the use of the ofi_rxm utility provider.  The
  net provider extends its msg endpoint protocol to support rdm endpoints.
  As a result, the net provider's rdm endpoint protocol is not compatible
  with the ofi_rxm;tcp layered protocol.

*Endpoint capabilities*
: The net provider supports *FI_MSG*, *FI_RMA*, and *FI_TAGGED*.

*Shared Rx Context*
: The net provider supports shared receive context

*Multi recv buffers*
: The net provider supports multi recv buffers

# RUNTIME PARAMETERS

A full list of supported environment variables and their use can be obtained
using the fi_info application.  For example, "fi_info -g net" will show
all environment variables usable with the net provider.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_tcp`(7)](fi_tcp.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
