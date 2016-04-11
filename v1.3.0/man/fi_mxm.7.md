---
layout: page
title: fi_mxm(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The MXM Fabric Provider

# OVERVIEW

The *mxm* provider runs over the MXM (Mellanox messaging) interface
that is currently supported by the Mellanox infiniband fabrics.
The *mxm* provider makes use of MXM tag matching API in order to
implement a limited set of the libfabric data transfer APIs, namely,
tagged message queue.

# LIMITATIONS

The *mxm* provider doesn't support all the features defined in the
libfabric API. Here are some of the limitations:

Endpoint types
: Only supported type:  *FI_RDM*

Endpoint capabilities
: Endpoints can support the only data transfer capability
  *FI_TAGGED*.


Modes
: *FI_CONTEXT* is required. That means, all the requests that generate
  completions must have a valid pointer to type *struct fi_context*
  passed as the operation context.

Threading
: The supported mode is FI_THREAD_DOMAIN, i.e. the *mxm* provider is not thread safe.


Unsupported features
: These features are unsupported: connection management, event queue, 
  scalable endpoint, passive endpoint, shared receive context,
  rma, atomics.

Mem tag format
: MXM library provides a tag matching interface with just 32 bits wide tag.
  Another 16 bits are available for matching via specifying an MXM MQ. Hence
  total maximum available matching bits equal 48. This is why the only allowed
  mem_tag_format is 0xFFFF00000000.


# RUNTIME PARAMETERS


# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
