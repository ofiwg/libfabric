---
layout: page
title: fi_mlx(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_mlx \- The MLX Fabric Provider

# OVERVIEW

The *mlx* provider runs over the UCX library
that is currently supported by the Mellanox infiniband fabrics.
The *mlx* provider makes use of UCX tag matching API in order to
implement a limited set of the libfabric data transfer APIs, namely,
tagged message queue.

Supported UCP API version: 1.0

# LIMITATIONS

The *mlx* provider doesn't support all the features defined in the
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
: The supported mode is FI_THREAD_DOMAIN, i.e. the *mlx* provider is not thread safe.


Unsupported features
: These features are unsupported: connection management, event queue, 
  scalable endpoint, passive endpoint, shared receive context,
  rma, atomics.


# RUNTIME PARAMETERS


# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
