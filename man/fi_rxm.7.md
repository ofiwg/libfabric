---
layout: page
title: fi_rxm(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The RxM (RDM over MSG) Utility Provider

# OVERVIEW

The RxM provider is an utility provider that supports RDM endpoints
emulated over a base MSG provider.

# REQUIREMENTS

RxM provider requires the base MSG provider to support the following features:

  * MSG endpoints (FI_EP_MSG)

  * Shared receive contexts (FI_SHARED_CONTEXT)

  * RMA read/write (FI_RMA)

  * FI_OPT_CM_DATA_SIZE of atleast 24 bytes

RxM provider requires the app to support FI_LOCAL_MR mode (This requirement would
be removed in the future).

# SUPPORTED FEATURES

The RxM provider currently supports *FI_MSG* and *FI_TAGGED*
capabilities.

*Endpoint types*
: The provider supports only *FI_EP_RDM*.

*Endpoint capabilities*
: The following data transfer interface is supported: *FI_MSG*, *FI_TAGGED*.

*Progress*
: The RxM provider supports only *FI_PROGRESS_MANUAL* for now.

*Addressing Formats*
: FI_SOCKADDR, FI_SOCKADDR_IN

*Memory Region*
: FI_MR_BASIC

# LIMITATIONS

The RxM provider has hard-coded maximums for supported queue sizes and
data transfers. Some of these limits are set based on the selected
base MSG provider.

## Unsupported features

RxM provider does not support the following features:

  * op_flags: FI_INJECT, FI_COMPLETION, FI_CLAIM, FI_PEEK, FI_FENCE.

  * FI_ATOMIC

  * Scalable endpoints

  * Shared contexts

  * FABRIC_DIRECT

  * Multi recv

  * Counters

  * FI_MR_SCALABLE

  * Wait objects

# RUNTIME PARAMETERS

No runtime parameters are currently defined.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
