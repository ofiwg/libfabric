---
layout: page
title: fi_rxd(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_rxd \- The RxD (RDM over DGRAM) Utility Provider

# OVERVIEW

The RxD provider is a utility provider that supports RDM endpoints
emulated over a base DGRAM provider.

# SUPPORTED FEATURES

The RxD provider currently supports *FI_MSG*, *FI_TAGGED* and *FI_RMA*
capabilities. It requires the base DGRAM provider to support *FI_MSG*
capabilities.

*Endpoint types*
: The provider supports only endpoint type *FI_EP_RDM*.

*Endpoint capabilities* : The following data transfer interface is
supported: *fi_msg*, *fi_tagged* and *fi_rma*.

*Modes*
: The provider does not require the use of any mode bits.

*Progress*
: The RxD provider supports both *FI_PROGRESS_AUTO* and *FI_PROGRESS_MANUAL*,
  with a default set to auto.  However, receive side data buffers are not
  modified outside of completion processing routines.

# LIMITATIONS

The RxD provider has hard-coded maximums for supported queue sizes and
data transfers. Some of these limits are set based on the selected
base DGRAM provider.

No support for multi-recv.

No support for counters.

The RxD provider is still under development and is not extensively
tested.

# RUNTIME PARAMETERS

No runtime parameters are currently defined.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
