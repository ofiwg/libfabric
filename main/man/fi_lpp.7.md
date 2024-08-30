---
layout: page
title: fi_lpp(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_lpp \- The LPP Fabric Provider

# OVERVIEW

The LPP provider runs on FabreX PCIe networks. FabreX provides high performance
RDMA capabilities which form the foundation of the LPP provider. Higher level
primitives are implemented at the libfabric and LPP kernel module (KLPP)
layers.

# SUPPORTED FEATURES

The LPP provider supports a subset of the libfabric API.
Key features include:

*Endpoint types*
: LPP supports the *FI_EP_RDM* endpoint type with resource management.

*Endpoint capabilities*
: LPP supports *FI_MSG*, *FI_RMA*, and *FI_TAGGED* message types.

*Additional capabilities*
: LPP supports the additional features: *FI_DIRECTED_RECV*, *FI_MULTI_RECV*,
*FI_INJECT*, *FI_DELIVERY_COMPLETE*

*Progress*
: LPP currently supports only *FI_PROGRESS_MANUAL*. Therefore user applications
  are required to poll for progress.

# LIMITATIONS

These features are unsupported: connection management, event queue, scalable
endpoint, passive endpoint, shared receive context, atomics.

# RUNTIME PARAMETERS

The LPP provider checks for the following environment variables -

*FI_LPP_DISABLE_OSBYPASS*
: A bool which disables direct userspace writes when set. This can be used as
  a debugging aid, however it will degrade performance.

*FI_LPP_MAX_WR_OSBYPASS_SIZE*
: Sets the maximum size for PIO when performing write RDMA transfers. Transfers
  at or below this size will use CPU copy, while transfers above it will use
  a DMA engine.

*FI_LPP_MAX_RD_OSBYPASS_SIZE*
: Sets the maximum size for PIO when performing read RDMA transfers. Transfers
  at or below this size will use CPU copy, while transfers above it will use
  a DMA engine.

*FI_LPP_CQ_OVERCOMMIT*
: A bool which allows operations to start that may overrun the CQ. Normally,
  when resource management is enabled, the LPP provider will attempt to throttle
  operations that might overrun the CQ. This parameter disables that behavior
  while leaving the remainder of the resource management features enabled.

*FI_LPP_DOMAIN_CLEANUP*
: A bool which controls whether closing a domain with active resources will
  cause those resources to be automatically closed. If true (the default), the
  LPP provider will automatically close the resources.

*FI_LPP_SYSTEM_MEMCPY*
: Use the memcpy implementation in the system libc rather than provider-specific
  memcpy.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
