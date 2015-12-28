---
layout: page
title: fi_verbs(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The Verbs Fabric Provider

# OVERVIEW

The verbs provider enables applications using OFI to be run over any verbs
hardware (Infiniband, iWarp, etc). It uses the Linux Verbs API for network
transport and provides a translation of OFI calls to appropriate verbs API calls.
It uses librdmacm for communication management and libibverbs for other control
and data transfer operations.

# SUPPORTED FEATURES

The verbs provider supports a subset of OFI features.

*Endpoint types*
: FI_EP_MSG, FI_EP_RDM (Experimental support for only FI_TAGGED interface).

*Endpoint capabilities*
: FI_MSG, FI_RMA, FI_ATOMIC.

*Modes*
: Verbs provider requires applications to support the following modes:
  * FI_LOCAL_MR for all applications.
  * FI_RX_CQ_DATA for applications that want to use RMA. Applications must
    take responsibility of posting receives for any incoming CQ data.
  * FI_CONTEXT for applications making uses of the experimental FI_EP_RDM capability.

*Progress*
: Verbs provider supports FI_PROGRESS_AUTO: Asynchronous operations make forward
  progress automatically.

*Operation flags*
: Verbs provider supports FI_INJECT, FI_COMPLETION, FI_REMOTE_CQ_DATA.

*Msg Ordering*
: Verbs provider support the following messaging ordering on the TX side:
  * Read after Read
  * Read after Write
  * Read after Send
  * Write after Write
  * Write after Send
  * Send after Write
  * Send after Send

# UNSUPPORTED FEATURES

*Control Interfaces*
: Counters and address vectors are not supported.

*Data transfer interfaces*
: Multi-receive is not supported.

*Endpoint features*
: Scalable endpoints and shared contexts are not supported. fi_cancel,
  fi_tx/rx_size_left and fi_alias operations are not supported.

*Others*
: Other unsupported features include resource management, polling.

# LIMITATIONS

*CQ*
: cq_readfrom operations are not supported.

*Memory Regions*
: Adding regions via s/g list is not supported. Generic fi_mr_regattr is not
  supported. No support for binding memory regions to a counter.

*Wait objects*
: Only FI_WAIT_FD wait object is supported. Wait sets are not supported.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
