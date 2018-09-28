---
layout: page
title: fi_rxm(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_rxm \- The RxM (RDM over MSG) Utility Provider

# OVERVIEW

The RxM provider (ofi_rxm) is an utility provider that supports RDM
endpoint emulated over MSG endpoint of a core provider.

# REQUIREMENTS

RxM provider requires the core provider to support the following features:

  * MSG endpoints (FI_EP_MSG)

  * RMA read/write (FI_RMA)

  * FI_OPT_CM_DATA_SIZE of at least 24 bytes

# SUPPORTED FEATURES

The RxM provider currently supports *FI_MSG*, *FI_TAGGED* and *FI_RMA* capabilities.

*Endpoint types*
: The provider supports only *FI_EP_RDM*.

*Endpoint capabilities*
: The following data transfer interface is supported: *FI_MSG*, *FI_TAGGED*, *FI_RMA*.

*Progress*
: The RxM provider supports *FI_PROGRESS_AUTO*.

*Addressing Formats*
: FI_SOCKADDR, FI_SOCKADDR_IN

*Memory Region*
: FI_MR_VIRT_ADDR, FI_MR_ALLOCATED, FI_MR_PROV_KEY MR mode bits would be
  required from the app in case the core provider requires it.

# LIMITATIONS

When using RxM provider, some limitations from the underlying MSG provider could also show
up. Please refer to the corresponding MSG provider man pages to find about those limitations.

## Unsupported features

RxM provider does not support the following features:

  * op_flags: FI_FENCE.

  * FI_ATOMIC

  * Scalable endpoints

  * Shared contexts

  * FABRIC_DIRECT

  * FI_MR_SCALABLE

  * Authorization keys

  * Application error data buffers

  * Multicast

  * FI_ADDR_STR, FI_SYNC_ERR

  * Reporting unknown source addr data as part of completions

  * Triggered operations

## Auto progress

When sending large messages, an app doing an sread or waiting on the CQ file descriptor
may not get a completion when reading the CQ after being woken up from the wait.
The app has to do sread or wait on the file descriptor again.

# RUNTIME PARAMETERS

The ofi_rxm provider checks for the following environment variables.

*FI_OFI_RXM_BUFFER_SIZE*
: Defines the transmit buffer size / inject size. Messages of size less than this
  would be transmitted via an eager protocol and those above would be transmitted
  via a rendezvous protocol. Transmit data would be copied up to this size
  (default: ~16k).

*FI_OFI_RXM_COMP_PER_PROGRESS*
: Defines the maximum number of MSG provider CQ entries (default: 1) that would
  be read per progress (RxM CQ read).

*FI_OFI_RXM_TX_SIZE*
: Defines default TX context size (default: 1024)

*FI_OFI_RXM_RX_SIZE*
: Defines default RX context size (default: 1024)

*FI_OFI_RXM_MSG_TX_SIZE*
: Defines FI_EP_MSG TX size that would be requested (default: 128).

*FI_OFI_RXM_MSG_RX_SIZE*
: Defines FI_EP_MSG RX size that would be requested (default: 128).

*FI_UNIVERSE_SIZE*
: Defines the expected number of ranks / peers an endpoint would communicate
with (default: 256).

# Tuning

## Bandwidth

To optimize for bandwidth, ensure you use higher values than default for
FI_OFI_RXM_TX_SIZE, FI_OFI_RXM_RX_SIZE, FI_OFI_RXM_MSG_TX_SIZE, FI_OFI_RXM_MSG_RX_SIZE
subject to memory limits of the system and the tx and rx sizes supported by the
MSG provider.

## Memory

To conserve memory, ensure FI_UNIVERSE_SIZE set to what is required. Similarly
check that FI_OFI_RXM_TX_SIZE, FI_OFI_RXM_RX_SIZE, FI_OFI_RXM_MSG_TX_SIZE and
FI_OFI_RXM_MSG_RX_SIZE env variables are set to only required values.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
