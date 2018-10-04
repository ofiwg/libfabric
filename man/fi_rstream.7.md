---
layout: page
title: fi_rstream(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_rstream

# OVERVIEW

The rstream provider supports stream messaging over
 message based RMA. It maps stream to message over
 a core RMA-based OFI provider. Only Endpoints and EQs
 are needed for connection start-up and messaging.

# SUPPORTED FEATURES

The rstream provider currently supports *FI_MSG* capabilities.

*Endpoint types*
: The provider supports only endpoint type *FI_EP_SOCK_STREAM*.

*Endpoint capabilities* : The following data transfer interface is
supported: *fi_msg*.

*Modes*
: The provider does not require the use of any mode bits but supports
  core providers that require FI_CONTEXT and FI_RX_CQ_DATA.

*Progress*
: The rstream provider only supports *FI_PROGRESS_MANUAL*.

# LIMITATIONS

The rstream provider is experimental and lacks performance validation and
 extensive testing. Currently the rstream provider is used to by the
 rsockets-OFI library as a ULP and hooks into the core provider verbs.
 There are default settings that limit the message stream (provider
 memory region size and CQ size). These can be modified by fi_setopt.

# SETTINGS

The *rstream* provider settings can be modified via fi_setopt on the
 endpoint (FI_OPT_ENDPOINT) along with the following parameters:

*FI_OPT_SEND_BUF_SIZE*
: Size of the send buffer. Default is 32KB.

*FI_OPT_RECV_BUF_SIZE*
: Size of the recv buffer. Default is 32KB.

*FI_OPT_TX_SIZE*
: Size of the send queue. Default is 384.

*FI_OPT_RX_SIZE*
: Size of the recv queue. Default is 384.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
