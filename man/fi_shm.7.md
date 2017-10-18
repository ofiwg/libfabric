---
layout: page
title: fi_shm(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The SHM Fabric Provider

# OVERVIEW

The SHM provider is a complete provider that can be used on Linux
systems supporting shared memory and process_vm_readv/process_vm_writev
calls.  The provider is intended to provide high-performance communication
between processes on the same system.

# SUPPORTED FEATURES

This release contains an initial implementation of the SHM provider.  It
supports a minimal set of features useful for sending and
receiving datagram messages over an unreliable endpoint.

*Endpoint types*
: The provider supports only endpoint type *FI_EP_DGRAM*.

*Endpoint capabilities*
: The following data transfer interface is supported: *fi_msg*.

*Modes*
: The provider does not require the use of any mode bits.

*Progress*
: The UDP provider supports *FI_PROGRESS_MANUAL*.  Receive side data buffers are
  not modified outside of completion processing routines, and transmit
  operations require the receiver to process requests before being completed.

# LIMITATIONS

The SHM provider has hard-coded maximums for supported queue sizes and data
transfers.  These values are reflected in the related fabric attribute
structures

EPs must be bound to both RX and TX CQs.

No support for selective completions or multi-recv.

No support for counters.

# RUNTIME PARAMETERS

No runtime parameters are currently defined.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
