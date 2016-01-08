---
layout: page
title: fi_gni(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The GNI Fabric Provider

# OVERVIEW

The GNI provider runs on Cray XC (TM) systems utilizing the user-space
Generic Network Interface (uGNI) which provides low-level access to
the Aries interconnect.  The Aries interconnect is designed for
low-latency one-sided messaging and also includes direct hardware
support for common atomic operations and optimized collectives.

# REQUIREMENTS

The GNI provider runs on Cray XC systems running CLE 5.2 UP03 or higher
using gcc version 4.9 or higher.

# SUPPORTED FEATURES

The GNI provider supports the following features defined for the
libfabric API:

*Endpoint types*
: The provider supports the *FI_EP_RDM* endpoint type.

*Address vectors*
: The provider implements both the *FI_AV_MAP* and *FI_AV_TABLE*
  address vector types.

*Data transfer operations*

: The following data transfer interfaces are supported for a all
  endpoint types: *FI_ATOMIC*, *FI_MSG*, *FI_RMA*, *FI_TAGGED*.  See
  DATA TRANSFER OPERATIONS below for more details.

*Completion events*
: The GNI provider supports *FI_CQ_FORMAT_CONTEXT*, *FI_CQ_FORMAT_MSG*,
  *FI_CQ_FORMAT_DATA* and *FI_CQ_FORMAT_TAGGED* with wait objects of type
  *FI_WAIT_NONE*, *FI_WAIT_FD*, and *FI_WAIT_MUTEX_COND*.

*Modes*
: The GNI provider does not require any operation modes.

*Progress*
: For control progress, the GNI provider supports both
  *FI_PROGRESS_AUTO* and *FI_PROGRESS_MANUAL*, with a default set to
  auto.  For data progress, *FI_PROGRESS_MANUAL* is supported.  When
  progress is set to auto, a background thread runs to ensure that
  progress is made for asynchronous requests.

# DATA TRANSFER OPERATIONS

## FI_ATOMIC

Currently, the GNI provider only supports atomic operations supported
directly by the Aries NIC.  These include operations on 32- and
64-bit, signed and unsigned integer and floating point values.
Specifically,

### Basic (fi_atomic, etc.)
- *FI_MIN*, *FI_MAX* (no unsigned)
- *FI_SUM* (no 64-bit floating point)
- *FI_BOR*, *FI_BAND*, *FI_BXOR* (no floating point)
- *FI_ATOMIC_WRITE*

### Fetching (fi_fetch_atomic, etc.)
- All of the basic operations as above
- FI_ATOMIC_READ

### Comparison (fi_compare_atomic, etc.)
- FI_CSWAP
- FI_MSWAP

## FI_MSG

All *FI_MSG* operations are supported.

## FI_RMA

All *FI_RMA* operations are supported.

## FI_TAGGED

All *FI_TAGGED* operations are supported except `fi_tinjectdata`.

# GNI EXTENSIONS

The GNI provider exposes low-level tuning parameters via a domain
`fi_open_ops` interface named *FI_GNI_DOMAIN_OPS_1*.  The flags
parameter is currently ignored.  The fi_open_ops function takes a
`struct fi_gni_ops_domain` parameter and populates it with the
following:

{% highlight c %}
struct fi_gni_ops_domain {
	int (*set_val)(struct fid *fid, dom_ops_val_t t, void *val);
	int (*get_val)(struct fid *fid, dom_ops_val_t t, void *val);
};
{% endhighlight %}

The `set_val` function sets the value of a given parameter; the
`get_val` function returns the current value.  The currently supported
values are:

*GNI_MSG_RENDEZVOUS_THRESHOLD*
: Threshold message size at which a rendezvous protocol is used for
  *FI_MSG* data transfers.  The value is of type uint32_t.

*GNI_RMA_RDMA_THRESHOLD*
: Threshold message size at which RDMA is used for *FI_RMA* data
  transfers.  The value is of type uint32_t.

*GNI_CONN_TABLE_INITIAL_SIZE*
: Initial size of the internal table data structure used to manage
  connections.  The value is of type uint32_t.

*GNI_CONN_TABLE_MAX_SIZE*
: Maximum size of the internal table data structure used to manage
  connections.  The value is of type uint32_t.

*GNI_CONN_TABLE_STEP_SIZE*
: Step size for increasing the size of the internal table data
  structure used to manage internal GNI connections.  The value is of
  type uint32_t.

*GNI_VC_ID_TABLE_CAPACITY*
: Size of the virtual channel (VC) table used for managing remote
  connections.  The value is of type uint32_t.

*GNI_MBOX_PAGE_SIZE*
: Page size for GNI SMSG mailbox allocations.  The value is of type
  uint32_t.

*GNI_MBOX_NUM_PER_SLAB*
: Number of GNI SMSG mailboxes per allocation slab.  The value is of
  type uint32_t.

*GNI_MBOX_MAX_CREDIT*
: Maximum number of credits per GNI SMSG mailbox.  The value is of
  type uint32_t.

*GNI_MBOX_MSG_MAX_SIZE*
: Maximum size of GNI SMSG messages.  The value is of type uint32_t.

*GNI_RX_CQ_SIZE*
: Recommended GNI receive CQ size.  The value is of type uint32_t.

*GNI_TX_CQ_SIZE*
: Recommended GNI transmit CQ size.  The value is of type uint32_t.

*GNI_MAX_RETRANSMITS*
: Maximum number of message retransmits before failure.  The value is
  of type uint32_t.

*GNI_MR_CACHE_LAZY_DEREG*
: Enable or disable lazy deregistration of memory.  The value is of
  type int32_t.



# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_open_ops`(3)](fi_open_ops.3.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

For more information on uGNI, see *Using the GNI and DMAPP APIs*
(S-2446-3103, Cray Inc.).  For more information on the GNI provider,
see *An Implementation of OFI libfabric in Support of Multithreaded
PGAS Solutions* (PGAS '15).

