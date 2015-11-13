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

The GNI provider runs on Cray XC systems running CLE 5.2 UP02 or higher
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
  endpoint types: *fi_atomic*, *fi_msg*, *fi_rma*, *fi_tagged*.

*Completion events*
: The GNI provider supports *FI_CQ_FORMAT_CONTEXT*, *FI_CQ_FORMAT_MSG*,
  *FI_CQ_FORMAT_DATA* and *FI_CQ_FORMAT_TAGGED* with wait objects of type
  *FI_WAIT_NONE*, *FI_WAIT_FD*, and *FI_WAIT_MUTEX_COND*.

*Modes*
: The GNI provider does not require any operation modes.

*Progress*
: The GNI provider supports both *FI_PROGRESS_AUTO* and
  *FI_PROGRESS_MANUAL*, with a default set to auto.  When progress is
  set to auto, a background thread runs to ensure that progress is
  made for asynchronous requests.

# CAVEATS

*This is where we will put caveats or things that differ from the convention.*

# GNI EXTENSIONS

The GNI provider exposes low-level tuning parameters via a domain
*fi_open_ops* interface named *FI_GNI_DOMAIN_PARAMS*.  The flags
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
  `fi_msg` data transfers.

*GNI_RMA_RDMA_THRESHOLD*
: Threshold message size at which RDMA is used for `fi_rma` data
  transfers.

*GNI_CONN_TABLE_INITIAL_SIZE*
: Initial size of the internal table data structure used to manage
  connections.

*GNI_CONN_TABLE_MAX_SIZE*
: Maximum size of the internal table data structure used to manage
  connections.

*GNI_CONN_TABLE_STEP_SIZE*
: Step size for increasing the size of the internal table data
  structure used to manage internal  GNI connections.

*GNI_VC_ID_TABLE_CAPACITY*
: Size of the virtual channel (VC) table used for managing remote
  connections.

*GNI_MBOX_PAGE_SIZE*
: Page size for GNI SMSG mailbox allocations.

*GNI_MBOX_NUM_PER_SLAB*
: Number of GNI SMSG mailboxes per allocation slab.

*GNI_MBOX_MAX_CREDIT*
: Maximum number of credits per GNI SMSG mailbox.

*GNI_MBOX_MSG_MAX_SIZE*
:  Maximum size of GNI SMSG messages.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_open_ops`(3)](fi_open_ops.3.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

For more information on uGNI, see *Using the GNI and DMAPP APIs*
(S-2446-3103, Cray Inc.).  For more information on the GNI provider,
see *An Implementation of OFI libfabric in Support of Multithreaded
PGAS Solutions* (PGAS '15).

