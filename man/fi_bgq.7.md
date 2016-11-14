---
layout: page
title: fi_bgq(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The Blue Gene/Q Fabric Provider

# OVERVIEW

The bgq provider is a native implementation of the libfabric interfaces
that makes direct use of the unique hardware features such as the
Messaging Unit (MU), Base Address Table (BAT), and L2 Atomics.

The purpose of this provider is to demonstrate the scalability and
performance of libfabric, and to provide an "extreme scale"
development environment for applications and middleware using the
libfabric API.

# SUPPORTED FEATURES

The bgq provider supports most features defined for the libfabric API. 
Key features include:

*Endpoint types*
: The Blue Gene/Q hardware is connectionless and reliable. Therefore, the
  bgq provider only supports the *FI_EP_RDM* endpoint type.

*Endpoint capabilities*
: The following data transfer interfaces are supported: *FI_MSG*, *FI_RMA*, *FI_TAGGED*,
  *FI_MULTI_RECV*, *FI_READ*, *FI_WRITE*, *FI_SEND*, *FI_RECV*, *FI_REMOTE_READ*,
  and *FI_REMOTE_WRITE*.

*Modes*
: The bgq provider requires *FI_CONTEXT* and *FI_ASYNC_IOV*

*Progress*
: Only *FI_PROGRESS_AUTO* is supported, however support for
  *FI_PROGRESS_MANUAL* may be added in the future.  When progress is set
  to auto, a background thread runs to ensure that progress is made for
  asynchronous requests.

*Address vector*
: Only the *FI_AV_MAP* address vector format is supported, however support
  for *FI_AV_TABLE* may be added in the future.

*Memory registration*
: Only *FI_MR_SCALABLE* is supported, however use of this mode for hardware
  accelerated rma transfers is severely limited due to a mismatch between
  the libfabric API and the Blue Gene/Q hardware capabilities. Support for
  a non-standard version of *FI_MR_BASIC* may be added in the future
  which would enable hardware accelerated rma read, rma write, and
  network atomic operations.

# UNSUPPORTED FEATURES

The bgq provider does not support *shared contexts*, *multiple endpoints*, or
*scalable endpoints*, however support for all of these features may be added
in the future.

The *FI_ATOMIC* endpoint capability is disabled, however support may be added
in the future. Native hardware accelerated atomic operations will require
a non-standard version of *FI_MR_BASIC* that uses offset-based remote
addressing and will be limited to the operation-datatype combinations common
to the libfabric API and the Blue Gene/Q network hardware.

# LIMITATIONS

The bgq provider only supports *FABRIC_DIRECT*. The size of the fi_context
structure for *FI_CONTEXT* is too small to be useful. In the 'direct' mode the
bgq provider can re-define the struct fi_context to a larger size - currently
64 bytes which is the L1 cache size.

The fi_context structure for *FI_CONTEXT* must be aligned to 8 bytes. This requirement is because
the bgq provider will use MU network atomics to track completions and the memory
used with MU atomic operations must be aligned to 8 bytes. Unfortunately, the libfabric API
has no mechanism for applications to programmatically determine these alignment
requirements. Because unaligned MU atomics operations are a fatal error, the
bgq provider will assert on the alignment for "debug" builds (i.e., the '-DNDEBUG'
pre-processor flag is not specified).

The progress thread used for *FI_PROGRESS_AUTO* effectively limits the maximum
number of ranks-per-node to 32.

The definition of the memory region key size (mr_key_size) effectively limits
the maximum number of ranks-per-node to 2 for hardware accelerated rma transfers.
This is because each compute node has a Base Address Table (BAT) composed of
512 entries and must be shared among all processes on the node. The mr_key_size
reports the number of *bytes* for the size of the memory region key variable which
limits the maximum number of keys to 256 (2^8) even when the hardware could support
512 keys with 1 rank-per-node. At 4 ranks-per-node and higher the maximum
number of keys for each process would be less than 256 and forces the bgq provider
to report a key size of zero bytes. This effectively disables support for memory
regions in *FI_MR_SCALABLE* mode.

It is invalid to register memory at the base virtual address "0" with a
length of "UINTPTR_MAX" (or equivalent). The Blue Gene/Q hardware operates on
37-bit physical addresses and all virtual addresses specified in the libfabric
API, such as the location of source/destination data and remote memory locations,
must be converted to a physical address before use. A 64-bit virtual address
space will not fit into a 37-bit physical address space.

fi_trecvmsg() fnd fi_recvmsg() unctions do not support non-contiguous receives
and the iovec count must be 1. The fi_trecvv() and fi_recvv() functions are
currently not supported.

# RUNTIME PARAMETERS

No runtime parameters are currently defined.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
