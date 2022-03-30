---
layout: page
title: fi_peer(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_export_fid / fi_import_fid
: Share a fabric object between different providers or resources

struct fid_peer_cq
: A completion queue that may be shared between independent providers

# SYNOPSIS

```c
#include <rdma/fabric.h>
#include <rdma/fi_ext.h>

int fi_export_fid(struct fid *fid, uint64_t flags,
    struct fid **expfid, void *context);

int fi_import_fid(struct fid *fid, struct fid *expfid, uint64_t flags);

struct fid_peer_cq {
    struct fid fid;
    struct fi_ops_cq_write *export_ops;
    struct fi_ops_cq_progress *import_ops;
};
```

# ARGUMENTS

*fid*
: Returned fabric identifier for opened object.

*expfid*
: Exported fabric object that may be shared with another provider.

*flags*
: Control flags for the operation.

*context:
: User defined context that will be associated with a fabric object.

# DESCRIPTION

Functions defined in this man page are typically used by providers to
communicate with other providers, known as peer providers, or by other
libraries to communicate with the libfabric core, known as peer libraries.
Most middleware and applications should not need to access this functionality,
as the documentation mainly targets provider developers.

Peer providers are a way for independently developed providers to be used
together in a tight fashion, such that layering overhead and duplicate
provider functionality can be avoided.  Peer providers are linked by having
one provider export specific functionality to another.  This is done by
having one provider export a sharable fabric object (fid), which is imported
by one or more peer providers.

As an example, a provider which uses TCP to communicate with remote peers
may wish to use the shared memory provider to communicate with local peers.
To remove layering overhead, the TCP based provider may export its completion
queue and shared receive context and import those into the shared memory
provider.

The general mechanisms used to share fabric objects between peer providers are
similar, independent from the object being shared.  However, because the goal
of using peer providers is to avoid overhead, providers must be explicitly
written to support the peer provider mechanisms.

# PEER CQ

The peer CQ defines a mechanism by which a peer provider may insert completions
into the CQ owned by another provider.  This avoids the overhead of the
libfabric user from needing to access multiple CQs.

To setup a peer CQ, a provider creates a fid_peer_cq object, which links
back to the provider's actual fid_cq.  The fid_peer_cq object is then
imported by a peer provider.  The fid_peer_cq defines callbacks that the
providers use to communicate with each other.  The provider that allocates
the fid_peer_cq is known as the owner, with the other provider referred to
as the peer.  An owner may setup peer relationships with multiple providers.

Peer CQs are configured by the owner calling the peer's fi_cq_open() call.
The owner passes in the FI_PEER_IMPORT flag to fi_cq_open().  When
FI_PEER_IMPORT is specified, the context parameter passed
into fi_cq_open() must reference a struct fi_peer_cq_context.  Providers that
do not support peer CQs must fail the fi_cq_open() call with -FI_EINVAL
(indicating an invalid flag).  The fid_peer_cq referenced by struct
fi_peer_cq_context must remain valid until the peer's CQ is closed.

The data structures to support peer CQs are defined as follows:

```c
struct fi_ops_cq_owner {
    size_t  size;
    ssize_t (*write)(struct fid_peer_cq *cq, void *context, uint64_t flags,
        size_t len, void *buf, uint64_t data, uint64_t tag, fi_addr_t src);
    ssize_t (*writeerr)(struct fid_peer_cq *cq,
        const struct fi_cq_err_entry *err_entry);
};

struct fi_ops_cq_peer {
    size_t size;
    void (*progress)(struct fid_peer_cq *cq);
};

struct fid_peer_cq {
    struct fid fid;
    struct fi_ops_cq_owner *owner_ops;
    struct fi_ops_cq_peer *peer_ops;
};

struct fi_peer_cq_context {
    size_t size;
    struct fid_peer_cq *cq;
};
```
For struct fid_peer_cq, the owner initializes the fid and owner_ops
fields.  struct fi_ops_cq_owner is used by the peer to communicate
with the owning provider.

The size field in struct fi_ops_cq_peer is filled by the owner
and used as a versioning check by the peer.  However, the function
handlers are filled out by the peer and must be initialized before
returning from fi_cq_open().  These calls are used by the owner to
communicate with the peer.

Because the peer's CQ should not be directly accessed, functions to
read the peer's CQ (i.e. read, readerr, sread, etc.) should be set
to return -FI_ENOSYS.

## fi_ops_cq_owner::write()

This call directs the owner to insert new completions into the CQ.
The fi_cq_attr::format field, along with other related attributes,
determines which input parameters are valid.  Parameters that are not
reported as part of a completion are ignored by the owner, and should
be set to 0, NULL, or other appropriate value by the user.  For example,
if source addressing is not returned with a completion, then the src
parameter should be set to FI_ADDR_NOTAVAIL and ignored on input.

The owner is responsible for locking, event signaling, and handling CQ
overflow.  Data passed through the write callback is relative to the
user.  For example, the fi_addr_t is relative to the peer's AV.
The owner is responsible for converting the address if source
addressing is needed.

(TBD: should CQ overflow push back to the user for flow control?
Do we need backoff / resume callbacks in ops_cq_user?)

## fi_ops_cq_owner::writeerr()

The behavior of this call is similar to the write() ops.  It inserts
a completion indicating that a data transfer has failed into the CQ.

## fi_ops_cq_peer::progress()

This callback is invoked by the owner to drive progress in the peer
provider on endpoints owned by the peer.  The owner must support the
thread calling the progress() function invoking one of the owner
ops from within the progress() function.  For example, the peer calling
write() to insert a new completion.

## EXAMPLE PEER CQ SETUP

The above description defines the generic mechanism for sharing CQs
between providers.  This section outlines one possible implementation
to demonstrate the use of the APIs.  In the example, provider A
uses provider B as a peer for data transfers targeting endpoints
on the local node.

```
1. Provider A is configured to use provider B as a peer.  This may be coded
   into provider A or set through an environment variable.
2. The application calls:
   fi_cq_open(domain_a, attr, &cq_a, app_context)
3. Provider A allocates cq_a and automatically configures it to be used
   as a peer cq.
4. Provider A takes these steps:
   allocate peer_cq and reference cq_a
   set peer_cq_context->cq = peer_cq
   set attr_b.flags |= FI_PEER_IMPORT
   fi_cq_open(domain_b, attr_b, &cq_b, peer_cq_context)
5. Provider B allocates a cq, but configures it such that all completions
   are written to the peer_cq.  The cq ops to read from the cq are
   set to enosys calls.
8. Provider B inserts its own callbacks into the peer_cq object.  It
   creates a reference between the peer_cq object and its own cq.
```

# fi_export_fid / fi_import_fid

The fi_export_fid function is reserved for future use.

The fi_import_fid call may be used to import a fabric object created
and owned by the libfabric user.  This allows upper level libraries or
the application to override or define low-level libfabric behavior.
Details on specific uses of fi_import_fid are outside the scope of
this documentation.

# RETURN VALUE

Returns FI_SUCCESS on success. On error, a negative value corresponding to
fabric errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# SEE ALSO

[`fi_provider`(7)](fi_provider.7.html),
[`fi_provider`(3)](fi_provider.3.html),
[`fi_cq`(3)](fi_cq.3.html),

