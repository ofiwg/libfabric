---
layout: page
title: fi_atomic(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_atomic - Remote atomic functions

fi_atomic / fi_atomicv / fi_atomicmsg / fi_inject_atomic
: Initiates an atomic operation to remote memory

fi_fetch_atomic / fi_fetch_atomicv / fi_fetch_atomicmsg
: Initiates an atomic operation to remote memory, retrieving the initial
  value.

fi_compare_atomic / fi_compare_atomicv / fi_compare_atomicmsg
: Initiates an atomic compare-operation to remote memory, retrieving
  the initial value.

fi_atomicvalid / fi_fetch_atomicvalid / fi_compare_atomicvalid /
fi_query_atomic
: Indicates if a provider supports a specific atomic operation

# SYNOPSIS

```c
#include <rdma/fi_atomic.h>

ssize_t fi_atomic(struct fid_ep *ep, const void *buf,
	size_t count, void *desc, fi_addr_t dest_addr,
	uint64_t addr, uint64_t key,
	enum fi_datatype datatype, enum fi_op op, void *context);

ssize_t fi_atomicv(struct fid_ep *ep, const struct fi_ioc *iov,
	void **desc, size_t count, fi_addr_t dest_addr,
	uint64_t addr, uint64_t key,
	enum fi_datatype datatype, enum fi_op op, void *context);

ssize_t fi_atomicmsg(struct fid_ep *ep, const struct fi_msg_atomic *msg,
	uint64_t flags);

ssize_t fi_inject_atomic(struct fid_ep *ep, const void *buf,
	size_t count, fi_addr_t dest_addr,
	uint64_t addr, uint64_t key,
	enum fi_datatype datatype, enum fi_op op);

ssize_t fi_fetch_atomic(struct fid_ep *ep, const void *buf,
	size_t count, void *desc, void *result, void *result_desc,
	fi_addr_t dest_addr, uint64_t addr, uint64_t key,
	enum fi_datatype datatype, enum fi_op op, void *context);

ssize_t fi_fetch_atomicv(struct fid_ep *ep, const struct fi_ioc *iov,
	void **desc, size_t count, struct fi_ioc *resultv,
	void **result_desc, size_t result_count, fi_addr_t dest_addr,
	uint64_t addr, uint64_t key, enum fi_datatype datatype,
	enum fi_op op, void *context);

ssize_t fi_fetch_atomicmsg(struct fid_ep *ep,
	const struct fi_msg_atomic *msg, struct fi_ioc *resultv,
	void **result_desc, size_t result_count, uint64_t flags);

ssize_t fi_compare_atomic(struct fid_ep *ep, const void *buf,
	size_t count, void *desc, const void *compare,
	void *compare_desc, void *result, void *result_desc,
	fi_addr_t dest_addr, uint64_t addr, uint64_t key,
	enum fi_datatype datatype, enum fi_op op, void *context);

size_t fi_compare_atomicv(struct fid_ep *ep, const struct fi_ioc *iov,
       void **desc, size_t count, const struct fi_ioc *comparev,
       void **compare_desc, size_t compare_count, struct fi_ioc *resultv,
       void **result_desc, size_t result_count, fi_addr_t dest_addr,
       uint64_t addr, uint64_t key, enum fi_datatype datatype,
       enum fi_op op, void *context);

ssize_t fi_compare_atomicmsg(struct fid_ep *ep,
	const struct fi_msg_atomic *msg, const struct fi_ioc *comparev,
	void **compare_desc, size_t compare_count,
	struct fi_ioc *resultv, void **result_desc, size_t result_count,
	uint64_t flags);

int fi_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
    enum fi_op op, size_t *count);

int fi_fetch_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
    enum fi_op op, size_t *count);

int fi_compare_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
    enum fi_op op, size_t *count);

int fi_query_atomic(struct fid_domain *domain,
    enum fi_datatype datatype, enum fi_op op,
    struct fi_atomic_attr *attr, uint64_t flags);
```

# ARGUMENTS

*ep*
: Fabric endpoint on which to initiate atomic operation.

*buf*
: Local data buffer that specifies first operand of atomic operation

*iov / comparev / resultv*
: Vectored data buffer(s).

*count / compare_count / result_count*
: Count of vectored data entries.  The number of elements referenced,
  where each element is the indicated datatype.

*addr*
: Address of remote memory to access.

*key*
: Protection key associated with the remote memory.

*datatype*
: Datatype associated with atomic operands

*op*
: Atomic operation to perform

*compare*
: Local compare buffer, containing comparison data.

*result*
: Local data buffer to store initial value of remote buffer

*desc / compare_desc / result_desc*
: Data descriptor associated with the local data buffer, local compare
  buffer, and local result buffer, respectively.

*dest_addr*
: Destination address for connectionless atomic operations.  Ignored for
  connected endpoints.

*msg*
: Message descriptor for atomic operations

*flags*
: Additional flags to apply for the atomic operation

*context*
: User specified pointer to associate with the operation.

# DESCRIPTION

Atomic transfers are used to read and update data located in remote
memory regions in an atomic fashion.  Conceptually, they are similar
to local atomic operations of a similar nature (e.g. atomic increment,
compare and swap, etc.).  Updates to remote data involve one of
several operations on the data, and act on specific types of data, as
listed below.  As such, atomic transfers have knowledge of the format
of the data being accessed.  A single atomic function may operate
across an array of data applying an atomic operation to each entry,
but the atomicity of an operation is limited to a single datatype or
entry.

## Atomic Data Types

Atomic functions may operate on one of the following identified data
types.  A given atomic function may support any datatype, subject to
provider implementation constraints.

*FI_INT8*
: Signed 8-bit integer.

*FI_UINT8*
: Unsigned 8-bit integer.

*FI_INT16*
: Signed 16-bit integer.

*FI_UINT16*
: Unsigned 16-bit integer.

*FI_INT32*
: Signed 32-bit integer.

*FI_UINT32*
: Unsigned 32-bit integer.

*FI_INT64*
: Signed 64-bit integer.

*FI_UINT64*
: Unsigned 64-bit integer.

*FI_FLOAT*
: A single-precision floating point value (IEEE 754).

*FI_DOUBLE*
: A double-precision floating point value (IEEE 754).

*FI_FLOAT_COMPLEX*
: An ordered pair of single-precision floating point values (IEEE
  754), with the first value representing the real portion of a
  complex number and the second representing the imaginary portion.

*FI_DOUBLE_COMPLEX*
: An ordered pair of double-precision floating point values (IEEE
  754), with the first value representing the real portion of a
  complex number and the second representing the imaginary portion.

*FI_LONG_DOUBLE*
: A double-extended precision floating point value (IEEE 754).
  Note that the size of a long double and number of bits used for
  precision is compiler, platform, and/or provider specific.
  Developers that use long double should ensure that libfabric is
  built using a long double format that is compatible with their
  application, and that format is supported by the provider.  The
  mechanism used for this validation is currently beyond the scope of
  the libfabric API.

*FI_LONG_DOUBLE_COMPLEX*
: An ordered pair of double-extended precision floating point values
  (IEEE 754), with the first value representing the real portion of
  a complex number and the second representing the imaginary
  portion.

## Atomic Operations

The following atomic operations are defined.  An atomic operation
often acts against a target value in the remote memory buffer and
source value provided with the atomic function.  It may also carry
source data to replace the target value in compare and swap
operations.  A conceptual description of each operation is provided.

*FI_MIN*
: Minimum

```c
if (buf[i] < addr[i])
    addr[i] = buf[i]
```

*FI_MAX*
: Maximum

```c
if (buf[i] > addr[i])
    addr[i] = buf[i]
```

*FI_SUM*
: Sum

```c
addr[i] = addr[i] + buf[i]
```

*FI_PROD*
: Product

```c
addr[i] = addr[i] * buf[i]
```

*FI_LOR*
: Logical OR

```c
addr[i] = (addr[i] || buf[i])
```

*FI_LAND*
: Logical AND

```c
addr[i] = (addr[i] && buf[i])
```

*FI_BOR*
: Bitwise OR

```c
addr[i] = addr[i] | buf[i]
```

*FI_BAND*
: Bitwise AND

```c
addr[i] = addr[i] & buf[i]
```

*FI_LXOR*
: Logical exclusive-OR (XOR)

```c
addr[i] = ((addr[i] && !buf[i]) || (!addr[i] && buf[i]))
```

*FI_BXOR*
: Bitwise exclusive-OR (XOR)

```c
addr[i] = addr[i] ^ buf[i]
```

*FI_ATOMIC_READ*
: Read data atomically

```c
result[i] = addr[i]
```

*FI_ATOMIC_WRITE*
: Write data atomically

```c
addr[i] = buf[i]
```

*FI_CSWAP*
: Compare values and if equal swap with data

```c
if (compare[i] == addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_NE*
: Compare values and if not equal swap with data

```c
if (compare[i] != addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_LE*
: Compare values and if less than or equal swap with data

```c
if (compare[i] <= addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_LT*
: Compare values and if less than swap with data

```c
if (compare[i] < addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_GE*
: Compare values and if greater than or equal swap with data

```c
if (compare[i] >= addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_GT*
: Compare values and if greater than swap with data

```c
if (compare[i] > addr[i])
    addr[i] = buf[i]
```

*FI_MSWAP*
: Swap masked bits with data

```c
addr[i] = (buf[i] & compare[i]) | (addr[i] & ~compare[i])
```

## Base Atomic Functions

The base atomic functions -- fi_atomic, fi_atomicv,
fi_atomicmsg -- are used to transmit data to a remote node, where the
specified atomic operation is performed against the target data.  The
result of a base atomic function is stored at the remote memory
region.  The main difference between atomic functions are the number
and type of parameters that they accept as input.  Otherwise, they
perform the same general function.

The call fi_atomic transfers the data contained in the user-specified
data buffer to a remote node.  For unconnected endpoints, the destination
endpoint is specified through the dest_addr parameter.  Unless
the endpoint has been configured differently, the data buffer passed
into fi_atomic must not be touched by the application until the
fi_atomic call completes asynchronously.  The target buffer of a base
atomic operation must allow for remote read an/or write access, as
appropriate.

The fi_atomicv call adds support for a scatter-gather list to
fi_atomic.  The fi_atomicv transfers the set of data buffers
referenced by the ioc parameter to the remote node for processing.

The fi_inject_atomic call is an optimized version of fi_atomic.  The
fi_inject_atomic function behaves as if the FI_INJECT transfer flag
were set, and FI_COMPLETION were not.  That is, the data buffer is
available for reuse immediately on returning from from
fi_inject_atomic, and no completion event will be generated for this
atomic.  The completion event will be suppressed even if the endpoint
has not been configured with FI_SELECTIVE_COMPLETION.  See the flags
discussion below for more details. The requested message size that
can be used with fi_inject_atomic is limited by inject_size.

The fi_atomicmsg call supports atomic functions over both connected
and unconnected endpoints, with the ability to control the atomic
operation per call through the use of flags.  The fi_atomicmsg
function takes a struct fi_msg_atomic as input.

```c
struct fi_msg_atomic {
	const struct fi_ioc *msg_iov; /* local scatter-gather array */
	void                **desc;   /* local access descriptors */
	size_t              iov_count;/* # elements in ioc */
	const void          *addr;    /* optional endpoint address */
	const struct fi_rma_ioc *rma_iov; /* remote SGL */
	size_t              rma_iov_count;/* # elements in remote SGL */
	enum fi_datatype    datatype; /* operand datatype */
	enum fi_op          op;       /* atomic operation */
	void                *context; /* user-defined context */
	uint64_t            data;     /* optional data */
};

struct fi_ioc {
	void		*addr;    /* local address */
	size_t		count;    /* # target operands */
};

struct fi_rma_ioc {
	uint64_t	addr;     /* target address */
	size_t		count;    /* # target operands */
	uint64_t	key;      /* access key */
};
```

The following list of atomic operations are usable with base
atomic operations: FI_MIN, FI_MAX, FI_SUM, FI_PROD,
FI_LOR, FI_LAND, FI_BOR, FI_BAND, FI_LXOR, FI_BXOR,
and FI_ATOMIC_WRITE.

## Fetch-Atomic Functions

The fetch atomic functions -- fi_fetch_atomic, fi_fetch_atomicv,
and fi_fetch atomicmsg -- behave similar to the
equivalent base atomic function.  The difference between the fetch and
base atomic calls are the fetch atomic routines return the initial
value that was stored at the target to the user.  The initial value is
read into the user provided result buffer.  The target buffer of
fetch-atomic operations must be enabled for remote read access.

The following list of atomic operations are usable with
fetch atomic operations: FI_MIN, FI_MAX, FI_SUM, FI_PROD,
FI_LOR, FI_LAND, FI_BOR, FI_BAND, FI_LXOR, FI_BXOR, FI_ATOMIC_READ,
and FI_ATOMIC_WRITE.

For FI_ATOMIC_READ operations, the source buffer operand (e.g.
fi_fetch_atomic buf parameter) is ignored and may be NULL.  The results
are written into the result buffer.

## Compare-Atomic Functions

The compare atomic functions -- fi_compare_atomic, fi_compare_atomicv,
and fi_compare atomicmsg -- are used for
operations that require comparing the target data against a value
before performing a swap operation.  The compare atomic functions
support: FI_CSWAP, FI_CSWAP_NE, FI_CSWAP_LE, FI_CSWAP_LT, FI_CSWAP_GE,
FI_CSWAP_GT, and FI_MSWAP.

## Atomic Valid Functions

The atomic valid functions -- fi_atomicvalid, fi_fetch_atomicvalid,
and fi_compare_atomicvalid --indicate which operations the local
provider supports.  Needed operations not supported by the provider
must be emulated by the application.  Each valid call corresponds to a
set of atomic functions.  fi_atomicvalid checks whether a provider
supports a specific base atomic operation for a given datatype and
operation.  fi_fetch_atomicvalid indicates if a provider supports a
specific fetch-atomic operation for a given datatype and operation.
And fi_compare_atomicvalid checks if a provider supports a specified
compare-atomic operation for a given datatype and operation.

If an operation is supported, an atomic valid call will return 0,
along with a count of atomic data units that a single function call
will operate on.

## Query Atomic Attributes

The fi_query_atomic call acts as an enhanced atomic valid operation (see
the atomic valid function definitions above).  It
is provided, in part, for future extensibility.  The query operation reports
which atomic operations are supported by the domain, for suitably configured
endpoints.

The behavior of fi_query_atomic is adjusted based on the flags parameter.
If flags is 0, then the operation reports the supported atomic attributes
for base atomic operations, similar to fi_atomicvalid for endpoints.  If
flags has the FI_FETCH_ATOMIC bit set, the operation behaves similar to
fi_fetch_atomicvalid.  Similarly, the flag bit FI_COMPARE_ATOMIC results in
query acting as fi_compare_atomicvalid.  The FI_FETCH_ATOMIC and
FI_COMPARE_ATOMIC bits may not both be set.

If the FI_TAGGED bit is set, the provider will indicate if it supports
atomic operations to tagged receive buffers.  The FI_TAGGED bit may be used
by itself, or in conjunction with the FI_FETCH_ATOMIC and FI_COMPARE_ATOMIC
flags.

The output of fi_query_atomic is struct fi_atomic_attr:

```c
struct fi_atomic_attr {
	size_t count;
	size_t size;
};
```

The count attribute field is as defined for the atomic valid calls.  The
size field indicates the size in bytes of the atomic datatype.

## Completions

Completed atomic operations are reported to the user through one or
more event collectors associated with the endpoint.  Users provide
context which are associated with each operation, and is returned to
the user as part of the event completion.  See fi_cq for completion
event details.

Updates to the target buffer of an atomic operation are visible to
processes running on the target system either after a completion has
been generated, or after the completion of an operation initiated
after the atomic call with a fencing operation occurring in between.
For example, the target process may be notified by the initiator
sending a message after the atomic call completes, or sending a fenced
message immediately after initiating the atomic operation.

# FLAGS

The fi_atomicmsg, fi_fetch_atomicmsg, and fi_compare_atomicmsg calls
allow the user to specify flags which can change the default data
transfer operation.  Flags specified with atomic message operations
override most flags previously configured with the endpoint, except
where noted (see fi_control).  The following list of flags are usable
with atomic message calls.

*FI_COMPLETION*
: Indicates that a completion entry should be generated for the
  specified operation.  The endpoint must be bound to a completion queue
  with FI_SELECTIVE_COMPLETION that corresponds to the specified operation,
  or this flag is ignored.

*FI_MORE*
: Indicates that the user has additional requests that will
  immediately be posted after the current call returns.  Use of this
  flag may improve performance by enabling the provider to optimize
  its access to the fabric hardware.

*FI_INJECT*
: Indicates that the outbound non-const data buffers (buf and compare
  parameters) should be returned to user immediately after the call
  returns, even if the operation is handled asynchronously.  This may
  require that the underlying provider implementation copy the data
  into a local buffer and transfer out of that buffer.  The use of
  output result buffers are not affected by this flag. This flag can only
  be used with messages smaller than inject_size.

*FI_FENCE*
: Applies to transmits.  Indicates that the requested operation, also
  known as the fenced operation, and any operation posted after the
  fenced operation will be deferred until all previous operations
  targeting the same peer endpoint have completed.  Operations posted
  after the fencing will see and/or replace the results of any
  operations initiated prior to the fenced operation.
  
  The ordering of operations starting at the posting of the fenced
  operation (inclusive) to the posting of a subsequent fenced operation
  (exclusive) is controlled by the endpoint's ordering semantics.

*FI_TAGGED*
: Specifies that the target of the atomic operation is a tagged receive
  buffer instead of an RMA buffer.  When a tagged buffer is the target
  memory region, the addr parameter is used as a 0-based byte offset into
  the tagged buffer, with the key parameter specifying the tag.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# ERRORS

*-FI_EAGAIN*
: See [`fi_msg`(3)](fi_msg.3.html) for a detailed description of handling
  FI_EAGAIN.

*-FI_EOPNOTSUPP*
: The requested atomic operation is not supported on this endpoint.

*-FI_EMSGSIZE*
: The number of atomic operations in a single request exceeds that
  supported by the underlying provider.

# NOTES

Atomic operations operate on an array of values of a specific data type.
Atomicity is only guaranteed for each data type operation, not across
the entire array.  The following pseudo-code demonstrates this operation
for 64-bit unsigned atomic write.  ATOMIC_WRITE_U64 is a platform
dependent macro that atomically writes 8 bytes to an aligned memory location.

```c
fi_atomic(ep, buf, count, NULL, dest_addr, addr, key,
	  FI_UINT64, FI_ATOMIC_WRITE, context)
{
	for (i = 1; i < count; i ++)
		ATOMIC_WRITE_U64(((uint64_t *) addr)[i],
				 ((uint64_t *) buf)[i]);
}
```

The number of array elements to operate on is specified through a count
parameter.  This must be between 1 and the maximum returned through the
relevant valid operation, inclusive.  The requested operation and data
type must also be valid for the given provider.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_cq`(3)](fi_cq.3.html),
[`fi_rma`(3)](fi_rma.3.html)
