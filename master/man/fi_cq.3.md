---
layout: page
title: fi_cq(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_cq \- Completion queue operations

fi_cq_open / fi_close
: Open/close a completion queue

fi_control
: Control CQ operation or attributes.

fi_cq_read / fi_cq_readfrom / fi_cq_readerr
: Read a completion from a completion queue

fi_cq_sread / fi_cq_sreadfrom
: A synchronous (blocking) read that waits until a specified condition
  has been met before reading a completion from a completion queue.

fi_cq_signal
: Unblock any thread waiting in fi_cq_sread or fi_cq_sreadfrom.

fi_cq_strerror
: Converts provider specific error information into a printable string

# SYNOPSIS

```c
#include <rdma/fi_domain.h>

int fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
    struct fid_cq **cq, void *context);

int fi_close(struct fid *cq);

int fi_control(struct fid *cq, int command, void *arg);

ssize_t fi_cq_read(struct fid_cq *cq, void *buf, size_t count);

ssize_t fi_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
    fi_addr_t *src_addr);

ssize_t fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
    uint64_t flags);

ssize_t fi_cq_sread(struct fid_cq *cq, void *buf, size_t count,
    const void *cond, int timeout);

ssize_t fi_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t count,
    fi_addr_t *src_addr, const void *cond, int timeout);

int fi_cq_signal(struct fid_cq *cq);

const char * fi_cq_strerror(struct fid_cq *cq, int prov_errno,
      const void *err_data, char *buf, size_t len);
```

# ARGUMENTS

*domain*
: Open resource domain

*cq*
: Completion queue

*attr*
: Completion queue attributes

*context*
: User specified context associated with the completion queue.

*buf*
: For read calls, the data buffer to write completions into.  For
  write calls, a completion to insert into the completion queue.  For
  fi_cq_strerror, an optional buffer that receives printable error
  information.

*count*
: Number of CQ entries.

*len*
: Length of data buffer

*src_addr*
: Source address of a completed receive operation

*flags*
: Additional flags to apply to the operation

*command*
: Command of control operation to perform on CQ.

*arg*
: Optional control argument

*cond*
: Condition that must be met before a completion is generated

*timeout*
: Time in milliseconds to wait.  A negative value indicates infinite
  timeout.

*prov_errno*
: Provider specific error value

*err_data*
: Provider specific error data related to a completion

# DESCRIPTION

Completion queues are used to report events associated with data
transfers.  They are associated with message sends and receives, RMA,
atomic, tagged messages, and triggered events.  Reported events are
usually associated with a fabric endpoint, but may also refer to
memory regions used as the target of an RMA or atomic operation.

## fi_cq_open

fi_cq_open allocates a new completion queue.  Unlike event queues,
completion queues are associated with a resource domain and may be
offloaded entirely in provider hardware.

The properties and behavior of a completion queue are defined by
`struct fi_cq_attr`.

```c
struct fi_cq_attr {
	size_t               size;      /* # entries for CQ */
	uint64_t             flags;     /* operation flags */
	enum fi_cq_format    format;    /* completion format */
	enum fi_wait_obj     wait_obj;  /* requested wait object */
	int                  signaling_vector; /* interrupt affinity */
	enum fi_cq_wait_cond wait_cond; /* wait condition format */
	struct fid_wait     *wait_set;  /* optional wait set */
};
```

*size*
: Specifies the minimum size of a completion queue. A value of 0 indicates that
  the provider may choose a default value.

*flags*
: Flags that control the configuration of the CQ.

- *FI_AFFINITY*
: Indicates that the signaling_vector field (see below) is valid.

*format*
: Completion queues allow the application to select the amount of
  detail that it must store and report.  The format attribute allows
  the application to select one of several completion formats,
  indicating the structure of the data that the completion queue
  should return when read.  Supported formats and the structures that
  correspond to each are listed below.  The meaning of the CQ entry
  fields are defined in the _Completion Fields_ section.

- *FI_CQ_FORMAT_UNSPEC*
: If an unspecified format is requested, then the CQ will use a
  provider selected default format.

- *FI_CQ_FORMAT_CONTEXT*
: Provides only user specified context that was associated with the
  completion.

```c
struct fi_cq_entry {
	void     *op_context; /* operation context */
};
```

- *FI_CQ_FORMAT_MSG*
: Provides minimal data for processing completions, with expanded
  support for reporting information about received messages.

```c
struct fi_cq_msg_entry {
	void     *op_context; /* operation context */
	uint64_t flags;       /* completion flags */
	size_t   len;         /* size of received data */
};
```

- *FI_CQ_FORMAT_DATA*
: Provides data associated with a completion.  Includes support for
  received message length, remote CQ data, and multi-receive buffers.

```c
struct fi_cq_data_entry {
	void     *op_context; /* operation context */
	uint64_t flags;       /* completion flags */
	size_t   len;         /* size of received data */
	void     *buf;        /* receive data buffer */
	uint64_t data;        /* completion data */
};
```

- *FI_CQ_FORMAT_TAGGED*
: Expands completion data to include support for the tagged message
  interfaces.

```c
struct fi_cq_tagged_entry {
	void     *op_context; /* operation context */
	uint64_t flags;       /* completion flags */
	size_t   len;         /* size of received data */
	void     *buf;        /* receive data buffer */
	uint64_t data;        /* completion data */
	uint64_t tag;         /* received tag */
};
```

*wait_obj*
: CQ's may be associated with a specific wait object.  Wait objects
  allow applications to block until the wait object is signaled,
  indicating that a completion is available to be read.  Users may use
  fi_control to retrieve the underlying wait object associated with a
  CQ, in order to use it in other system calls.  The following values
  may be used to specify the type of wait object associated with a
  CQ: FI_WAIT_NONE, FI_WAIT_UNSPEC, FI_WAIT_SET, FI_WAIT_FD, and
  FI_WAIT_MUTEX_COND.  The default is FI_WAIT_NONE.

- *FI_WAIT_NONE*
: Used to indicate that the user will not block (wait) for completions
  on the CQ.  When FI_WAIT_NONE is specified, the application may not
  call fi_cq_sread or fi_cq_sreadfrom.

- *FI_WAIT_UNSPEC*
: Specifies that the user will only wait on the CQ using fabric
  interface calls, such as fi_cq_sread or fi_cq_sreadfrom.  In this
  case, the underlying provider may select the most appropriate or
  highest performing wait object available, including custom wait
  mechanisms.  Applications that select FI_WAIT_UNSPEC are not
  guaranteed to retrieve the underlying wait object.

- *FI_WAIT_SET*
: Indicates that the completion queue should use a wait set object to
  wait for completions.  If specified, the wait_set field must
  reference an existing wait set object.

- *FI_WAIT_FD*
: Indicates that the CQ should use a file descriptor as its wait
  mechanism.  A file descriptor wait object must be usable in select,
  poll, and epoll routines.  However, a provider may signal an FD wait
  object by marking it as readable, writable, or with an error.

- *FI_WAIT_MUTEX_COND*
: Specifies that the CQ should use a pthread mutex and cond variable
  as a wait object.

- *FI_WAIT_CRITSEC_COND*
: Windows specific.  Specifies that the CQ should use a critical
  section and condition variable as a wait object.

*signaling_vector*
: If the FI_AFFINITY flag is set, this indicates the logical cpu number
  (0..max cpu - 1) that interrupts associated with the CQ should target.
  This field should be treated as a hint to the provider and may be
  ignored if the provider does not support interrupt affinity.

*wait_cond*
: By default, when a completion is inserted into a CQ that supports
  blocking reads (fi_cq_sread/fi_cq_sreadfrom), the corresponding wait
  object is signaled.  Users may specify a condition that must first
  be met before the wait is satisfied.  This field indicates how the
  provider should interpret the cond field, which describes the
  condition needed to signal the wait object.

  A wait condition should be treated as an optimization.  Providers
  are not required to meet the requirements of the condition before
  signaling the wait object.  Applications should not rely on the
  condition necessarily being true when a blocking read call returns.

  If wait_cond is set to FI_CQ_COND_NONE, then no additional
  conditions are applied to the signaling of the CQ wait object, and
  the insertion of any new entry will trigger the wait condition.  If
  wait_cond is set to FI_CQ_COND_THRESHOLD, then the cond field is
  interpreted as a size_t threshold value.  The threshold indicates
  the number of entries that are to be queued before at the CQ before
  the wait is satisfied.

  This field is ignored if wait_obj is set to FI_WAIT_NONE.

*wait_set*
: If wait_obj is FI_WAIT_SET, this field references a wait object to
  which the completion queue should attach.  When an event is inserted
  into the completion queue, the corresponding wait set will be
  signaled if all necessary conditions are met.  The use of a wait_set
  enables an optimized method of waiting for events across multiple
  event and completion queues.  This field is ignored if wait_obj is
  not FI_WAIT_SET.

## fi_close

The fi_close call releases all resources associated with a completion
queue. Any completions which remain on the CQ when it is closed are
lost.

When closing the CQ, there must be no opened endpoints, transmit contexts, or
receive contexts associated with the CQ.  If resources are still associated
with the CQ when attempting to close, the call will return -FI_EBUSY.

## fi_control

The fi_control call is used to access provider or implementation
specific details of the completion queue.  Access to the CQ should be
serialized across all calls when fi_control is invoked, as it may
redirect the implementation of CQ operations.  The following control
commands are usable with a CQ.

*FI_GETWAIT (void \*\*)*
: This command allows the user to retrieve the low-level wait object
  associated with the CQ.  The format of the wait-object is specified
  during CQ creation, through the CQ attributes.  The fi_control arg
  parameter should be an address where a pointer to the returned wait
  object will be written.  See fi_eq.3 for addition details using
  fi_control with FI_GETWAIT.

## fi_cq_read

The fi_cq_read operation performs a non-blocking
read of completion data from the CQ.  The format of the completion
event is determined using the fi_cq_format option that was specified
when the CQ was opened.  Multiple completions may be retrieved from a
CQ in a single call.  The maximum number of entries to return is
limited to the specified count parameter, with the number of entries
successfully read from the CQ returned by the call.  (See return
values section below.)

CQs are optimized to report operations which have completed
successfully.  Operations which fail are reported 'out of band'.  Such
operations are retrieved using the fi_cq_readerr function.  When an
operation that has completed with an unexpected error is encountered,
it is placed into a temporary error queue.  Attempting to read
from a CQ while an item is in the error queue results in fi_cq_read
failing with a return code of -FI_EAVAIL.  Applications may use this
return code to determine when to call fi_cq_readerr.

## fi_cq_readfrom

The fi_cq_readfrom call behaves identical to fi_cq_read, with the
exception that it allows the CQ to return source address
information to the user for any received data.  Source address data is
only available for those endpoints configured with FI_SOURCE
capability.  If fi_cq_readfrom is called on an endpoint for which
source addressing data is not available, the source address will be
set to FI_ADDR_NOTAVAIL.  The number of input src_addr entries must
be the same as the count parameter.

Returned source addressing data is converted from the native address
used by the underlying fabric into an fi_addr_t, which may be used in
transmit operations.  Typically, returning fi_addr_t requires that
the source address be inserted into the address vector associated with the
receiving endpoint.  For endpoints allocated using the FI_SOURCE_ERR
capability, if the source address has not been inserted into the
address vector, fi_cq_readfrom will return -FI_EAVAIL.  The
completion will then be reported through fi_cq_readerr with error
code -FI_EADDRNOTAVAIL.  See fi_cq_readerr for details.

If FI_SOURCE is specified without FI_SOURCE_ERR, source addresses
which cannot be mapped to a local fi_addr_t will be reported as
FI_ADDR_NOTAVAIL.  The behavior is dependent on the type of address
vector in use.  For AVs of type FI_AV_MAP, source addresses may be
mapped directly to an fi_addr_t value, even if the source address
were not inserted into the AV.  This allows the provider to optimize
the reporting of the source fi_addr_t without the overhead of
verifying whether the address is in the AV.  If full address
validation is necessary, FI_SOURCE_ERR must be used.

## fi_cq_sread / fi_cq_sreadfrom

The fi_cq_sread and fi_cq_sreadfrom calls are the blocking equivalent
operations to fi_cq_read and fi_cq_readfrom.  Their behavior is
similar to the non-blocking calls, with the exception that the calls
will not return until either a completion has been read from the CQ or
an error or timeout occurs.

It is invalid for applications to call these functions if the CQ
has been configured with a wait object of FI_WAIT_NONE or FI_WAIT_SET.

## fi_cq_readerr

The read error function, fi_cq_readerr, retrieves information
regarding any asynchronous operation which has completed with an
unexpected error.  fi_cq_readerr is a non-blocking call, returning
immediately whether an error completion was found or not.

Error information is reported to the user through `struct
fi_cq_err_entry`.  The format of this structure is defined below.

```c
struct fi_cq_err_entry {
	void     *op_context; /* operation context */
	uint64_t flags;       /* completion flags */
	size_t   len;         /* size of received data */
	void     *buf;        /* receive data buffer */
	uint64_t data;        /* completion data */
	uint64_t tag;         /* message tag */
	size_t   olen;        /* overflow length */
	int      err;         /* positive error code */
	int      prov_errno;  /* provider error code */
	void    *err_data;    /*  error data */
	size_t   err_data_size; /* size of err_data */
};
```

The general reason for the error is provided through the err field.
Provider specific error information may also be available through the
prov_errno and err_data fields.  Users may call fi_cq_strerror to
convert provider specific error information into a printable string
for debugging purposes.  See field details below for more information
on the use of err_data and err_data_size.

Notable completion error codes are given below.

*FI_EADDRNOTAVAIL*
: This error code is used by CQs configured with FI_SOURCE_ERR to report
  completions for which a matching fi_addr_t source address could not
  be found.  An error code of FI_EADDRNOTAVAIL indicates that the data
  transfer was successfully received and processed, with the
  fi_cq_err_entry fields containing information about the completion.
  The err_data field will be set to the source address data.  The
  source address will be in the same format as specified through
  the fi_info addr_format field for the opened domain. This may be
  passed directly into an fi_av_insert call to add the source address
  to the address vector.

## fi_cq_signal

The fi_cq_signal call will unblock any thread waiting in fi_cq_sread
or fi_cq_sreadfrom.  This may be used to wake-up a thread
that is blocked waiting to read a completion operation.  The fi_cq_signal
operation is only available if the CQ was configured with a wait object.

# COMPLETION FIELDS

The CQ entry data structures share many of the same fields.  The meanings
of these fields are the same for all CQ entry structure formats.

*op_context*
: The operation context is the application specified context value that
  was provided with an asynchronous operation.  The op_context field is
  valid for all completions that are associated with an asynchronous
  operation.
  
  For completion events that are not associated with a posted operation,
  this field will be set to NULL.  This includes completions generated
  at the target in response to RMA write operations that carry CQ data
  (FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA flags set), when the FI_RX_CQ_DATA
  mode bit is not required.

*flags*
: This specifies flags associated with the completed operation.  The
  _Completion Flags_ section below lists valid flag values.  Flags are
  set for all relevant completions.

*len*
: This len field only applies to completed receive operations (e.g. fi_recv,
  fi_trecv, etc.).  It indicates the size of received _message_ data --
  i.e. how many data bytes were placed into the associated receive buffer by
  a corresponding fi_send/fi_tsend/et al call.  If an endpoint has
  been configured with the FI_MSG_PREFIX mode, the len also reflects the size
  of the prefix buffer.

*buf*
: The buf field is only valid for completed receive operations, and only
  applies when the receive buffer was posted with the FI_MULTI_RECV flag.
  In this case, buf points to the starting location where the receive
  data was placed.

*data*
: The data field is only valid if the FI_REMOTE_CQ_DATA completion flag
  is set, and only applies to receive completions.  If FI_REMOTE_CQ_DATA
  is set, this field will contain the completion data provided by the peer
  as part of their transmit request.  The completion data will be given
  in host byte order.

*tag*
: A tag applies only to received messages that occur using the tagged
  interfaces.  This field contains the tag that was included with the
  received message.  The tag will be in host byte order.

*olen*
: The olen field applies to received messages.  It is used to indicate
  that a received message has overrun the available buffer space and
  has been truncated.  The olen specifies the amount of data that did
  not fit into the available receive buffer and was discarded.

*err*
: This err code is a positive fabric errno associated with a completion.
  The err value indicates the general reason for an error, if one occurred.
  See fi_errno.3 for a list of possible error codes.

*prov_errno*
: On an error, prov_errno may contain a provider specific error code.  The
  use of this field and its meaning is provider specific.  It is intended
  to be used as a debugging aid.  See fi_cq_strerror for additional details
  on converting this error value into a human readable string.

*err_data*
: On an error, err_data may reference a provider specific amount of data
  associated with an error.  The use of this field and its meaning is
  provider specific.  It is intended to be used as a debugging aid.  See
  fi_cq_strerror for additional details on converting this error data into
  a human readable string.

*err_data_size*
: On input, err_data_size indicates the size of the err_data buffer in bytes.
  On output, err_data_size will be set to the number of bytes copied to the
  err_data buffer.  The err_data information is typically used with
  fi_cq_strerror to provide details about the type of error that occurred.

  For compatibility purposes, if err_data_size is 0 on input, or the fabric
  was opened with release < 1.5, err_data will be set to a data buffer
  owned by the provider.  The contents of the buffer will remain valid until a
  subsequent read call against the CQ.  Applications must serialize access
  to the CQ when processing errors to ensure that the buffer referenced by
  err_data does not change.

# COMPLETION FLAGS

Completion flags provide additional details regarding the completed
operation.  The following completion flags are defined.

*FI_SEND*
: Indicates that the completion was for a send operation.  This flag
  may be combined with an FI_MSG or FI_TAGGED flag.

*FI_RECV*
: Indicates that the completion was for a receive operation.  This flag
  may be combined with an FI_MSG or FI_TAGGED flag.

*FI_RMA*
: Indicates that an RMA operation completed.  This flag may be combined
  with an FI_READ, FI_WRITE, FI_REMOTE_READ, or FI_REMOTE_WRITE flag.

*FI_ATOMIC*
: Indicates that an atomic operation completed.  This flag may be combined
  with an FI_READ, FI_WRITE, FI_REMOTE_READ, or FI_REMOTE_WRITE flag.

*FI_MSG*
: Indicates that a message-based operation completed.  This flag may be
  combined with an FI_SEND or FI_RECV flag.

*FI_TAGGED*
: Indicates that a tagged message operation completed.  This flag may be
  combined with an FI_SEND or FI_RECV flag.

*FI_MULTICAST*
: Indicates that a multicast operation completed.  This flag may be combined
  with FI_MSG and relevant flags.  This flag is only guaranteed to be valid
  for received messages if the endpoint has been configured with FI_SOURCE.

*FI_READ*
: Indicates that a locally initiated RMA or atomic read operation has
  completed.  This flag may be combined with an FI_RMA or FI_ATOMIC flag.

*FI_WRITE*
: Indicates that a locally initiated RMA or atomic write operation has
  completed.  This flag may be combined with an FI_RMA or FI_ATOMIC flag.

*FI_REMOTE_READ*
: Indicates that a remotely initiated RMA or atomic read operation has
  completed.  This flag may be combined with an FI_RMA or FI_ATOMIC flag.

*FI_REMOTE_WRITE*
: Indicates that a remotely initiated RMA or atomic write operation has
  completed.  This flag may be combined with an FI_RMA or FI_ATOMIC flag.

*FI_REMOTE_CQ_DATA*
: This indicates that remote CQ data is available as part of the
  completion.

*FI_MULTI_RECV*
: This flag applies to receive buffers that were posted with the
  FI_MULTI_RECV flag set.  This completion flag indicates that the
  original receive buffer referenced by the completion has been
  consumed and was released by the provider.  Providers may set
  this flag on the last message that is received into the multi-
  recv buffer, or may generate a separate completion that indicates
  that the buffer has been released.

  Applications can distinguish between these two cases by examining
  the completion entry flags field.  If additional flags, such as
  FI_RECV, are set, the completion is associated with a received message.  In
  this case, the buf field will reference the location where the received
  message was placed into the multi-recv buffer.  Other fields in the
  completion entry will be determined based on the received message.
  If other flag bits are zero, the provider is reporting that the multi-recv
  buffer has been released, and the completion entry is not associated
  with a received message.

# NOTES

A completion queue must be bound to at least one enabled endpoint before any
operation such as fi_cq_read, fi_cq_readfrom, fi_cq_sread, fi_cq_sreadfrom etc.
can be called on it.

Completion flags may be suppressed if the FI_NOTIFY_FLAGS_ONLY mode bit
has been set.  When enabled, only the following flags are guaranteed to
be set in completion data when they are valid: FI_REMOTE_READ and
FI_REMOTE_WRITE (when FI_RMA_EVENT capability bit has been set),
FI_REMOTE_CQ_DATA, and FI_MULTI_RECV.

If a completion queue has been overrun, it will be placed into an 'overrun'
state.  Read operations will continue to return any valid, non-corrupted
completions, if available.  After all valid completions have been retrieved,
any attempt to read the CQ will result in it returning an FI_EOVERRUN error
event.  Overrun completion queues are considered fatal and may not be used
to report additional completions once the overrun occurs.

# RETURN VALUES

fi_cq_open / fi_cq_signal
: Returns 0 on success.  On error, a negative value corresponding to
  fabric errno is returned.

fi_cq_read / fi_cq_readfrom / fi_cq_readerr
fi_cq_sread / fi_cq_sreadfrom
: On success, returns the number of completion events retrieved from
  the completion queue.  On error, a negative value corresponding to
  fabric errno is returned.  If no completions are available to
  return from the CQ, -FI_EAGAIN will be returned.

fi_cq_strerror
: Returns a character string interpretation of the provider specific
  error returned with a completion.

Fabric errno values are defined in
`rdma/fi_errno.h`.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_eq`(3)](fi_eq.3.html),
[`fi_cntr`(3)](fi_cntr.3.html),
[`fi_poll`(3)](fi_poll.3.html)
