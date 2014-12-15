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

fi_cq_read / fi_cq_readfrom / fi_cq_readerr
: Read a completion from a completion queue

fi_cq_write / fi_cq_writeerr
: Writes a completion or error event to a completion queue

fi_cq_sread / fi_cq_sreadfrom
: A synchronous (blocking) read that waits until a specified condition
  has been met before reading a completion from a completion queue.

fi_cq_strerror
: Converts provider specific error information into a printable string

# SYNOPSIS

{% highlight c %}
#include <rdma/fi_domain.h>

int fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
    struct fid_cq **cq, void *context);

int fi_close(struct fid *cq);

int fi_control(struct fid *cq, int command, void *arg);

int fi_cq_read(struct fid_cq *cq, void *buf, size_t count);

int fi_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
    fi_addr_t *src_addr);

int fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
    uint64_t flags);

int fi_cq_write(struct fid_cq *cq, const void *buf, size_t len);

int fi_cq_writeerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
    size_t len, uint64_t flags);

int fi_cq_sread(struct fid_cq *cq, void *buf, size_t count,
    const void *cond, int timeout);

int fi_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t count,
    fi_addr_t *src_addr, const void *cond, int timeout);

const char * fi_cq_strerror(struct fid_cq *cq, int prov_errno,
      const void *err_data, char *buf, size_t len);
{% endhighlight %}

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

{% highlight c %}
struct fi_cq_attr {
	size_t               size;      /* # entries for CQ */
	uint64_t             flags;     /* operation flags */
	enum fi_cq_format    format;    /* completion format */
	enum fi_wait_obj     wait_obj;  /* requested wait object */
	int                  signaling_vector; /* interrupt affinity */
	enum fi_cq_wait_cond wait_cond; /* wait condition format */
	struct fid_wait     *wait_set;  /* optional wait set */
};
{% endhighlight %}

*size*
: Specifies the minimum size of an event queue. A value of 0 indicates that
  the provider may choose a default value.

*flags*
: Flags that control the configuration of the CQ.

- *FI_WRITE*
: Indicates that the application requires support for inserting user
  events into the CQ.  If this flag is set, then the fi_cq_write and
  fi_cq_writeerr operations must be supported by the provider.  If the
  FI_WRITE flag is not set, then the application may not invoke
  fi_cq_write of fi_cq_writeerr.

- *FI_REMOTE_SIGNAL*
: If specified, this indicates that the CQ should only signal its wait
  object upon receiving a remote operation with FI_REMOTE_SIGNAL set,
  provided that all other wait conditions have been met.  The use of
  FI_REMOTE_SIGNAL may improve system utilization by deferring
  processing of an CQ until a remote endpoint has completed all
  necessary operations.  FI_REMOTE_SIGNAL should be treated as an
  optimization.  Providers are not required to wait until a remote
  operation with FI_REMOTE_SIGNAL is received before signaling a wait
  object associated with an CQ.

*format*
: Completion queues allow the application to select the amount of
  detail that it must store and report.  The format attribute allows
  the application to select one of several completion formats,
  indicating the structure of the data that the completion queue
  should return when read.  Supported formats and the structures that
  correspond to each are listed below.

- *FI_CQ_FORMAT_UNSPEC*
: If an unspecified format is requested, then the CQ will use a
  provider selected default format.

- *FI_CQ_FORMAT_CONTEXT*
: Provides only user specified context that was associated with the
  completion.

{% highlight c %}
struct fi_cq_entry {
	void     *op_context; /* operation context */
};
{% endhighlight %}

- *FI_CQ_FORMAT_MSG*
: Provides minimal data for processing completions, with expanded
  support for reporting information about received messages.

{% highlight c %}
struct fi_cq_msg_entry {
	void     *op_context; /* operation context */
	uint64_t flags;       /* completion flags */
	size_t   len;         /* size of received data */
};
{% endhighlight %}

- *FI_CQ_FORMAT_DATA*
: Provides data associated with a completion.  Includes support for
  received message length, remote EQ data, and multi-receive buffers.

{% highlight c %}
struct fi_cq_data_entry {
	void     *op_context; /* operation context */
	uint64_t flags;       /* completion flags */
	size_t   len;         /* size of received data */
	void     *buf;        /* receive data buffer */
	uint64_t data;        /* completion data */
};
{% endhighlight %}

- *FI_CQ_FORMAT_TAGGED*
: Expands completion data to include support for the tagged message
  interfaces.

{% highlight c %}
struct fi_cq_tagged_entry {
	void     *op_context; /* operation context */
	uint64_t flags;       /* completion flags */
	size_t   len;         /* size of received data */
	void     *buf;        /* receive data buffer */
	uint64_t data;        /* completion data */
	uint64_t tag;         /* received tag */
};
{% endhighlight %}

*wait_obj*
: CQ's may be associated with a specific wait object.  Wait objects
  allow applications to block until the wait object is signaled,
  indicating that a completion is available to be read.  Users may use
  fi_control to retrieve the underlying wait object associated with an
  CQ, in order to use it in other system calls.  The following values
  may be used to specify the type of wait object associated with an
  CQ: FI_WAIT_NONE, FI_WAIT_UNSPEC, FI_WAIT_SET, FI_WAIT_FD, and
  FI_WAIT_MUTEX_COND.

- *FI_WAIT_NONE*
: Used to indicate that the user will not block (wait) for completions
  on the CQ.  When FI_WAIT_NONE is specified, the application may not
  call fi_cq_sread or fi_cq_sreadfrom.

- *FI_WAIT_UNSPEC*
: Specifies that the user will only wait on the CQ using fabric
  interface calls, such as fi_cq_readcond or fi_cq_sreadfrom.  In this
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

*signaling_vector*
: Indicates which processor core interrupts associated with the EQ should
  target.

*wait_cond*
: By default, when a completion is inserted into an CQ that supports
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
queue.  The CQ must not be bound to any other resources prior to being
closed.  Any completions which remain on the CQ when it is closed are
lost.

## fi_control

The fi_control call is used to access provider or implementation
specific details of the completion queue.  Access to the CQ should be
serialized across all calls when fi_control is invoked, as it may
redirect the implementation of CQ operations.  The following control
commands are usable with an CQ.

*FI_GETWAIT (void **)*
: This command allows the user to retrieve the low-level wait object
  associated with the CQ.  The format of the wait-object is specified
  during CQ creation, through the CQ attributes.
  See fi_eq.3 for addition details using control with FI_GETWAIT.


## fi_cq_read / fi_cq_readfrom

The fi_cq_read and fi_cq_readfrom operations perform a non-blocking
read of completion data from the CQ.  The format of the completion
event is determined using the fi_cq_format option that was specified
when the CQ was opened.  Multiple completions may be retrieved from a
CQ in a single call.  The maximum number of entries to return is
limited to the specified count parameter, with the number of entries
successfully read from the CQ returned by the call.

The fi_cq_readfrom call allows the CQ to return source address
information to the user for any received data.  Source address data is
only available for those endpoints configured with FI_SOURCE
capability.  If fi_cq_readfrom is called on an endpoint for which
source addressing data is not available, the source address will be
set to FI_ADDR_NOTAVAIL.  The number of input src_addr entries must
the the same as the count parameter.

CQs are optimized to report operations which have completed
successfully.  Operations which fail are reported 'out of band'.  Such
operations are retrieved using the fi_cq_readerr function.  When an
operation that completes with an unexpected error is inserted into an
CQ, it is placed into a temporary error queue.  Attempting to read
from an CQ while an item is in the error queue results in an FI_EAVAIL
failure.  Applications may use this return code to determine when to
call fi_cq_readerr.

## fi_cq_sread / fi_cq_sreadfrom

The fi_cq_sread and fi_cq_sreadfrom calls are the blocking equivalent
operations to fi_cq_read and fi_cq_readfrom.  Their behavior is
similar to the non-blocking calls, with the exception that the calls
will not return until either a completion has been read from the CQ or
an error or timeout occurs.

## fi_cq_readerr

The read error function, fi_cq_readerr, retrieves information
regarding any asynchronous operation which has completed with an
unexpected error.  fi_cq_readerr is a non-blocking call, returning
immediately whether an error completion was found or not.

Error information is reported to the user through `struct
fi_cq_err_entry`.  The format of this structure is defined below.

{% highlight c %}
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
};
{% endhighlight %}

The general reason for the error is provided through the err field.
Provider specific error information may also be available through the
prov_errno and err_data fields.  The err_data field, if set, will
reference an internal buffer owned by the provider.  The contents of
the buffer will remain valid until a subsequent read call against the
CQ.  Users may call fi_cq_strerror to convert provider specific error
information into a printable string for debugging purposes.

## fi_cq_write / fi_cq_writeerr

The fi_cq_write and fi_cq_writeerr operations insert user-generated
completion entries into a completion queue.  fi_cq_write inserts
non-error events into the CQ.  The format of the fi_cq_write event
must be the same as the fi_cq_format attribute defined for the CQ when
it was created.  fi_cq_writeerr inserts error events into the CQ.  The
error event format is struct fi_cq_err_entry.  The number of entries
to insert into the CQ is determined by the len parameter.  Len must be
a multiple of the size of the event to insert.

User events inserted into a CQ with be associated with the source address
FI_ADDR_NOTAVAIL.

# RETURN VALUES

fi_cq_open
: Returns 0 on success.  On error, a negative value corresponding to
  fabric errno is returned.

fi_cq_read / fi_cq_readfrom / fi_cq_readerr
fi_cq_sread / fi_cq_sreadfrom
: On success, returns the number of completion events retrieved from
  the completion queue.  On error, a negative value corresponding to
  fabric errno is returned. On timeout, -FI_ETIMEDOUT is returned.

fi_cq_write / fi_cq_writeerr
: On success, returns the number of bytes read from or written to the
  completion queue.  On error, a negative value corresponding to
  fabric errno is returned.

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
