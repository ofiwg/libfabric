---
layout: page
title: fi_poll(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_poll \- Polling and wait set operations

fi_poll_open / fi_close
: Open/close a polling set

fi_poll_add / fi_poll_del
: Add/remove a completion queue or counter to/from a poll set.

fi_poll
: Poll for progress and events across multiple completion queues
  and counters.

fi_wait_open / fi_close
: Open/close a wait set

fi_wait
: Waits for one or more wait objects in a set to be signaled.

fi_trywait
: Indicate when it is safe to block on wait objects using native OS calls.

fi_control
: Control wait set operation or attributes.

# SYNOPSIS

```c
#include <rdma/fi_domain.h>

int fi_poll_open(struct fid_domain *domain, struct fi_poll_attr *attr,
    struct fid_poll **pollset);

int fi_close(struct fid *pollset);

int fi_poll_add(struct fid_poll *pollset, struct fid *event_fid,
    uint64_t flags);

int fi_poll_del(struct fid_poll *pollset, struct fid *event_fid,
    uint64_t flags);

int fi_poll(struct fid_poll *pollset, void **context, int count);

int fi_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
    struct fid_wait **waitset);

int fi_close(struct fid *waitset);

int fi_wait(struct fid_wait *waitset, int timeout);

int fi_trywait(struct fid_fabric *fabric, struct fid **fids, size_t count);

int fi_control(struct fid *waitset, int command, void *arg);
```

# ARGUMENTS

*fabric*
: Fabric provider

*domain*
: Resource domain

*pollset*
: Event poll set

*waitset*
: Wait object set

*attr*
: Poll or wait set attributes

*context*
: On success, an array of user context values associated with
  completion queues or counters.

*fids*
: An array of fabric descriptors, each one associated with a native
  wait object.

*count*
: Number of entries in context or fids array.

*timeout*
: Time to wait for a signal, in milliseconds.

*command*
: Command of control operation to perform on the wait set.

*arg*
: Optional control argument.

# DESCRIPTION


## fi_poll_open

fi_poll_open creates a new polling set.  A poll set enables an
optimized method for progressing asynchronous operations across
multiple completion queues and counters and checking for their completions.

A poll set is defined with the following attributes.

```c
struct fi_poll_attr {
	uint64_t             flags;     /* operation flags */
};
```

*flags*
: Flags that set the default operation of the poll set.  The use of
  this field is reserved and must be set to 0 by the caller.

## fi_close

The fi_close call releases all resources associated with a poll set.
The poll set must not be associated with any other resources prior to
being closed, otherwise the call will return -FI_EBUSY.

## fi_poll_add

Associates a completion queue or counter with a poll set.

## fi_poll_del

Removes a completion queue or counter from a poll set.

## fi_poll

Progresses all completion queues and counters associated with a poll set
and checks for events.  If events might have occurred, contexts associated
with the completion queues and/or counters are returned.  Completion
queues will return their context if they are not empty.  The context
associated with a counter will be returned if the counter's success
value or error value have changed since the last time fi_poll, fi_cntr_set,
or fi_cntr_add were called.  The number of contexts is limited to the
size of the context array, indicated by the count parameter.

Note that fi_poll only indicates that events might be available.  In some
cases, providers may consume such events internally, to drive progress, for
example.  This can result in fi_poll returning false positives.  Applications
should drive their progress based on the results of reading events from a
completion queue or reading counter values.  The fi_poll function will always
return all completion queues and counters that do have new events.

## fi_wait_open

fi_wait_open allocates a new wait set.  A wait set enables an
optimized method of waiting for events across multiple completion queues
and counters.  Where possible, a wait set uses a single underlying
wait object that is signaled when a specified condition occurs on an
associated completion queue or counter.

The properties and behavior of a wait set are defined by struct
fi_wait_attr.

```c
struct fi_wait_attr {
	enum fi_wait_obj     wait_obj;  /* requested wait object */
	uint64_t             flags;     /* operation flags */
};
```

*wait_obj*
: Wait sets are associated with specific wait object(s).  Wait objects
  allow applications to block until the wait object is signaled,
  indicating that an event is available to be read.  The following
  values may be used to specify the type of wait object associated
  with a wait set: FI_WAIT_UNSPEC, FI_WAIT_FD, and FI_WAIT_MUTEX_COND.

- *FI_WAIT_UNSPEC*
: Specifies that the user will only wait on the wait set using
  fabric interface calls, such as fi_wait.  In this case, the
  underlying provider may select the most appropriate or highest
  performing wait object available, including custom wait mechanisms.
  Applications that select FI_WAIT_UNSPEC are not guaranteed to
  retrieve the underlying wait object.

- *FI_WAIT_FD*
: Indicates that the wait set should use file descriptor(s) as its wait
  mechanism. It may not always be possible for a wait set to be implemented
  using a single underlying file descriptor, but all wait objects will be file
  descriptors. File descriptor wait objects must be usable in the
  POSIX select(2), poll(2), and epoll(7) routines (if
  available). However, a provider may signal an FD wait object by
  marking it as readable or with an error.

- *FI_WAIT_MUTEX_COND*
: Specifies that the wait set should use a pthread mutex and cond
  variable as a wait object.

- *FI_WAIT_CRITSEC_COND*
: Windows specific.  Specifies that the EQ should use a critical
  section and condition variable as a wait object.

*flags*
: Flags that set the default operation of the wait set.  The use of
  this field is reserved and must be set to 0 by the caller.

## fi_close

The fi_close call releases all resources associated with a wait set.
The wait set must not be bound to any other opened resources prior to
being closed, otherwise the call will return -FI_EBUSY.

## fi_wait

Waits on a wait set until one or more of its underlying wait objects
is signaled.

## fi_trywait

The fi_trywait call was introduced in libfabric version 1.3.  The behavior
of using native wait objects without the use of fi_trywait is provider
specific and should be considered non-deterministic.

The fi_trywait() call is used in conjunction with native operating
system calls to block on wait objects, such as file descriptors.  The
application must call fi_trywait and obtain a return value of
FI_SUCCESS prior to blocking on a native wait object.  Failure to
do so may result in the wait object not being signaled, and the
application not observing the desired events.  The following
pseudo-code demonstrates the use of fi_trywait in conjunction with
the OS select(2) call.

```c
fi_control(&cq->fid, FI_GETWAIT, (void *) &fd);
FD_ZERO(&fds);
FD_SET(fd, &fds);

while (1) {
	if (fi_trywait(&cq, 1) == FI_SUCCESS)
		select(fd + 1, &fds, NULL, &fds, &timeout);

	do {
		ret = fi_cq_read(cq, &comp, 1);
	} while (ret > 0);
}
```

fi_trywait() will return FI_SUCCESS if it is safe to block on the wait object(s)
corresponding to the fabric descriptor(s), or -FI_EAGAIN if there are
events queued on the fabric descriptor or if blocking could hang the
application.

The call takes an array of fabric descriptors.  For each wait object
that will be passed to the native wait routine, the corresponding
fabric descriptor should first be passed to fi_trywait.  All fabric
descriptors passed into a single fi_trywait call must make use of the
same underlying wait object type.

The following types of fabric descriptors may be passed into fi_trywait:
event queues, completion queues, counters, and wait sets.  Applications
that wish to use native wait calls should select specific wait objects
when allocating such resources.  For example, by setting the item's
creation attribute wait_obj value to FI_WAIT_FD.

In the case the wait object to check belongs to a wait set, only
the wait set itself needs to be passed into fi_trywait.  The fabric
resources associated with the wait set do not.

On receiving a return value of -FI_EAGAIN from fi_trywait, an application
should read all queued completions and events, and call fi_trywait again
before attempting to block.  Applications can make use of a fabric
poll set to identify completion queues and counters that may require
processing.

## fi_control

The fi_control call is used to access provider or implementation specific
details of the wait set. Access to the wait set should be serialized across
all calls when fi_control is invoked, as it may redirect the implementation
of wait set operations. The following control commands are usable with a
wait set.

*FI_GETWAIT (void \*\*)*
: This command allows the user to retrieve the low-level wait object
  associated with the wait set. The format of the wait set is specified
  during wait set creation, through the wait set attributes. The fi_control
  arg parameter should be an address where a pointer to the returned wait
  object will be written. This should be an 'int *' for FI_WAIT_FD,
  or 'struct fi_mutex_cond' for FI_WAIT_MUTEX_COND. Support for FI_GETWAIT
  is provider specific and may fail if not supported or if the wait set is
  implemented using more than one wait object.

# RETURN VALUES

Returns FI_SUCCESS on success.  On error, a negative value corresponding to
fabric errno is returned.

Fabric errno values are defined in
`rdma/fi_errno.h`.

fi_poll
: On success, if events are available, returns the number of entries
  written to the context array.

# NOTES


# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_cntr`(3)](fi_cntr.3.html),
[`fi_eq`(3)](fi_eq.3.html)
