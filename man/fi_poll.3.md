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
: Add/remove an event queue or counter to/from a poll set.

fi_poll
: Poll for progress and events across multiple event queues.

fi_wait_open / fi_close
: Open/close a wait set

fi_wait
: Waits for one or more wait objects in a set to be signaled.

# SYNOPSIS

{% highlight c %}
#include <rdma/fi_domain.h>

int fi_poll_open(struct fid_domain *domain, struct fi_poll_attr *attr,
    struct fid_poll **pollset);

int fi_close(struct fid *pollset);

int fi_poll_add(struct fid_poll *pollset, struct fid *event_fid,
    uint64_t flags);

int fi_poll_del(struct fid_poll *pollset, struct fid *event_fid,
    uint64_t flags);

int fi_poll(struct fid_poll *pollset, void **context, int count);

int fi_wait_open(struct fid_domain *domain, struct fi_wait_attr *attr,
    struct fid_wait **waitset);

int fi_close(struct fid *waitset);

int fi_wait(struct fid_wait *waitset, int timeout);
{% endhighlight %}

# ARGUMENTS

*domain*
: Resource domain

*pollset*
: Event poll set

*waitset*
: Wait object set

*attr*
: Poll or wait set attributes

*context*
: On success, an array of user context values associated with an event
  queue or counter.

*count*
: Number of entries in context array.

*timeout*
: Time to wait for a signal, in milliseconds.

# DESCRIPTION


## fi_poll_open

fi_poll_open creates a new polling set.  A poll set enables an
optimized method for progressing asynchronous operations across
multiple event queues and counters and checking for their completions.

A poll set is defined with the following attributes.

{% highlight c %}
struct fi_poll_attr {
	uint64_t             flags;     /* operation flags */
};
{% endhighlight %}

*flags*
: Flags that set the default operation of the poll set.  The use of
  this field is reserved and must be set to 0 by the caller.

## fi_close

The fi_close call releases all resources associated with a poll set.
The poll set must not be associated with any other resources prior to
being closed.

## fi_poll_add

Associates an event queue or counter with a poll set.

## fi_poll_del

Removes an event queue or counter from a poll set.

## fi_poll

Progresses all event queues and counters associated with a poll set
and checks for events.  If events has occurred, contexts associated
with the event queues and/or counters are returned.  The number of
contexts is limited to the size of the context array, indicated by the
count parameter.

## fi_wait_open

fi_wait_open allocates a new wait set.  A wait set enables an
optimized method of waiting for events across multiple event queues
and counters.  Where possible, a wait set uses a single underlying
wait object that is signaled when a specified condition occurs on an
associated event queue or counter.

The properties and behavior of a wait set are defined by struct
fi_wait_attr.

{% highlight c %}
struct fi_wait_attr {
	enum fi_wait_obj     wait_obj;  /* requested wait object */
	uint64_t             flags;     /* operation flags */
};
{% endhighlight %}

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
: Indicates that the wait set should use a file descriptor as its wait
  mechanism.  A file descriptor wait object must be usable in select,
  poll, and epoll routines.  However, a provider may signal an FD wait
  object by marking it as readable, writable, or with an error.

- *FI_WAIT_MUTEX_COND*
: Specifies that the wait set should use a pthread mutex and cond
  variable as a wait object.

*flags*
: Flags that set the default operation of the wait set.  The use of
  this field is reserved and must be set to 0 by the caller.

## fi_close

The fi_close call releases all resources associated with a wait set.
The wait set must not be bound to any other opened resources prior to
being closed.

## fi_wait

Waits on a wait set until one or more of its underlying wait objects
is signaled.

# RETURN VALUES

Returns 0 on success.  On error, a negative value corresponding to
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
