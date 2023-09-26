---
layout: page
title: fi_poll(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_trywait
: Indicate when it is safe to block on wait objects using native OS calls.

fi_control
: Control fid attributes.

# SYNOPSIS

```c
#include <rdma/fi_domain.h>

int fi_trywait(struct fid_fabric *fabric, struct fid **fids, size_t count);

int fi_control(struct fid *fid, int command, void *arg);
```

# ARGUMENTS

*fabric*
: Fabric provider

*fids*
: An array of fabric descriptors, each one associated with a native
  wait object.

*count*
: Number of entries in context or fids array.

*command*
: Command of control operation to perform on the fid.

*arg*
: Optional control argument

# DESCRIPTION

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
details of a fids that support blocking calls, such as completion
queues, counters, and event queues.  Access to the fid should be
serialized across all calls when fi_control is invoked, as it may redirect
the implementation of wait set operations. The following control commands
are usable with a wait set or fid.

*FI_GETWAIT (void \*\*)*
: This command allows the user to retrieve the low-level wait object
  associated with a fid. The format of the wait object is specified
  during its creation, through the corresponding attributes. The fi_control
  arg parameter should be an address where a pointer to the returned wait
  object will be written. This should be an 'int *' for FI_WAIT_FD.
  Support for FI_GETWAIT is provider specific.

*FI_GETWAITOBJ (enum fi_wait_obj \*)*
: This command returns the type of wait object associated with a fid.

# RETURN VALUES

Returns FI_SUCCESS on success.  On error, a negative value corresponding to
fabric errno is returned.

Fabric errno values are defined in
`rdma/fi_errno.h`.

# NOTES

In many situations, blocking calls may need to wait on signals sent
to a number of file descriptors.  For example, this is the case for
socket based providers, such as tcp and udp, as well as utility providers
such as multi-rail.  For simplicity, when epoll is available, it can
be used to limit the number of file descriptors that an application
must monitor.  The use of epoll may also be required in order
to support FI_WAIT_FD.

However, in order to support waiting on multiple file descriptors on systems
where epoll support is not available, or where epoll performance may
negatively impact performance, FI_WAIT_POLLFD provides this mechanism.
A significant difference between using FI_WAIT_POLLFD versus FI_WAIT_FD
is the file descriptors to poll may change dynamically.
As an example, the file descriptors associated with a completion queues'
wait set may change as endpoint associations with the CQ are added and
removed.

Struct fi_wait_pollfd is used to retrieve all file descriptors for fids
using FI_WAIT_POLLFD to support blocking calls.

```c
struct fi_wait_pollfd {
    uint64_t      change_index;
    size_t        nfds;
    struct pollfd *fd;
};
```

*change_index*
: The change_index may be used to determine if there have been any changes
  to the file descriptor list.  Anytime a file descriptor is added, removed,
  or its events are updated, this field is incremented by the provider.
  Applications wishing to wait on file descriptors directly should cache
  the change_index value.  Before blocking on file descriptor events, the
  app should use fi_control() to retrieve the current change_index and
  compare that against its cached value.  If the values differ, then the
  app should update its file descriptor list prior to blocking.

*nfds*
: On input to fi_control(), this indicates the number of entries in the
  struct pollfd * array.  On output, this will be set to the number of
  entries needed to store the current number of file descriptors.  If
  the input value is smaller than the output value, fi_control() will
  return the error -FI_ETOOSMALL.  Note that setting nfds = 0 allows
  an efficient way of checking the change_index.

*fd*
: This points to an array of struct pollfd entries.  The number of entries
  is specified through the nfds field.  If the number of needed entries
  is less than or equal to the number of entries available, the struct
  pollfd array will be filled out with a list of file descriptors and
  corresponding events that can be used in the select(2) and poll(2)
  calls.

The change_index is updated only when the file descriptors associated with
the pollfd file set has changed.  Checking the change_index is an additional
step needed when working with FI_WAIT_POLLFD wait objects directly.  The use
of the fi_trywait() function is still required if accessing wait objects
directly.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_cntr`(3)](fi_cntr.3.html),
[`fi_eq`(3)](fi_eq.3.html)
