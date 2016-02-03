---
layout: page
title: fi_control(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_control \- Perform an operation on a fabric resource.

# SYNOPSIS

```c
#include <rdma/fabric.h>

int fi_control(struct fid *fid, int command, void *arg);
```


# ARGUMENTS

*fid*
: Fabric resource

*command*
: Operation to perform

*arg*
: Optional argument to the command

# DESCRIPTION

The fi_control operation is used to perform one or more operations on a
fabric resource.  Conceptually, fi_control is similar to the POSIX fcntl
routine.  The exact behavior of using fi_control depends on the fabric
resource being operated on, the specified command, and any provided
arguments for the command.  For specific details, see the fabric resource
specific help pages noted below.

# SEE ALSO

[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_cm`(3)](fi_cm.3.html),
[`fi_cntr`(3)](fi_cntr.3.html),
[`fi_cq`(3)](fi_cq.3.html),
[`fi_eq`(3)](fi_eq.3.html),
