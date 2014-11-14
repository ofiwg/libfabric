---
layout: page
title: fi_version(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_version \- Version of the library interfaces

# SYNOPSIS

{% highlight c %}
#include <rdma/fabric.h>

uint32_t fi_version();

FI_MAJOR(version)

FI_MINOR(version)
{% endhighlight %}

# DESCRIPTION

This call returns the current version of the library interfaces.  The
version includes major and a minor numbers.  These may be extracted
from the returned value using the FI_MAJOR() and FI_MINOR() macros.

# NOTES

The library may support older versions of the interfaces.

# RETURN VALUE

Returns the current library version.  The upper 16-bits of the version
correspond to the major number, and the lower 16-bits correspond with
the minor number.

# SEE ALSO

[`fi_fabric`(7)](fi_fabric.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
