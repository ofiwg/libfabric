---
layout: page
title: fi_fabric(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_fabric \- Fabric domain operations

fi_fabric / fi_close
: Open / close a fabric domain

fi_tostr
: Convert fabric attributes, flags, and capabilities to printable string

# SYNOPSIS

```c
#include <rdma/fabric.h>

int fi_fabric(struct fi_fabric_attr *attr,
    struct fid_fabric **fabric, void *context);

int fi_close(struct fid *fabric);

char * fi_tostr(const void *data, enum fi_type datatype);
```

# ARGUMENTS

*attr*
: Attributes of fabric to open.

*fabric*
: Fabric domain

*context*
: User specified context associated with the opened object.  This
  context is returned as part of any associated asynchronous event.

# DESCRIPTION

A fabric domain represents a collection of hardware and software
resources that access a single physical or virtual network.  All
network ports on a system that can communicate with each other through
their attached networks belong to the same fabric domain.  A fabric
domain shares network addresses and can span multiple providers.

## fi_fabric

Opens a fabric provider.  The attributes of the fabric provider are
specified through the open call, and may be obtained by calling
fi_getinfo.

## fi_close

The fi_close call is used to release all resources associated with a
fabric domain or interface.  All items associated with the opened
fabric must be released prior to calling fi_close.

## fi_tostr

Converts fabric interface attributes, capabilities, flags, and enum
values into a printable string.  The data parameter accepts a pointer
to the attribute or value(s) to display, with the datatype parameter
indicating the type of data referenced by the data parameter.  Valid
values for the datatype are listed below, along with the corresponding
datatype or field value.

*FI_TYPE_INFO*
: struct fi_info

*FI_TYPE_EP_TYPE*
: struct fi_info::type field

*FI_TYPE_EP_CAP*
: struct fi_info::ep_cap field

*FI_TYPE_OP_FLAGS*
: struct fi_info::op_flags field, or general uint64_t flags

*FI_TYPE_ADDR_FORMAT*
: struct fi_info::addr_format field

*FI_TYPE_TX_ATTR*
: struct fi_tx_attr

*FI_TYPE_RX_ATTR*
: struct fi_rx_attr

*FI_TYPE_EP_ATTR*
: struct fi_ep_attr

*FI_TYPE_DOMAIN_ATTR*
: struct fi_domain_attr

*FI_TYPE_FABRIC_ATTR*
: struct fi_fabric_attr

*FI_TYPE_DOMAIN_CAP*
: struct fi_info::domain_cap field

*FI_TYPE_THREADING*
: enum fi_threading

*FI_TYPE_PROGRESS*
: enum fi_progress

*FI_TYPE_PROTO*
: struct fi_ep_attr::protocol field

*FI_TYPE_MSG_ORDER*
: struct fi_ep_attr::msg_order field

*FI_TYPE_VERSION*
: Returns the library version of libfabric in string form.  The data
  parameter is ignored.

fi_tostr() will return a pointer to an internal libfabric buffer that
should not be modified, and will be overwritten the next time
fi_tostr() is invoked.  fi_tostr() is not thread safe.

# NOTES

The following resources are associated with fabric domains: access
domains, passive endpoints, and CM event queues.

# FABRIC ATTRIBUTES

The fi_fabric_attr structure defines the set of attributes associated
with a fabric and a fabric provider.

```c
struct fi_fabric_attr {
	struct fid_fabric *fabric;
	char              *name;
	char              *prov_name;
	uint32_t          prov_version;
	uint32_t          api_version;
};
```

## fabric

On input to fi_getinfo, a user may set this to an opened fabric
instance to restrict output to the given fabric.  On output from
fi_getinfo, if no fabric was specified, but the user has an opened
instance of the named fabric, this will reference the first opened
instance.  If no instance has been opened, this field will be NULL.

## name

A fabric identifier.

## prov_name - Provider Name

The name of the underlying fabric provider.

To request an utility provider layered over a specific core provider, both
the provider names have to be specified using ";" as delimiter.

e.g. "ofi_rxm;verbs" or "verbs;ofi_rxm"

For debugging and administrative purposes, environment variables can be used
to control which fabric providers will be registered with libfabric.
Specifying "FI_PROVIDER=foo,bar" will allow any providers with the names "foo"
or "bar" to be registered.  Similarly, specifying "FI_PROVIDER=^foo,bar" will
prevent any providers with the names "foo" or "bar" from being registered.
Providers which are not registered will not appear in fi_getinfo results.
Applications which need a specific set of providers should implement
their own filtering of fi_getinfo's results rather than relying on these
environment variables in a production setting.

## prov_version - Provider Version

Version information for the fabric provider, in a major.minor format.  The
use of the FI_MAJOR() and FI_MINOR() version macros may be used to extract
the major and minor version data.  See `fi_version(3)`.

In case of an utility provider layered over a core provider, the version
would always refer to that of the utility provider.

## api_version

The interface version requested by the application.  This value corresponds to
the version parameter passed into `fi_getinfo(3)`.

# RETURN VALUE

Returns FI_SUCCESS on success. On error, a negative value corresponding to
fabric errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# ERRORS


# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_domain`(3)](fi_domain.3.html),
[`fi_eq`(3)](fi_eq.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html)
