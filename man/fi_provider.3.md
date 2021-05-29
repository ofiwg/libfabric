---
layout: page
title: fi_provider(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_prov_ini \- External provider entry point

fi_param_define / fi_param_get
: Register and retrieve environment variables with the libfabric core

fi_log_enabled / fi_log
: Control and output debug logging information.

# SYNOPSIS

```c
#include <rdma/fabric.h>
#include <rdma/prov/fi_prov.h>
#include <rdma/prov/fi_log.h>

struct fi_provider* fi_prov_ini(void);

int fi_param_define(const struct fi_provider *provider, const char *param_name,
	enum fi_param_type type, const char *help_string_fmt, ...);

int fi_param_get_str(struct fi_provider *provider, const char *param_name,
    char **value);

int fi_param_get_int(struct fi_provider *provider, const char *param_name,
    int *value);

int fi_param_get_bool(struct fi_provider *provider, const char *param_name,
    int *value);

int fi_param_get_size_t(struct fi_provider *provider, const char *param_name,
    size_t *value);

int fi_log_enabled(const struct fi_provider *prov, enum fi_log_level level,
    enum fi_log_subsys subsys);

void fi_log(const struct fi_provider *prov, enum fi_log_level level,
    enum fi_log_subsys subsys, const char *func, int line,
    const char *fmt, ...);
```

# ARGUMENTS

*provider*
: Reference to the provider.

# DESCRIPTION

A fabric provider implements the application facing software
interfaces needed to access network specific protocols,
drivers, and hardware.  The interfaces and structures defined by
this man page are exported by the libfabric library, but are
targeted for provider implementations, rather than for direct
use by most applications.

Integrated providers are those built directly into the libfabric
library itself.  External providers are loaded dynamically by
libfabric at initialization time.  External providers must be in
a standard library path or in the libfabric library search path
as specified by environment variable.  Additionally, external
providers must be named with the suffix "-fi.so" at the end of
the name.

## fi_prov_ini

This entry point must be defined by external providers.  On loading,
libfabric will invoke fi_prov_ini() to retrieve the provider's
fi_provider structure.  Additional interactions between the libfabric
core and the provider will be through the interfaces defined by that
struct.

## fi_param_define / fi_param_get

TODO

## fi_log_enabled / fi_log

TODO

# NOTES

TODO

# PROVIDER INTERFACE

The fi_provider structure defines entry points for the libfabric core
to use to access the provider.  All other calls into a provider are
through function pointers associated with allocated objects.

```c
struct fi_provider {
	uint32_t version;
	uint32_t fi_version;
	struct fi_context context;
	const char *name;
	int	(*getinfo)(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **info);
	int	(*fabric)(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
			void *context);
	void	(*cleanup)(void);
};
```

## version

The provider version.  For providers integrated with the library, this is
often the same as the library version.

## fi_version

The library interface version that the provider was implemented against.
The provider's fi_version must be greater than or equal to an application's
requested api version for the application to use the provider.  It is a
provider's responsibility to support older versions of the api if it
wishes to supports legacy applications.  For integrated providers

## TODO

# RETURN VALUE

Returns FI_SUCCESS on success. On error, a negative value corresponding to
fabric errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# ERRORS


# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
