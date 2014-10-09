/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2006 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013 Intel Corp., Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include <rdma/fi_errno.h>
#include "fi.h"

#ifdef HAVE_LIBDL
#include <dlfcn.h>
#endif

#define FI_WARN(fmt, ...) do { fprintf(stderr, "%s: " fmt, PACKAGE, ##__VA_ARGS__); } while (0)

struct fi_prov {
	struct fi_prov		*next;
	struct fi_provider	*provider;
};

static struct fi_prov *prov_head, *prov_tail;

__attribute__((visibility ("default")))
int fi_register_provider_(uint32_t version, struct fi_provider *provider)
{
	struct fi_prov *prov;

	if (FI_MAJOR(version) != FI_MAJOR_VERSION ||
	    FI_MINOR(version) > FI_MINOR_VERSION)
		return -FI_ENOSYS;

	prov = calloc(sizeof *prov, 1);
	if (!prov)
		return -FI_ENOMEM;

	prov->provider = provider;
	if (prov_tail)
		prov_tail->next = prov;
	else
		prov_head = prov;
	prov_tail = prov;
	return 0;
}

#ifdef HAVE_LIBDL
static int lib_filter(const struct dirent *entry)
{
	size_t l = strlen(entry->d_name);
	size_t sfx = sizeof (FI_LIB_SUFFIX) - 1;

	if (l > sfx)
		return !strcmp(&(entry->d_name[l-sfx]), FI_LIB_SUFFIX);
	else
		return 0;
}

static void __attribute__((constructor)) fi_ini(void)
{
	struct dirent **liblist;
	int n;
	char *lib, *extdir = getenv("FI_EXTDIR");
	void *dlhandle;

	if (!extdir)
		extdir = EXTDIR;

	if (dlopen(NULL, RTLD_NOW) == NULL) {
		FI_WARN("dlopen(NULL) failed, assuming static linking\n");
		return;
	}

	n = scandir(extdir, &liblist, lib_filter, NULL);

	if (n < 0) {
		FI_WARN("scandir error reading %s: %s\n", extdir, strerror(n));
		return;
	}

	while (n--) {
		if (asprintf(&lib, "%s/%s", extdir, liblist[n]->d_name) < 0) {
			FI_WARN("asprintf failed to allocate memory\n");
			return;
		}

		dlhandle = dlopen(lib, RTLD_NOW);
		if (dlhandle == NULL)
			FI_WARN("dlopen(%s): %s\n", lib, dlerror());

		free(liblist[n]);
		free(lib);
	}

	free(liblist);
}
#endif

static void __attribute__((destructor)) fi_fini(void)
{
}

static struct fi_prov *fi_getprov(const char *prov_name)
{
	struct fi_prov *prov;

	for (prov = prov_head; prov; prov = prov->next) {
		if (!strcmp(prov_name, prov->provider->name))
			return prov;
	}

	return NULL;
}

__attribute__((visibility ("default")))
int fi_getinfo_(uint32_t version, const char *node, const char *service,
	       uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	struct fi_prov *prov;
	struct fi_info *tail, *cur;
	int ret = -ENOSYS;

	*info = tail = NULL;
	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->provider->getinfo)
			continue;

		ret = prov->provider->getinfo(version, node, service, flags,
					      hints, &cur);
		if (ret) {
			if (ret == -FI_ENODATA)
				continue;
			break;
		}

		if (!*info)
			*info = cur;
		else
			tail->next = cur;
		for (tail = cur; tail->next; tail = tail->next) {
			tail->fabric_attr->prov_name = strdup(prov->provider->name);
			tail->fabric_attr->prov_version = prov->provider->version;
		}
		tail->fabric_attr->prov_name = strdup(prov->provider->name);
		tail->fabric_attr->prov_version = prov->provider->version;
	}

	return *info ? 0 : ret;
}

__attribute__((visibility ("default")))
void fi_freeinfo_(struct fi_info *info)
{
	struct fi_prov *prov;
	struct fi_info *next;

	for (; info; info = next) {
		next = info->next;
		prov = info->fabric_attr ?
		       fi_getprov(info->fabric_attr->prov_name) : NULL;

		if (prov && prov->provider->freeinfo)
			prov->provider->freeinfo(info);
		else
			__fi_freeinfo(info);
	}
}

__attribute__((visibility ("default")))
int fi_fabric_(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context)
{
	struct fi_prov *prov;

	if (!attr || !attr->prov_name || !attr->name)
		return -FI_EINVAL;

	prov = fi_getprov(attr->prov_name);
	if (!prov || !prov->provider->fabric)
		return -FI_ENODEV;

	return prov->provider->fabric(attr, fabric, context);
}

__attribute__((visibility ("default")))
uint32_t if_version_(void)
{
	return FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION);
}

#define FI_ERRNO_OFFSET	256
#define FI_ERRNO_MAX	FI_EOPBADSTATE

static const char *const errstr[] = {
	[FI_EOTHER - FI_ERRNO_OFFSET] = "Unspecified error",
	[FI_ETOOSMALL - FI_ERRNO_OFFSET] = "Provided buffer is too small",
	[FI_EOPBADSTATE - FI_ERRNO_OFFSET] = "Operation not permitted in current state",
	[FI_EAVAIL - FI_ERRNO_OFFSET]  = "Error available",
	[FI_EBADFLAGS - FI_ERRNO_OFFSET] = "Flags not supported",
	[FI_ENOEQ - FI_ERRNO_OFFSET] = "Missing or unavailable event queue",
	[FI_EDOMAIN - FI_ERRNO_OFFSET] = "Invalid resource domain",
};

__attribute__((visibility ("default")))
const char *fi_strerror_(int errnum)
{
	if (errnum < FI_ERRNO_OFFSET)
		return strerror(errnum);
	else if (errno < FI_ERRNO_MAX)
		return errstr[errnum - FI_ERRNO_OFFSET];
	else
		return errstr[FI_EOTHER - FI_ERRNO_OFFSET];
}

static const size_t __fi_datatype_size[] = {
	[FI_INT8]   = sizeof(int8_t),
	[FI_UINT8]  = sizeof(uint8_t),
	[FI_INT16]  = sizeof(int16_t),
	[FI_UINT16] = sizeof(uint16_t),
	[FI_INT32]  = sizeof(int32_t),
	[FI_UINT32] = sizeof(uint32_t),
	[FI_INT64]  = sizeof(int64_t),
	[FI_UINT64] = sizeof(uint64_t),
	[FI_FLOAT]  = sizeof(float),
	[FI_DOUBLE] = sizeof(double),
	[FI_FLOAT_COMPLEX]  = sizeof(float complex),
	[FI_DOUBLE_COMPLEX] = sizeof(double complex),
	[FI_LONG_DOUBLE]    = sizeof(long double),
	[FI_LONG_DOUBLE_COMPLEX] = sizeof(long double complex),
};

size_t fi_datatype_size(enum fi_datatype datatype)
{
	if (datatype >= FI_DATATYPE_LAST) {
		errno = FI_EINVAL;
		return 0;
	}
	return __fi_datatype_size[datatype];
}
