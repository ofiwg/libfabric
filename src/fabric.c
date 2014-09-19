/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2006 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013 Intel Corp., Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_errno.h>
#include "fi.h"


struct fi_prov {
	struct fi_prov		*next;
	struct fi_provider	*provider;
};

static struct fi_prov *prov_head, *prov_tail;


int fi_read_file(const char *dir, const char *file, char *buf, size_t size)
{
	char *path;
	int fd, len;

	if (asprintf(&path, "%s/%s", dir, file) < 0)
		return -1;

	fd = open(path, O_RDONLY);
	if (fd < 0) {
		free(path);
		return -1;
	}

	len = read(fd, buf, size);
	close(fd);
	free(path);

	if (len > 0 && buf[len - 1] == '\n')
		buf[--len] = '\0';

	return len;
}

int fi_version_register(int version, struct fi_provider *provider)
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

int fi_poll_fd(int fd, int timeout)
{
	struct pollfd fds;

	fds.fd = fd;
	fds.events = POLLIN;
	return poll(&fds, 1, timeout) < 0 ? -errno : 0;
}

int fi_wait_cond(pthread_cond_t *cond, pthread_mutex_t *mut, int timeout)
{
	struct timespec ts;

	if (timeout < 0)
		return pthread_cond_wait(cond, mut);

	clock_gettime(CLOCK_REALTIME, &ts);
	ts.tv_sec += timeout / 1000;
	ts.tv_nsec += (timeout % 1000) * 1000000;
	return pthread_cond_timedwait(cond, mut, &ts);
}

static void __attribute__((constructor)) fi_ini(void)
{
	sock_ini();
	ibv_ini();
	psmx_ini();
}

static void __attribute__((destructor)) fi_fini(void)
{
	psmx_fini();
	ibv_fini();
	sock_fini();
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

int fi_getinfo(int version, const char *node, const char *service,
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

struct fi_info *__fi_allocinfo(void)
{
	struct fi_info *info;

	info = calloc(1, sizeof(*info));
	if (!info)
		return NULL;

	info->ep_attr = calloc(1, sizeof(*info->ep_attr));
	info->domain_attr = calloc(1, sizeof(*info->domain_attr));
	info->fabric_attr = calloc(1, sizeof(*info->fabric_attr));
	if (!info->ep_attr || !info->domain_attr || !info->fabric_attr)
		goto err;

	return info;
err:
	__fi_freeinfo(info);
	return NULL;
}

void __fi_freeinfo(struct fi_info *info)
{
	free(info->src_addr);
	free(info->dest_addr);
	free(info->ep_attr);
	if (info->domain_attr) {
		free(info->domain_attr->name);
		free(info->domain_attr);
	}
	if (info->fabric_attr) {
		free(info->fabric_attr->name);
		free(info->fabric_attr->prov_name);
		free(info->fabric_attr);
	}
	free(info->data);
	free(info);
}

void fi_freeinfo(struct fi_info *info)
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

int fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context)
{
	struct fi_prov *prov;

	if (!attr || !attr->prov_name || !attr->name)
		return -FI_EINVAL;

	prov = fi_getprov(attr->prov_name);
	if (!prov || !prov->provider->fabric)
		return -FI_ENODEV;

	return prov->provider->fabric(attr, fabric, context);
}

uint32_t fi_version(void)
{
	return FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION);
}

uint64_t fi_tag_bits(uint64_t mem_tag_format)
{
	return UINT64_MAX >> (ffsll(htonll(mem_tag_format)) -1);
}

uint64_t fi_tag_format(uint64_t tag_bits)
{
	return FI_TAG_GENERIC >> (ffsll(htonll(tag_bits)) - 1);
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

const char *fi_strerror(int errnum)
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
