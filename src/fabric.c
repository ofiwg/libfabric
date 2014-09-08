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


static struct fi_prov *prov_head, *prov_tail;


const char *fi_sysfs_path(void)
{
	static char *sysfs_path;
	char *env = NULL;

	if (sysfs_path)
		return sysfs_path;

	/*
	 * Only follow path passed in through the calling user's
	 * environment if we're not running SUID.
	 */
	if (getuid() == geteuid())
		env = getenv("SYSFS_PATH");

	if (env) {
		int len;

		sysfs_path = strndup(env, FI_PATH_MAX);
		len = strlen(sysfs_path);
		while (len > 0 && sysfs_path[len - 1] == '/') {
			--len;
			sysfs_path[len] = '\0';
		}
	} else {
		sysfs_path = "/sys";
	}

	return sysfs_path;
}

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

int fi_version_register(int version, struct fi_ops_prov *ops)
{
	struct fi_prov *prov;

	if (FI_MAJOR(version) != FI_MAJOR_VERSION ||
	    FI_MINOR(version) > FI_MINOR_VERSION)
		return -FI_ENOSYS;

	prov = calloc(sizeof *prov, 1);
	if (!prov)
		return -FI_ENOMEM;

	prov->ops = ops;
	if (prov_tail)
		prov_tail->next = prov;
	else
		prov_head = prov;
	prov_tail = prov;
	return 0;
}

int fi_poll_fd(int fd)
{
	struct pollfd fds;

	fds.fd = fd;
	fds.events = POLLIN;
	return poll(&fds, 1, -1) < 0 ? -errno : 0;
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

int fi_getinfo(int version, const char *node, const char *service,
	       uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	struct fi_prov *prov;
	struct fi_info *tail, *cur;
	int ret = -ENOSYS;

	*info = tail = NULL;
	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->ops->getinfo)
			continue;

		ret = prov->ops->getinfo(version, node, service, flags,
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
		for (tail = cur; tail->next; tail = tail->next)
			;
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
	if (!info->ep_attr)
		goto err;

	info->domain_attr = calloc(1, sizeof(*info->domain_attr));
	if (!info->domain_attr)
		goto err;

	return info;
err:
	__fi_freeinfo(info);
	return NULL;
}

void __fi_freeinfo(struct fi_info *info)
{
	if (info->src_addr)
		free(info->src_addr);
	if (info->dest_addr)
		free(info->dest_addr);
	if (info->fabric_name)
		free(info->fabric_name);
	if (info->auth_key)
		free(info->auth_key);
	if (info->ep_attr)
		free(info->ep_attr);
	if (info->domain_attr) {
		if (info->domain_attr->name)
			free(info->domain_attr->name);
		free(info->domain_attr);
	}
	if (info->data)
		free(info->data);

	free(info);
}

void fi_freeinfo(struct fi_info *info)
{
	struct fi_prov *prov;
	struct fi_info *next;
	int ret;

	while (info) {
		next = info->next;
		for (prov = prov_head; prov && info; prov = prov->next) {
			if (!prov->ops->freeinfo)
				continue;

			ret = prov->ops->freeinfo(info);
			if (!ret)
				goto cont;
		}
		__fi_freeinfo(info);
cont:
		info = next;
	}
}

int fi_fabric(const char *name, uint64_t flags, struct fid_fabric **fabric,
	      void *context)
{
	struct fi_prov *prov;
	int ret = -FI_ENOSYS;

	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->ops->fabric)
			continue;

		ret = prov->ops->fabric(name, flags, fabric, context);
		if (!ret)
			break;
	}

	return ret;
}

#define FI_ERRNO_OFFSET	256
#define FI_ERRNO_MAX	FI_EOPBADSTATE

static const char *const errstr[] = {
	[FI_EOTHER - FI_ERRNO_OFFSET] = "Unspecified error",
	[FI_ETOOSMALL - FI_ERRNO_OFFSET] = "Provided buffer is too small",
	[FI_EOPBADSTATE - FI_ERRNO_OFFSET] = "Operation not permitted in current state",
	[FI_EAVAIL - FI_ERRNO_OFFSET]  = "Error available",
	[FI_EBADFLAGS - FI_ERRNO_OFFSET] = "Flags not supported",
	[FI_ENOEC - FI_ERRNO_OFFSET] = "Missing or unavailable EC",
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
