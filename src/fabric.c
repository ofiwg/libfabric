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

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_arch.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_rdma.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_ucma.h>
#include <rdma/fi_umad.h>
#include <rdma/fi_uverbs.h>
#include <rdma/fi_errno.h>
#include "fi.h"

static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
static int init;
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

void fi_register(struct fi_ops_prov *ops)
{
	struct fi_prov *prov;

	prov = calloc(sizeof *prov, 1);
	if (!prov)
		return;

	prov->ops = ops;
	if (prov_tail)
		prov_tail->next = prov;
	else
		prov_head = prov;
	prov_tail = prov;
}

int  ucma_init(void);
int fi_init()
{
	int ret = 0;

	pthread_mutex_lock(&mut);
	if (init)
		goto out;

	ret = uv_init();
	if (ret)
		goto out;

	ret = ucma_init();
	if (ret)
		goto out;

	init = 1;
out:
	pthread_mutex_unlock(&mut);
	return ret;
}

static void __attribute__((constructor)) fi_ini(void)
{
	uv_ini();
	ibv_ini();
	ucma_ini();
	rdma_cm_ini();
	psmx_ini();
	mlx4_ini();
}

static void __attribute__((destructor)) fi_fini(void)
{
	mlx4_fini();
	psmx_fini();
	rdma_cm_fini();
	ucma_fini();
	ibv_fini();
	uv_fini();
}

int fi_getinfo(char *node, char *service, struct fi_info *hints,
	       struct fi_info **info)
{
	struct fi_prov *prov;
	struct fi_info *tail, *cur;
	int ret = -ENOSYS;

	if (!init)
		fi_init();

	*info = tail = NULL;
	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->ops->getinfo)
			continue;

		ret = prov->ops->getinfo(node, service, hints, &cur);
		if (ret)
			continue;

		if (!*info)
			*info = cur;
		else
			tail->next = cur;
		for (tail = cur; tail->next; tail = tail->next)
			;
	}

	return *info ? 0 : ret;
}

void __fi_freeinfo(struct fi_info *info)
{
	if (info->src_addr)
		free(info->src_addr);
	if (info->dst_addr)
		free(info->dst_addr);
	if (info->domain_name)
		free(info->domain_name);
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

int fi_open(char *name, struct fi_info *info, fid_t *fid, void *context)
{
	struct fi_prov *prov;
	int ret = -ENOSYS;

	if (!init)
		fi_init();

	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->ops->open)
			continue;

		ret = prov->ops->open(name, info, fid, context);
		if (!ret)
			break;
	}

	return ret;
}

int fi_endpoint(struct fi_info *info, fid_t *fid, void *context)
{
	struct fi_prov *prov;
	int ret = -ENOSYS;

	if (!init)
		fi_init();

	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->ops->endpoint)
			continue;

		ret = prov->ops->endpoint(info, fid, context);
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
	[FI_EOPBADSTATE - FI_ERRNO_OFFSET] = "Operation not permitted in current state"
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
