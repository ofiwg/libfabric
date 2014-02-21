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
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#include <rdma/fi_ucma.h>
#include <rdma/fi_uverbs.h>
#include <rdma/fi_errno.h>
#include "fi.h"
#include <rdma/rdma_cma.h>

struct __fid_fabric {
	struct fid_fabric	fabric_fid;
	uint64_t		flags;
	char			name[FI_NAME_MAX];
};

struct __fid_ec_cm {
	struct fid_ec		ec_fid;
	struct __fid_fabric	*fab;
	struct rdma_event_channel *channel;
	uint64_t		flags;
	struct fi_ec_err_entry	err;
};

struct __fid_pep {
	struct fid_pep		pep_fid;
	struct __fid_ec_cm	*cm_ec;
	struct rdma_cm_id	*id;
};


static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
static int init;
static struct fi_prov *prov_head, *prov_tail;
fid_t fabric;


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

int fi_poll_fd(int fd)
{
	struct pollfd fds;

	fds.fd = fd;
	fds.events = POLLIN;
	return poll(&fds, 1, -1) < 0 ? -errno : 0;
}

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

	ret = fi_fabric("RDMA", 0, &fabric, NULL);
	if (ret) {
		fprintf(stderr, "fi_init: fatal: unable to open fabric\n");
		return  -ENODEV;
	}
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

int fi_getinfo(const char *node, const char *service, struct fi_info *hints,
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

void __fi_freeinfo(struct fi_info *info)
{
	if (info->src_addr)
		free(info->src_addr);
	if (info->dest_addr)
		free(info->dest_addr);
	if (info->fabric_name)
		free(info->fabric_name);
	if (info->domain_name)
		free(info->domain_name);
	if (info->auth_key)
		free(info->auth_key);
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

static int __fi_domain(fid_t fid, struct fi_info *info, fid_t *dom, void *context)
{
	struct __fid_fabric *fab;
	struct fi_prov *prov;
	int ret = -FI_ENOSYS;

	fab = container_of(fid, struct __fid_fabric, fabric_fid.fid);
	if (strcmp(fab->name, info->fabric_name))
		return -FI_EINVAL;

	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->ops->domain)
			continue;

		ret = prov->ops->domain(fid, info, dom, context);
		if (!ret)
			break;
	}

	return ret;
}

static ssize_t __fi_ec_cm_readerr(fid_t fid, void *buf, size_t len, uint64_t flags)
{
	struct __fid_ec_cm *ec;
	struct fi_ec_err_entry *entry;

	ec = container_of(fid, struct __fid_ec_cm, ec_fid.fid);
	if (!ec->err.err)
		return 0;

	if (len < sizeof(*entry))
		return -EINVAL;

	entry = (struct fi_ec_err_entry *) buf;
	*entry = ec->err;
	ec->err.err = 0;
	ec->err.prov_errno = 0;
	return sizeof(*entry);
}

static struct fi_info *
__fi_ec_cm_getinfo(struct __fid_fabric *fab, struct rdma_cm_event *event)
{
	struct fi_info *fi;

	fi = calloc(1, sizeof *fi);
	if (!fi)
		return NULL;

	fi->size = sizeof *fi;
	fi->type = FID_MSG;
	if (event->id->verbs->device->transport_type == IBV_TRANSPORT_IWARP) {
		fi->protocol = FI_PROTO_IWARP;
	} else {
		fi->protocol = FI_PROTO_IB_RC;
	}
	fi->protocol_cap = FI_PROTO_CAP_MSG | FI_PROTO_CAP_RMA;

	fi->src_addrlen = rdma_addrlen(rdma_get_local_addr(event->id));
	if (!(fi->src_addr = malloc(fi->src_addrlen)))
		goto err;
	memcpy(fi->src_addr, rdma_get_local_addr(event->id), fi->src_addrlen);

	fi->dest_addrlen = rdma_addrlen(rdma_get_peer_addr(event->id));
	if (!(fi->dest_addr = malloc(fi->dest_addrlen)))
		goto err;
	memcpy(fi->dest_addr, rdma_get_peer_addr(event->id), fi->dest_addrlen);

	if (!(fi->fabric_name = strdup(fab->name)))
		goto err;

	if (!(fi->domain_name = strdup(event->id->verbs->device->name)))
		goto err;

	fi->datalen = sizeof event->id;
	fi->data = event->id;
	return fi;
err:
	fi_freeinfo(fi);
	return NULL;
}

static ssize_t __fi_ec_cm_process_event(struct __fid_ec_cm *ec,
	struct rdma_cm_event *event, struct fi_ec_cm_entry *entry, size_t len)
{
	fid_t fid;
	size_t datalen;

	fid = event->id->context;
	switch (event->event) {
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		rdma_migrate_id(event->id, NULL);
		entry->event = FI_CONNREQ;
		entry->info = __fi_ec_cm_getinfo(ec->fab, event);
		if (!entry->info) {
			rdma_destroy_id(event->id);
			return 0;
		}
		break;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		ec->err.fid_context = fid->context;
		ec->err.err = ENODEV;
		return -FI_EAVAIL;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		ec->err.fid_context = fid->context;
		ec->err.err = EADDRNOTAVAIL;
		return -FI_EAVAIL;
	default:
		return 0;
	}

	entry->fid_context = fid->context;
	entry->flags = 0;
	datalen = min(len - sizeof(*entry), event->param.conn.private_data_len);
	if (datalen)
		memcpy(entry->data, event->param.conn.private_data, datalen);
	return sizeof(*entry) + datalen;
}

static ssize_t __fi_ec_cm_read_data(fid_t fid, void *buf, size_t len)
{
	struct __fid_ec_cm *ec;
	struct fi_ec_cm_entry *entry;
	struct rdma_cm_event *event;
	size_t left;
	ssize_t ret = -FI_EINVAL;

	ec = container_of(fid, struct __fid_ec_cm, ec_fid.fid);
	entry = (struct fi_ec_cm_entry *) buf;
	if (ec->err.err)
		return -FI_EAVAIL;

	for (left = len; left >= sizeof(*entry); ) {
		ret = rdma_get_cm_event(ec->channel, &event);
		if (!ret) {
			ret = __fi_ec_cm_process_event(ec, event, entry, left);
			rdma_ack_cm_event(event);
			if (ret < 0)
				break;
			else if (!ret)
				continue;

			left -= ret;
			entry = ((void *) entry) + ret;
		} else if (errno == EAGAIN) {
			if (left < len)
				return len - left;

			if (!(ec->flags & FI_BLOCK))
				return 0;

			fi_poll_fd(ec->channel->fd);
		} else {
			ret = -errno;
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static const char * __fi_ec_cm_strerror(fid_t fid, int prov_errno, const void *prov_data,
				       void *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

struct fi_ops_ec __fi_ec_cm_data_ops = {
	.size = sizeof(struct fi_ops_ec),
	.read = __fi_ec_cm_read_data,
	.readerr = __fi_ec_cm_readerr,
	.strerror = __fi_ec_cm_strerror
};

static int __fi_ec_cm_close(fid_t fid)
{
	struct __fid_ec_cm *ec;

	ec = container_of(fid, struct __fid_ec_cm, ec_fid.fid);
	if (ec->channel)
		rdma_destroy_event_channel(ec->channel);

	free(ec);
	return 0;
}

static int __fi_ec_cm_control(fid_t fid, int command, void *arg)
{
	struct __fid_ec_cm *ec;
	int ret = 0;

	ec = container_of(fid, struct __fid_ec_cm, ec_fid.fid);
	switch(command) {
	case FI_GETECWAIT:
		if (!ec->channel) {
			ret = -FI_ENODATA;
			break;
		}
		*(void **) arg = &ec->channel->fd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

struct fi_ops __fi_ec_cm_ops = {
	.size = sizeof(struct fi_ops),
	.close = __fi_ec_cm_close,
	.control = __fi_ec_cm_control,
};

static int
__fi_ec_open(fid_t fid, const struct fi_ec_attr *attr, fid_t *ec, void *context)
{
	struct __fid_ec_cm *xec;
	long flags = 0;
	int ret;

	if (attr->type != FI_EC_QUEUE || attr->format != FI_EC_FORMAT_CM)
		return -FI_ENOSYS;

	xec = calloc(1, sizeof *xec);
	if (!xec)
		return -FI_ENOMEM;

	xec->fab = container_of(fid, struct __fid_fabric, fabric_fid.fid);

	switch (attr->wait_obj) {
	case FI_EC_WAIT_FD:
		xec->channel = rdma_create_event_channel();
		if (!xec->channel) {
			ret = -errno;
			goto err1;
		}
		fcntl(xec->channel->fd, F_GETFL, &flags);
		ret = fcntl(xec->channel->fd, F_SETFL, flags | O_NONBLOCK);
		if (ret) {
			ret = -errno;
			goto err2;
		}
		break;
	case FI_EC_WAIT_NONE:
		break;
	default:
		return -FI_ENOSYS;
	}

	xec->flags = attr->flags;
	xec->ec_fid.fid.fclass = FID_CLASS_EC;
	xec->ec_fid.fid.size = sizeof(struct fid_ec);
	xec->ec_fid.fid.context = context;
	xec->ec_fid.fid.ops = &__fi_ec_cm_ops;
	xec->ec_fid.ops = &__fi_ec_cm_data_ops;

	*ec = &xec->ec_fid.fid;
	return 0;
err2:
	if (xec->channel)
		rdma_destroy_event_channel(xec->channel);
err1:
	free(xec);
	return ret;
}

static int __fi_pep_listen(fid_t fid)
{
	struct __fid_pep *pep;

	pep = container_of(fid, struct __fid_pep, pep_fid.fid);
	return rdma_listen(pep->id, 0) ? -errno : 0;
}

static struct fi_ops_cm __fi_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.listen = __fi_pep_listen,
};

static int __fi_pep_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	struct __fid_pep *pep;
	struct __fid_ec_cm *ec;
	int ret;

	pep = container_of(fid, struct __fid_pep, pep_fid.fid);
	if ((nfids != 1) || (fids[0].fid->fclass != FID_CLASS_EC))
		return -FI_EINVAL;

	ec = container_of(fids[0].fid, struct __fid_ec_cm, ec_fid.fid);
	pep->cm_ec = ec;
	ret = rdma_migrate_id(pep->id, pep->cm_ec->channel);
	if (ret)
		return -errno;

	return 0;
}

static int __fi_pep_close(fid_t fid)
{
	struct __fid_pep *pep;

	pep = container_of(fid, struct __fid_pep, pep_fid.fid);
	if (pep->id)
		rdma_destroy_ep(pep->id);

	free(pep);
	return 0;
}

static struct fi_ops __fi_pep_ops = {
	.size = sizeof(struct fi_ops),
	.close = __fi_pep_close,
	.bind = __fi_pep_bind
};

static int __fi_endpoint(fid_t fid, struct fi_info *info, fid_t *pep, void *context)
{
	struct __fid_fabric *fab;
	struct __fid_pep *xpep;

	fab = container_of(fid, struct __fid_fabric, fabric_fid.fid);
	if (strcmp(fab->name, info->fabric_name))
		return -FI_EINVAL;

	if (!info->data || info->datalen != sizeof(xpep->id))
		return -FI_ENOSYS;

	xpep = calloc(1, sizeof *xpep);
	if (!xpep)
		return -FI_ENOMEM;

	xpep->id = info->data;
	xpep->id->context = &xpep->pep_fid.fid;
	info->data = NULL;
	info->datalen = 0;

	xpep->pep_fid.fid.fclass = FID_CLASS_PEP;
	xpep->pep_fid.fid.size = sizeof(struct fid_pep);
	xpep->pep_fid.fid.context = context;
	xpep->pep_fid.fid.ops = &__fi_pep_ops;
	xpep->pep_fid.cm = &__fi_pep_cm_ops;

	*pep = &xpep->pep_fid.fid;
	return 0;
}

static int __fi_fabric_close(fid_t fid)
{
	free(fid);
	return 0;
}

static struct fi_ops __fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = __fi_fabric_close,
};

static int __fi_open(fid_t fid, const char *name, uint64_t flags,
	fid_t *fif, void *context)
{
	struct __fid_fabric *fab;
	struct fi_prov *prov;
	int ret = -FI_ENOSYS;

	fab = container_of(fid, struct __fid_fabric, fabric_fid.fid);
	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->ops->if_open)
			continue;

		ret = prov->ops->if_open(fab->name, name, flags, fif, context);
		if (!ret)
			break;
	}

	return ret;
}

static struct fi_ops_fabric __fi_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = __fi_domain,
	.endpoint = __fi_endpoint,
	.ec_open = __fi_ec_open,
	.if_open = __fi_open
};

int fi_fabric(const char *name, uint64_t flags, fid_t *fid, void *context)
{
	struct __fid_fabric *fab;

	if (!init)
		fi_init();

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	fab->fabric_fid.fid.fclass = FID_CLASS_FABRIC;
	fab->fabric_fid.fid.size = sizeof(struct fid_fabric);
	fab->fabric_fid.fid.context = context;
	fab->fabric_fid.fid.ops = &__fi_ops;
	fab->fabric_fid.ops = &__fi_ops_fabric;
	strncpy(fab->name, name, FI_NAME_MAX);
	fab->flags = flags;
	*fid = &fab->fabric_fid.fid;
	return 0;
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
