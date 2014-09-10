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

// TODO: Move all HAVE_VERBS functionality
#ifdef HAVE_VERBS
#include <rdma/rdma_cma.h>
#include <infiniband/ib.h>
#endif

struct __fid_fabric {
	struct fid_fabric	fabric_fid;
	uint64_t		flags;
	char			name[FI_NAME_MAX];
};

#ifdef HAVE_VERBS
struct __fid_eq_cm {
	struct fid_eq		eq_fid;
	struct __fid_fabric	*fab;
	struct rdma_event_channel *channel;
	uint64_t		flags;
	struct fi_eq_err_entry	err;
};

struct __fid_pep {
	struct fid_pep		pep_fid;
	struct __fid_eq_cm	*cm_eq;
	struct rdma_cm_id	*id;
};
#endif // HAVE_VERBS

static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
static int init;
static struct fi_prov *prov_head, *prov_tail;
struct fid_fabric *g_fabric;


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

int fi_init()
{
	int ret = 0;

	pthread_mutex_lock(&mut);
	if (init)
		goto out;

	init = 1;

#ifdef HAVE_VERBS
	ret = fi_fabric("RDMA", 0, &g_fabric, NULL);
	if (ret) {
		fprintf(stderr, "fi_init: fatal: unable to open fabric\n");
		return  -ENODEV;
	}
#endif // HAVE_VERBS
out:
	pthread_mutex_unlock(&mut);
	return ret;
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

	if (!init)
		fi_init();

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
	if (info->ep_attr)
		free(info->ep_attr);
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

static int
__fi_domain(struct fid_fabric *fabric, struct fi_info *info,
	    struct fid_domain **dom, void *context)
{
	struct __fid_fabric *fab;
	struct fi_prov *prov;
	int ret = -FI_ENOSYS;

	fab = container_of(fabric, struct __fid_fabric, fabric_fid);
	if (strcmp(fab->name, info->fabric_name))
		return -FI_EINVAL;

	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->ops->domain)
			continue;

		ret = prov->ops->domain(fabric, info, dom, context);
		if (!ret)
			break;
	}

	return ret;
}

int fi_sockaddr_len(struct sockaddr *addr)
{
	if (!addr)
		return 0;

	switch (addr->sa_family) {
	case AF_INET:
		return sizeof(struct sockaddr_in);
	case AF_INET6:
		return sizeof(struct sockaddr_in6);
#ifdef HAVE_VERBS
	case AF_IB:
		return sizeof(struct sockaddr_ib);
#endif // HAVE_VERBS
	default:
		return 0;
	}
}

#ifdef HAVE_VERBS
static ssize_t
__fi_eq_cm_readerr(struct fid_eq *eq, struct fi_eq_err_entry *entry,
		   size_t len, uint64_t flags)
{
	struct __fid_eq_cm *_eq;

	_eq = container_of(eq, struct __fid_eq_cm, eq_fid);
	if (!_eq->err.err)
		return 0;

	if (len < sizeof(*entry))
		return -EINVAL;

	*entry = _eq->err;
	_eq->err.err = 0;
	_eq->err.prov_errno = 0;
	return sizeof(*entry);
}

static struct fi_info *
__fi_eq_cm_getinfo(struct __fid_fabric *fab, struct rdma_cm_event *event)
{
	struct fi_info *fi;

	fi = calloc(1, sizeof *fi);
	if (!fi)
		return NULL;

	fi->type = FID_MSG;
	if (event->id->verbs->device->transport_type == IBV_TRANSPORT_IWARP) {
		fi->protocol = FI_PROTO_IWARP;
	} else {
		fi->protocol = FI_PROTO_IB_RC;
	}
	fi->ep_cap = FI_MSG | FI_RMA;

	fi->src_addrlen = fi_sockaddr_len(rdma_get_local_addr(event->id));
	if (!(fi->src_addr = malloc(fi->src_addrlen)))
		goto err;
	memcpy(fi->src_addr, rdma_get_local_addr(event->id), fi->src_addrlen);

	fi->dest_addrlen = fi_sockaddr_len(rdma_get_peer_addr(event->id));
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

static ssize_t __fi_eq_cm_process_event(struct __fid_eq_cm *eq,
	struct rdma_cm_event *event, struct fi_eq_cm_entry *entry, size_t len)
{
	fid_t fid;
	size_t datalen;

	fid = event->id->context;
	switch (event->event) {
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		rdma_migrate_id(event->id, NULL);
		entry->event = FI_CONNREQ;
		entry->info = __fi_eq_cm_getinfo(eq->fab, event);
		if (!entry->info) {
			rdma_destroy_id(event->id);
			return 0;
		}
		break;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		eq->err.fid_context = fid->context;
		eq->err.err = ENODEV;
		return -FI_EAVAIL;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		eq->err.fid_context = fid->context;
		eq->err.err = EADDRNOTAVAIL;
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

static ssize_t __fi_eq_cm_read_data(struct fid_eq *eq, void *buf, size_t len)
{
	struct __fid_eq_cm *_eq;
	struct fi_eq_cm_entry *entry;
	struct rdma_cm_event *event;
	size_t left;
	ssize_t ret = -FI_EINVAL;

	_eq = container_of(eq, struct __fid_eq_cm, eq_fid);
	entry = (struct fi_eq_cm_entry *) buf;
	if (_eq->err.err)
		return -FI_EAVAIL;

	for (left = len; left >= sizeof(*entry); ) {
		ret = rdma_get_cm_event(_eq->channel, &event);
		if (!ret) {
			ret = __fi_eq_cm_process_event(_eq, event, entry, left);
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

			if (!(_eq->flags & FI_BLOCK))
				return 0;

			fi_poll_fd(_eq->channel->fd);
		} else {
			ret = -errno;
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static ssize_t
__fi_eq_cm_condread_data(struct fid_eq *eq, void *buf, size_t len, const void *cond)
{
	ssize_t rc, cur, left;
	ssize_t  threshold;
	struct __fid_eq_cm *_eq;
	struct fi_eq_cm_entry *entry = (struct fi_eq_cm_entry *) buf;

	_eq = container_of(eq, struct __fid_eq_cm, eq_fid);
	threshold = cond ? (long) cond : sizeof(*entry);

	for(cur = 0, left = len; cur < threshold && left > 0; ) {
		rc = __fi_eq_cm_read_data(eq, (void*)entry, left);
		if (rc < 0)
			return rc;
		if (rc > 0) {
			left -= rc;
			entry += (rc / sizeof(*entry));
			cur += rc;
		}

		if (cur >= threshold || left <= 0)
			break;

		fi_poll_fd(_eq->channel->fd);
	}
	
	return cur;
}

static const char *
__fi_eq_cm_strerror(struct fid_eq *eq, int prov_errno, const void *prov_data,
		    void *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

struct fi_ops_eq __fi_eq_cm_data_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = __fi_eq_cm_read_data,
	.condread = __fi_eq_cm_condread_data,
	.readerr = __fi_eq_cm_readerr,
	.strerror = __fi_eq_cm_strerror
};

static int __fi_eq_cm_close(fid_t fid)
{
	struct __fid_eq_cm *eq;

	eq = container_of(fid, struct __fid_eq_cm, eq_fid.fid);
	if (eq->channel)
		rdma_destroy_event_channel(eq->channel);

	free(eq);
	return 0;
}

static int __fi_eq_cm_control(fid_t fid, int command, void *arg)
{
	struct __fid_eq_cm *eq;
	int ret = 0;

	eq = container_of(fid, struct __fid_eq_cm, eq_fid.fid);
	switch(command) {
	case FI_GETWAIT:
		if (!eq->channel) {
			ret = -FI_ENODATA;
			break;
		}
		*(void **) arg = &eq->channel->fd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

struct fi_ops __fi_eq_cm_ops = {
	.size = sizeof(struct fi_ops),
	.close = __fi_eq_cm_close,
	.control = __fi_eq_cm_control,
};

static int
__fi_eq_open(struct fid_fabric *fabric, const struct fi_eq_attr *attr,
	     struct fid_eq **eq, void *context)
{
	struct __fid_eq_cm *_eq;
	long flags = 0;
	int ret;

	if (attr->format != FI_EQ_FORMAT_CM)
		return -FI_ENOSYS;

	_eq = calloc(1, sizeof *_eq);
	if (!_eq)
		return -FI_ENOMEM;

	_eq->fab = container_of(fabric, struct __fid_fabric, fabric_fid);

	switch (attr->wait_obj) {
	case FI_WAIT_FD:
		_eq->channel = rdma_create_event_channel();
		if (!_eq->channel) {
			ret = -errno;
			goto err1;
		}
		fcntl(_eq->channel->fd, F_GETFL, &flags);
		ret = fcntl(_eq->channel->fd, F_SETFL, flags | O_NONBLOCK);
		if (ret) {
			ret = -errno;
			goto err2;
		}
		break;
	case FI_WAIT_NONE:
		break;
	default:
		return -FI_ENOSYS;
	}

	_eq->flags = attr->flags;
	_eq->eq_fid.fid.fclass = FID_CLASS_EQ;
	_eq->eq_fid.fid.context = context;
	_eq->eq_fid.fid.ops = &__fi_eq_cm_ops;
	_eq->eq_fid.ops = &__fi_eq_cm_data_ops;

	*eq = &_eq->eq_fid;
	return 0;
err2:
	if (_eq->channel)
		rdma_destroy_event_channel(_eq->channel);
err1:
	free(_eq);
	return ret;
}

static int __fi_pep_listen(struct fid_pep *pep)
{
	struct __fid_pep *_pep;

	_pep = container_of(pep, struct __fid_pep, pep_fid);
	return rdma_listen(_pep->id, 0) ? -errno : 0;
}

static struct fi_ops_cm __fi_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.listen = __fi_pep_listen,
};

static int __fi_pep_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
	struct __fid_pep *pep;
	struct __fid_eq_cm *eq;
	int ret;

	pep = container_of(fid, struct __fid_pep, pep_fid.fid);
	if (bfid->fclass != FID_CLASS_EQ)
		return -FI_EINVAL;

	eq = container_of(bfid, struct __fid_eq_cm, eq_fid.fid);
	pep->cm_eq = eq;
	ret = rdma_migrate_id(pep->id, pep->cm_eq->channel);
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

static int
__fi_endpoint(struct fid_fabric *fabric, struct fi_info *info,
	      struct fid_pep **pep, void *context)
{
	struct __fid_fabric *fab;
	struct __fid_pep *_pep;

	fab = container_of(fabric, struct __fid_fabric, fabric_fid);
	if (strcmp(fab->name, info->fabric_name))
		return -FI_EINVAL;

	if (!info->data || info->datalen != sizeof(_pep->id))
		return -FI_ENOSYS;

	_pep = calloc(1, sizeof *_pep);
	if (!_pep)
		return -FI_ENOMEM;

	_pep->id = info->data;
	_pep->id->context = &_pep->pep_fid.fid;
	info->data = NULL;
	info->datalen = 0;

	_pep->pep_fid.fid.fclass = FID_CLASS_PEP;
	_pep->pep_fid.fid.context = context;
	_pep->pep_fid.fid.ops = &__fi_pep_ops;
	_pep->pep_fid.cm = &__fi_pep_cm_ops;

	*pep = &_pep->pep_fid;
	return 0;
}
#else // HAVE_VERBS
static int
__fi_eq_open(struct fid_fabric *fabric, const struct fi_eq_attr *attr,
	     struct fid_eq **eq, void *context)
{
	return -FI_ENOSYS;
}

static int
__fi_endpoint(struct fid_fabric *fabric, struct fi_info *info,
	      struct fid_pep **pep, void *context)
{
	return -FI_ENOSYS;
}
#endif // HAVE_VERBS

static int __fi_fabric_close(fid_t fid)
{
	free(fid);
	return 0;
}

static struct fi_ops __fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = __fi_fabric_close,
};

static struct fi_ops_fabric __fi_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = __fi_domain,
	.endpoint = __fi_endpoint,
	.eq_open = __fi_eq_open,
};

int __fi_fabric(const char *name, uint64_t flags, struct fid_fabric **fabric,
	      void *context)
{
	struct __fid_fabric *fab;

	if (!init)
		fi_init();

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	fab->fabric_fid.fid.fclass = FID_CLASS_FABRIC;
	fab->fabric_fid.fid.context = context;
	fab->fabric_fid.fid.ops = &__fi_ops;
	fab->fabric_fid.ops = &__fi_ops_fabric;
	strncpy(fab->name, name, FI_NAME_MAX);
	fab->flags = flags;
	*fabric = &fab->fabric_fid;
	return 0;
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
