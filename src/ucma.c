/*
 * Copyright (c) 2005-2012 Intel Corporation.  All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <glob.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <poll.h>
#include <unistd.h>
#include <pthread.h>
#include <endian.h>
#include <byteswap.h>
#include <stddef.h>
#include <netdb.h>
#include <syslog.h>

#include <rdma/fabric.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_ucma.h>
#include "fi.h"


static int ucma_abi_ver = RDMA_USER_CM_MAX_ABI_VERSION;

#define UCMA_INIT_CMD(req, req_size, op)	\
do {						\
	(req)->cmd = UCMA_CMD_##op;		\
	(req)->in  = (req_size) - sizeof(struct ucma_abi_cmd_hdr); \
	(req)->out = 0;				\
} while (0)

#define UCMA_INIT_CMD_RESP(req, req_size, op, resp, resp_size) \
do {						\
	(req)->cmd = UCMA_CMD_##op;		\
	(req)->in  = (req_size) - sizeof(struct ucma_abi_cmd_hdr); \
	(req)->out = (resp_size);			\
	(req)->response = (uintptr_t) (resp);	\
} while (0)

static int ucma_open(const char *name, struct fi_info *info, fid_t *fid, void *context);

static struct fi_ops_prov ucma_prov_ops = {
	.size = sizeof(struct fi_ops_prov),
	.getinfo = NULL,
	.freeinfo = NULL,
	.socket = NULL,
	.open = ucma_open
};


static int ucma_abi_version(void)
{
	char value[8];

	if ((fi_read_file(fi_sysfs_path(), "class/misc/rdma_cm/abi_version",
			 value, sizeof value) < 0) &&
	    (fi_read_file(fi_sysfs_path(), "class/infiniband_ucma/abi_version",
			 value, sizeof value) < 0)) {
		return -ENOSYS;
	}

	ucma_abi_ver = strtol(value, NULL, 10);
	if (ucma_abi_ver < RDMA_USER_CM_MIN_ABI_VERSION ||
	    ucma_abi_ver > RDMA_USER_CM_MAX_ABI_VERSION) {
		fprintf(stderr, PFX "ucma kernel ABI version %d not supported (%d).\n",
			ucma_abi_ver, RDMA_USER_CM_MAX_ABI_VERSION);
		return -ENOSYS;
	}

	return 0;
}

int ucma_init(void)
{
	return ucma_abi_version();
}

void ucma_ini(void)
{
	fi_register(&ucma_prov_ops);
}

void ucma_fini(void)
{
}

static int __ucma_create_id(fid_t fid,
			struct ucma_abi_create_id *cmd, size_t cmd_size,
			struct ucma_abi_create_id_resp *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, CREATE_ID, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_destroy_id(fid_t fid,
			struct ucma_abi_destroy_id *cmd, size_t cmd_size,
			struct ucma_abi_destroy_id_resp *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, DESTROY_ID, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_bind_ip(fid_t fid,
			struct ucma_abi_bind_ip *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, BIND_IP);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_bind(fid_t fid,
		struct ucma_abi_bind *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, BIND);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}
static int __ucma_resolve_ip(fid_t fid,
		struct ucma_abi_resolve_ip *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, RESOLVE_IP);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_resolve_addr(fid_t fid,
		struct ucma_abi_resolve_addr *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, RESOLVE_ADDR);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_resolve_route(fid_t fid,
		struct ucma_abi_resolve_route *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, RESOLVE_ROUTE);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_query_route(fid_t fid,
			struct ucma_abi_query *cmd, size_t cmd_size,
			struct ucma_abi_query_route_resp *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, QUERY_ROUTE, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_query(fid_t fid,
			struct ucma_abi_query *cmd, size_t cmd_size,
			void *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, QUERY, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_connect(fid_t fid,
			struct ucma_abi_connect *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, CONNECT);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_listen(fid_t fid,
			struct ucma_abi_listen *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, LISTEN);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_accept(fid_t fid,
			struct ucma_abi_accept *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, ACCEPT);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_reject(fid_t fid,
			struct ucma_abi_reject *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, REJECT);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_disconnect(fid_t fid,
			struct ucma_abi_disconnect *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, DISCONNECT);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_init_qp_attr(fid_t fid,
			struct ucma_abi_init_qp_attr *cmd, size_t cmd_size,
			struct ibv_kern_qp_attr *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, INIT_QP_ATTR, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_get_event(fid_t fid,
			struct ucma_abi_get_event *cmd, size_t cmd_size,
			struct ucma_abi_event_resp *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, GET_EVENT, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_set_option(fid_t fid,
			struct ucma_abi_set_option *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, SET_OPTION);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_notify(fid_t fid,
			struct ucma_abi_notify *cmd, size_t cmd_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD(cmd, cmd_size, NOTIFY);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	return 0;
}

static int __ucma_join_ip_mcast(fid_t fid,
			struct ucma_abi_join_ip_mcast *cmd, size_t cmd_size,
			struct ucma_abi_create_id_resp *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, JOIN_IP_MCAST, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_join_mcast(fid_t fid,
			struct ucma_abi_join_mcast *cmd, size_t cmd_size,
			struct ucma_abi_create_id_resp *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, JOIN_MCAST, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_leave_mcast(fid_t fid,
			struct ucma_abi_destroy_id *cmd, size_t cmd_size,
			struct ucma_abi_destroy_id_resp *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, LEAVE_MCAST, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}

static int __ucma_migrate_id(fid_t fid,
			struct ucma_abi_migrate_id *cmd, size_t cmd_size,
			struct ucma_abi_migrate_resp *resp, size_t resp_size)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	UCMA_INIT_CMD_RESP(cmd, cmd_size, MIGRATE_ID, resp, resp_size);
	if (write(ucma->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof resp);
	return 0;
}


static struct fi_ops_ucma ops_ucma = {
	.size = sizeof(struct fi_ops_ucma),
	.create_id = __ucma_create_id,
	.destroy_id = __ucma_destroy_id,
	.bind_ip = __ucma_bind_ip,
	.bind = __ucma_bind,
	.resolve_ip = __ucma_resolve_ip,
	.resolve_addr = __ucma_resolve_addr,
	.resolve_route = __ucma_resolve_route,
	.query_route = __ucma_query_route,
	.query = __ucma_query,
	.connect = __ucma_connect,
	.listen = __ucma_listen,
	.accept = __ucma_accept,
	.reject = __ucma_reject,
	.disconnect = __ucma_disconnect,
	.init_qp_attr = __ucma_init_qp_attr,
	.get_event = __ucma_get_event,
	.set_option = __ucma_set_option,
	.notify = __ucma_notify,
	.join_ip_mcast = __ucma_join_ip_mcast,
	.join_mcast = __ucma_join_mcast,
	.leave_mcast = __ucma_leave_mcast,
	.migrate_id = __ucma_migrate_id
};

static int ucma_close(fid_t fid)
{
	struct fid_ucma *ucma;

	ucma = container_of(fid, struct fid_ucma, fid);
	close(ucma->fd);
	free(ucma);
	return 0;
}

static struct fi_ops ops_fi = {
	.size = sizeof(struct fi_ops),
	.close = ucma_close
};

static int ucma_open(const char *name, struct fi_info *info, fid_t *fid, void *context)
{
	struct fid_ucma *ucma;

	if (!name || strcmp(FI_UCMA_INTERFACE, name))
		return -ENOSYS;

 	ucma = calloc(1, sizeof(*ucma));
 	if (!ucma)
 		return -ENOMEM;

	ucma->fd = open("/dev/infiniband/rdma_cm", O_RDWR | O_CLOEXEC);
	if (ucma->fd < 0) {
		free(ucma);
		return -errno;
	}

	ucma->fid.fclass = FID_CLASS_INTERFACE;
	ucma->fid.size = sizeof(*ucma);
	ucma->fid.ops = &ops_fi;
	ucma->fid.context = context;
	ucma->ops = &ops_ucma;

	*fid = &ucma->fid;
	return 0;
}
