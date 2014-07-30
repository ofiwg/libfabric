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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <rdma/fabric.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_uverbs.h>
#include "fi.h"


int uv_abi_ver;
struct uv_dev *udev_head, *udev_tail;

#define UV_INIT_CMD(cmd, size, opcode)				\
	do {							\
		(cmd)->command = UVERBS_CMD_##opcode;		\
		(cmd)->in_words  = (size) / 4;			\
		(cmd)->out_words = 0;				\
	} while (0)

#define UV_INIT_CMD_RESP(cmd, size, opcode, out, outsize)	\
	do {							\
		(cmd)->command = UVERBS_CMD_##opcode;		\
		(cmd)->in_words  = (size) / 4;			\
		(cmd)->out_words = (outsize) / 4;		\
		(cmd)->response  = (uintptr_t) (out);		\
	} while (0)

static int uv_open(const char *res_name, const char *if_name,
	uint64_t flags, fid_t *fif, void *context);

static struct fi_ops_prov uv_prov_ops = {
	.if_open = uv_open
};

static int uv_abi_version(void)
{
	char value[8];

	if (fi_read_file(fi_sysfs_path(), "class/infiniband_verbs/abi_version",
			 value, sizeof value) < 0) {
		return -ENOSYS;
	}

	uv_abi_ver = strtol(value, NULL, 10);
	if (uv_abi_ver < UVERBS_MIN_ABI_VERSION ||
	    uv_abi_ver > UVERBS_MAX_ABI_VERSION) {
		fprintf(stderr, PFX "uverbs kernel ABI version %d not supported (%d).\n",
			uv_abi_ver, UVERBS_MAX_ABI_VERSION);
		return -ENOSYS;
	}

	return 0;
}

int uv_init(void)
{
	char class_path[FI_PATH_MAX];
	DIR *class_dir;
	struct dirent *dent;
	struct uv_dev *udev = NULL;
	struct stat buf;
	int ret;

	ret = uv_abi_version();
	if (ret)
		return ret;

	snprintf(class_path, sizeof class_path, "%s/class/infiniband_verbs",
		 fi_sysfs_path());

	class_dir = opendir(class_path);
	if (!class_dir)
		return -ENOSYS;

	while ((dent = readdir(class_dir))) {
		if (dent->d_name[0] == '.')
			continue;

		if (!udev)
			udev = calloc(sizeof *udev, 1);
		if (!udev) {
			ret = -ENOMEM;
			break;
		}

		snprintf(udev->sysfs_path, sizeof udev->sysfs_path,
			 "%s/%s", class_path, dent->d_name);

		if (stat(udev->sysfs_path, &buf)) {
			fprintf(stderr, PFX "Warning: couldn't stat '%s'.\n",
				udev->sysfs_path);
			continue;
		}

		if (!S_ISDIR(buf.st_mode))
			continue;

		snprintf(udev->sysfs_name, sizeof udev->sysfs_name, "%s", dent->d_name);

		if (fi_read_file(udev->sysfs_path, "ibdev", udev->dev_name,
				 sizeof udev->dev_name) < 0) {
			fprintf(stderr, PFX "Warning: no dev class attr for '%s'.\n",
				dent->d_name);
			continue;
		}

		snprintf(udev->dev_path, sizeof udev->dev_path,
			 "%s/class/infiniband/%s", fi_sysfs_path(), udev->dev_name);

		if (udev_tail)
			udev_tail->next = udev;
		else
			udev_head = udev;
		udev_tail = udev;
		udev = NULL;
	}

	if (udev)
		free(udev);

	closedir(class_dir);
	return ret;
}

void uv_ini(void)
{
	fi_register(&uv_prov_ops);
}

void uv_fini(void)
{
}

static int __uv_get_context(fid_t fid,
			struct ibv_get_context *cmd, size_t cmd_size,
			struct ibv_get_context_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, GET_CONTEXT, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_query_device(fid_t fid,
			struct ibv_query_device *cmd, size_t cmd_size,
			struct ibv_query_device_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, QUERY_DEVICE, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_query_port(fid_t fid,
			struct ibv_query_port *cmd, size_t cmd_size,
			struct ibv_query_port_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, QUERY_PORT, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_alloc_pd(fid_t fid,
			struct ibv_alloc_pd *cmd, size_t cmd_size,
			struct ibv_alloc_pd_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, ALLOC_PD, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_dealloc_pd(fid_t fid,
			struct ibv_dealloc_pd *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, DEALLOC_PD);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static int __uv_open_xrcd(fid_t fid,
			struct ibv_open_xrcd *cmd, size_t cmd_size,
			struct ibv_open_xrcd_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, OPEN_XRCD, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_close_xrcd(fid_t fid,
			struct ibv_close_xrcd *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, CLOSE_XRCD);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static int __uv_reg_mr(fid_t fid,
			struct ibv_reg_mr *cmd, size_t cmd_size,
			struct ibv_reg_mr_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, REG_MR, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_dereg_mr(fid_t fid,
			struct ibv_dereg_mr *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, DEREG_MR);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static int __uv_create_comp_channel(fid_t fid,
			struct ibv_create_comp_channel *cmd, size_t cmd_size,
			struct ibv_create_comp_channel_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, CREATE_COMP_CHANNEL, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_create_cq(fid_t fid,
			struct ibv_create_cq *cmd, size_t cmd_size,
			struct ibv_create_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, CREATE_CQ, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_poll_cq(fid_t fid,
			struct ibv_poll_cq *cmd, size_t cmd_size,
			struct ibv_poll_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, POLL_CQ, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_req_notify_cq(fid_t fid,
			struct ibv_req_notify_cq *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, REQ_NOTIFY_CQ);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static int __uv_resize_cq(fid_t fid,
			struct ibv_resize_cq *cmd, size_t cmd_size,
			struct ibv_resize_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, RESIZE_CQ, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_destroy_cq(fid_t fid,
			struct ibv_destroy_cq *cmd, size_t cmd_size,
			struct ibv_destroy_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, DESTROY_CQ, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_create_srq(fid_t fid,
			struct ibv_create_srq *cmd, size_t cmd_size,
			struct ibv_create_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, CREATE_SRQ, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_modify_srq(fid_t fid,
			struct ibv_modify_srq *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, MODIFY_SRQ);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static int __uv_query_srq(fid_t fid,
			struct ibv_query_srq *cmd, size_t cmd_size,
			struct ibv_query_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, QUERY_SRQ, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_destroy_srq(fid_t fid,
			struct ibv_destroy_srq *cmd, size_t cmd_size,
			struct ibv_destroy_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, DESTROY_SRQ, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_create_qp(fid_t fid,
			struct ibv_create_qp *cmd, size_t cmd_size,
			struct ibv_create_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, CREATE_QP, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_open_qp(fid_t fid,
			struct ibv_open_qp *cmd, size_t cmd_size,
			struct ibv_create_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, OPEN_QP, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_query_qp(fid_t fid,
			struct ibv_query_qp *cmd, size_t cmd_size,
			struct ibv_query_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, QUERY_QP, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_modify_qp(fid_t fid,
			struct ibv_modify_qp *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, MODIFY_QP);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static int __uv_destroy_qp(fid_t fid,
			struct ibv_destroy_qp *cmd, size_t cmd_size,
			struct ibv_destroy_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, DESTROY_QP, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_post_send(fid_t fid,
			struct ibv_post_send *cmd, size_t cmd_size,
			struct ibv_post_send_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, POST_SEND, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_post_recv(fid_t fid,
			struct ibv_post_recv *cmd, size_t cmd_size,
			struct ibv_post_recv_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, POST_RECV, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_post_srq_recv(fid_t fid,
			struct ibv_post_srq_recv *cmd, size_t cmd_size,
			struct ibv_post_srq_recv_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, POST_SRQ_RECV, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_create_ah(fid_t fid,
			struct ibv_create_ah *cmd, size_t cmd_size,
			struct ibv_create_ah_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD_RESP(cmd, cmd_size, CREATE_AH, resp, resp_size);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);
	return 0;
}

static int __uv_destroy_ah(fid_t fid,
			struct ibv_destroy_ah *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, DESTROY_AH);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static int __uv_attach_mcast(fid_t fid,
			struct ibv_attach_mcast *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, ATTACH_MCAST);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static int __uv_detach_mcast(fid_t fid,
			struct ibv_detach_mcast *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	UV_INIT_CMD(cmd, cmd_size, DETACH_MCAST);
	if (write(uv->fd, cmd, cmd_size) != cmd_size)
		return -errno;
	return 0;
}

static struct fi_ops_uverbs ops_uv = {
	.get_context = __uv_get_context,
	.query_device = __uv_query_device,
	.query_port = __uv_query_port,
	.alloc_pd = __uv_alloc_pd,
	.dealloc_pd = __uv_dealloc_pd,
	.open_xrcd = __uv_open_xrcd,
	.close_xrcd = __uv_close_xrcd,
	.reg_mr = __uv_reg_mr,
	.dereg_mr = __uv_dereg_mr,
	.create_comp_channel = __uv_create_comp_channel,
	.create_cq = __uv_create_cq,
	.poll_cq = __uv_poll_cq,
	.req_notify_cq = __uv_req_notify_cq,
	.resize_cq = __uv_resize_cq,
	.destroy_cq = __uv_destroy_cq,
	.create_srq = __uv_create_srq,
	.modify_srq = __uv_modify_srq,
	.query_srq = __uv_query_srq,
	.destroy_srq = __uv_destroy_srq,
	.create_qp = __uv_create_qp,
	.open_qp = __uv_open_qp,
	.query_qp = __uv_query_qp,
	.modify_qp = __uv_modify_qp,
	.destroy_qp = __uv_destroy_qp,
	.post_send = __uv_post_send,
	.post_recv = __uv_post_recv,
	.post_srq_recv = __uv_post_srq_recv,
	.create_ah = __uv_create_ah,
	.destroy_ah = __uv_destroy_ah,
	.attach_mcast = __uv_attach_mcast,
	.detach_mcast = __uv_detach_mcast
};

static int uv_close(fid_t fid)
{
	struct fid_uverbs *uv;

	uv = container_of(fid, struct fid_uverbs, fid);
	close(uv->fd);
	free(uv);
	return 0;
}

static struct fi_ops ops_fi = {
	.close = uv_close
};

static int uv_open(const char *res_name, const char *if_name,
	uint64_t flags, fid_t *fid, void *context)
{
	struct fid_uverbs *uv;
	char *dev_path;
	int ret = 0;

	if (!if_name || strncmp(FI_UVERBS_INTERFACE "/", if_name, 7))
		return -ENOSYS;

	if (asprintf(&dev_path, "/dev/infiniband%s", strstr(if_name, "/")) < 0)
		return -ENOMEM;

 	uv = calloc(1, sizeof(*uv));
 	if (!uv) {
 		ret = -ENOMEM;
 		goto out;
 	}

	uv->fd = open(dev_path, O_RDWR | O_CLOEXEC);
	if (uv->fd < 1) {
		ret = -errno;
		free(uv);
		goto out;
	}

	uv->fid.fclass = FID_CLASS_INTERFACE;
	uv->fid.size = sizeof(*uv);
	uv->fid.ops = &ops_fi;
	uv->fid.context = context;
	uv->ops = &ops_uv;

	*fid = &uv->fid;
out:
	free(dev_path);
	return ret;
}
