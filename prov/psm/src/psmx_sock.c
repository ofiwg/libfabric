/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#include "psmx.h"

static ssize_t psmx_sock_cancel(fid_t fid, struct fi_context *context)
{
	struct psmx_fid_socket *fid_socket;
	int err;

	fid_socket = container_of(fid, struct psmx_fid_socket, socket.fid);
	if (!fid_socket->domain)
		return -EBADF;

	if (!context)
		return -EINVAL;

	if (context->internal[0] == NULL)
		return 0;

	err = psm_mq_cancel((psm_mq_req_t *)&context->internal[0]);
	return psmx_errno(err);
}

static int psmx_sock_getopt(fid_t fid, int level, int optname,
			void *optval, size_t *optlen)
{
	struct psmx_fid_socket *fid_socket;
	uint32_t size;
	int err;

	fid_socket = container_of(fid, struct psmx_fid_socket, socket.fid);

	if (level != FI_OPT_SOCKET)
		return -ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MAX_BUFFERED_SEND:
		if (!optval)
			return 0;

		if (!optlen || *optlen < sizeof(size_t))
			return -EINVAL;

		if (!fid_socket->domain)
			return -EBADF;

		/* FIXME:
		 * PSM has different thresholds for fabric & shm data path. Just use
		 * the value of the fabric path at this point.
		 */
		err = psm_mq_getopt(fid_socket->domain->psm_mq, PSM_MQ_RNDV_IPATH_SZ, &size);
		if (err)
			return psmx_errno(err);

		*(size_t *)optval = size;
		*optlen = sizeof(size_t);
		break;

	default:
		return -ENOPROTOOPT;
	}

	return 0;
}

static int psmx_sock_setopt(fid_t fid, int level, int optname,
			const void *optval, size_t optlen)
{
	struct psmx_fid_socket *fid_socket;
	uint32_t size;
	int err;

	if (level != FI_OPT_SOCKET)
		return -ENOPROTOOPT;

	fid_socket = container_of(fid, struct psmx_fid_socket, socket.fid);
	switch (optname) {
	case FI_OPT_MAX_BUFFERED_SEND:
		if (!optval)
			return -EFAULT;

		if (optlen != sizeof(size_t))
			return -EINVAL;

		if (!fid_socket->domain)
			return -EBADF;

		/* FIXME:
		 * PSM has different thresholds for fabric & shm data path. Only set
		 * the value of the fabric path at this point.
		 */
		size = *(size_t *)optval;
		err = psm_mq_setopt(fid_socket->domain->psm_mq, PSM_MQ_RNDV_IPATH_SZ, &size);
		if (err)
			return psmx_errno(err);
		break;

	default:
		return -ENOPROTOOPT;
	}

	return 0;
}

static int psmx_sock_close(fid_t fid)
{
	struct psmx_fid_socket *fid_socket;

	fid_socket = container_of(fid, struct psmx_fid_socket, socket.fid);
	free(fid_socket);

	return 0;
}

static int psmx_sock_bind(fid_t fid, struct fi_resource *ress, int nress)
{
	int i;
	struct psmx_fid_socket *fid_socket;
	struct psmx_fid_domain *domain;
	struct psmx_fid_av *av;
	struct psmx_fid_ec *ec;

	fid_socket = container_of(fid, struct psmx_fid_socket, socket.fid);

	for (i=0; i<nress; i++) {
		if (!ress[i].fid)
			return -EINVAL;
		switch (ress[i].fid->fclass) {
		case FID_CLASS_RESOURCE_DOMAIN:
			domain = container_of(ress[i].fid,
					struct psmx_fid_domain, domain.fid);
			if (fid_socket->domain && fid_socket->domain != domain)
				return -EEXIST;
			fid_socket->domain = domain;
			break;

		case FID_CLASS_EC:
			/* TODO: check ress flags for send/recv EQs */
			ec = container_of(ress[i].fid,
					struct psmx_fid_ec, ec.fid);
			if (fid_socket->ec && fid_socket->ec != ec)
				return -EEXIST;
			if (fid_socket->domain && fid_socket->domain != ec->domain)
				return -EINVAL;
			fid_socket->ec = ec;
			fid_socket->domain = ec->domain;
			break;

		case FID_CLASS_AV:
			av = container_of(ress[i].fid,
					struct psmx_fid_av, av.fid);
			if (fid_socket->av && fid_socket->av != av)
				return -EEXIST;
			if (fid_socket->domain && fid_socket->domain != av->domain)
				return -EINVAL;
			fid_socket->av = av;
			fid_socket->domain = av->domain;
			break;

		default:
			return -ENOSYS;
		}
	}

	return 0;
}

static int psmx_sock_sync(fid_t fid, uint64_t flags, void *context)
{
	return -ENOSYS;
}

static int psmx_sock_control(fid_t fid, int command, void *arg)
{
	return -ENOSYS;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_sock_close,
	.bind = psmx_sock_bind,
	.sync = psmx_sock_sync,
	.control = psmx_sock_control,
};

static struct fi_ops_sock psmx_sock_ops = {
	.size = sizeof(struct fi_ops_sock),
	.cancel = psmx_sock_cancel,
	.getopt = psmx_sock_getopt,
	.setopt = psmx_sock_setopt,
};

int psmx_sock_open(struct fi_info *info, fid_t *fid, void *context)
{
	struct psmx_fid_socket *fid_socket;

	fid_socket = (struct psmx_fid_socket *) calloc(1, sizeof *fid_socket);
	if (!fid_socket)
		return -ENOMEM;

	fid_socket->socket.fid.size = sizeof(struct fid_socket);
	fid_socket->socket.fid.fclass = FID_CLASS_SOCKET;
	fid_socket->socket.fid.context = context;
	fid_socket->socket.fid.ops = &psmx_fi_ops;
	fid_socket->socket.ops = &psmx_sock_ops;
	fid_socket->socket.cm = &psmx_cm_ops;
	fid_socket->socket.tagged = &psmx_tagged_ops;

	if (info)
		fid_socket->flags = info->flags;

	*fid = &fid_socket->socket.fid;

	return 0;
}

