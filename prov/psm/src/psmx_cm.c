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

static int psmx_cm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	if (!fid_ep->domain)
		return -EBADF;

	if (*addrlen < sizeof(psm_epid_t))
		return -FI_ETOOSMALL;

	*(psm_epid_t *)addr = fid_ep->domain->psm_epid;
	*addrlen = sizeof(psm_epid_t);

	return 0;
}

static int psmx_cm_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);
	if (!fid_ep->domain)
		return -EBADF;

	if (!fid_ep->connected)
		return -ENOTCONN;

	if (*addrlen < sizeof(psm_epid_t))
		return -FI_ETOOSMALL;

	*(psm_epid_t *)addr = fid_ep->peer_psm_epid;
	*addrlen = sizeof(psm_epid_t);

	return 0;
}

static int psmx_cm_connect(struct fid_ep *ep, const void *addr,
			   const void *param, size_t paramlen)
{
	struct psmx_fid_ep *fid_ep;
	psm_epid_t epid;
	psm_epaddr_t epaddr;
	int err;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);
	if (!fid_ep->domain)
		return -EBADF;

	epid = (psm_epid_t)addr;
	err = psmx_epid_to_epaddr(fid_ep->domain, epid, &epaddr);
	if (err)
		return err;

	fid_ep->connected = 1;
	fid_ep->peer_psm_epid = epid;
	fid_ep->peer_psm_epaddr = epaddr;

	return 0;
}

static int psmx_cm_listen(struct fid_pep *pep)
{
	return -ENOSYS;
}

static int psmx_cm_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	return -ENOSYS;
}

static int psmx_cm_reject(struct fid_pep *pep, struct fi_info *info,
			const void *param, size_t paramlen)
{
	return -ENOSYS;
}

static int psmx_cm_shutdown(struct fid_ep *ep, uint64_t flags)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);
	if (!fid_ep->domain)
		return -EBADF;

	if (!fid_ep->connected)
		return -ENOTCONN;

	fid_ep->connected = 0;
	fid_ep->peer_psm_epid = 0;
	fid_ep->peer_psm_epaddr = 0;

	return 0;
}

static int psmx_cm_join(struct fid_ep *ep, void *addr, void **fi_addr, uint64_t flags)
{
	return -ENOSYS;
}

static int psmx_cm_leave(struct fid_ep *ep, void *addr, void *fi_addr, uint64_t flags)
{
	return -ENOSYS;
}

struct fi_ops_cm psmx_cm_ops = {
	.getname = psmx_cm_getname,
	.getpeer = psmx_cm_getpeer,
	.connect = psmx_cm_connect,
	.listen = psmx_cm_listen,
	.accept = psmx_cm_accept,
	.reject = psmx_cm_reject,
	.shutdown = psmx_cm_shutdown,
	.join = psmx_cm_join,
	.leave = psmx_cm_leave,
};

