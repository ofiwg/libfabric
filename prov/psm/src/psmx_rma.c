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

static int psmx_readfrom(fid_t fid, void *buf, size_t len,
				const void *src_addr, uint64_t addr,
				be64_t key, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	return -ENOSYS;
}

static int psmx_readmemfrom(fid_t fid, void *buf, size_t len,
				uint64_t mem_desc, const void *src_addr,
				uint64_t addr, be64_t key, void *context)
{
	return -ENOSYS;
}

static int psmx_readmsg(fid_t fid, const struct fi_msg_rma *msg,
				uint64_t flags)
{
	return -ENOSYS;
}

static int psmx_read(fid_t fid, void *buf, size_t len, uint64_t addr,
				be64_t key, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	if (!fid_ep->connected)
		return -ENOTCONN;

	return psmx_readfrom(fid, buf, len, fid_ep->peer_psm_epaddr,
					addr, key, context);
}

static int psmx_readmem(fid_t fid, void *buf, size_t len,
				uint64_t mem_desc, uint64_t addr,
				be64_t key, void *context)
{
	return -ENOSYS;
}

static int psmx_readv(fid_t fid, const void *iov, size_t count,
				uint64_t addr, be64_t key, void *context)
{
	return -ENOSYS;
}

static int psmx_writeto(fid_t fid, const void *buf, size_t len,
				const void *dest_addr, uint64_t addr,
				be64_t key, void *context)
{
	struct psmx_fid_ep *fid_ep;
	psm_epaddr_t psm_epaddr;
	int flags;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	psm_epaddr = (psm_epaddr_t) dest_addr;

	flags = fid_ep->flags;

	return -ENOSYS;
}

static int psmx_writememto(fid_t fid, const void *buf, size_t len,
				uint64_t mem_desc, const void *dest_addr,
				uint64_t addr, be64_t key, void *context)
{
	return -ENOSYS;
}

static int psmx_writemsg(fid_t fid, const struct fi_msg_rma *msg,
				uint64_t flags)
{
	return -ENOSYS;
}

static int psmx_write(fid_t fid, const void *buf, size_t len,
				uint64_t addr, be64_t key, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	if (!fid_ep->connected)
		return -ENOTCONN;

	return psmx_writeto(fid, buf, len, fid_ep->peer_psm_epaddr,
				addr, key, context);
}

static int psmx_writemem(fid_t fid, const void *buf, size_t len,
				uint64_t mem_desc, uint64_t addr,
				be64_t key, void *context)
{
	return -ENOSYS;
}

static int psmx_writev(fid_t fid, const void *iov, size_t count,
				uint64_t addr, be64_t key, void *context)
{
	return -ENOSYS;
}

struct fi_ops_rma psmx_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = psmx_read,
	.readmem = psmx_readmem,
	.readv = psmx_readv,
	.readfrom = psmx_readfrom,
	.readmemfrom = psmx_readmemfrom,
	.readmsg = psmx_readmsg,
	.write = psmx_write,
	.writemem = psmx_writemem,
	.writev = psmx_writev,
	.writeto = psmx_writeto,
	.writememto = psmx_writememto,
	.writemsg = psmx_writemsg,
};

