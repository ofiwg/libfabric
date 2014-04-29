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

static struct fi_ec_err_entry error_ece;
static int error_state = 0;

static ssize_t psmx_ec_readfrom(fid_t fid, void *buf, size_t len,
				void *src_addr, size_t *addrlen)
{
	struct psmx_fid_ec *fid_ec;
	psm_mq_req_t psm_req;
	psm_mq_status_t psm_status;
	struct fi_ec_tagged_entry *ece;
	int err;

	fid_ec = container_of(fid, struct psmx_fid_ec, ec.fid);
	assert(fid_ec->domain);
	assert(fid_ec->format == FI_EC_FORMAT_TAGGED);

	if (len < sizeof *ece)
		return -FI_ETOOSMALL;

	err = psm_mq_ipeek(fid_ec->domain->psm_mq, &psm_req, NULL);
	if (err == PSM_OK) {
		err = psm_mq_test(&psm_req, &psm_status);

		if (psm_status.error_code) {
			error_ece.fid_context = fid_ec->ec.fid.context;
			error_ece.op_context = psm_status.context;
			error_ece.flags = 0;
			error_ece.err = psmx_errno(psm_status.error_code);
			error_ece.prov_errno = psm_status.error_code;
			error_ece.data = 0;
			error_ece.prov_data = NULL;
			error_state = 1;
			return error_ece.err;
		}

		ece = (struct fi_ec_tagged_entry *) buf;
		ece->op_context = psm_status.context;
		ece->flags = 0;
		ece->len = psm_status.nbytes;
		ece->data = 0;
		ece->tag = psm_status.msg_tag;
		ece->olen = psm_status.msg_length;

		if (src_addr) {
			if ((fid_ec->domain->reserved_tag_bits & PSMX_NONMATCH_BIT) &&
				psm_status.msg_tag & PSMX_NONMATCH_BIT) {
				err = psmx_epid_to_epaddr(
					fid_ec->domain->psm_ep,
					psm_status.msg_tag & ~PSMX_NONMATCH_BIT,
					src_addr);
			}
		}

		return 1;
	} else if (err == PSM_MQ_NO_COMPLETIONS) {
		return 0;
	} else {
		return -1;
	}
}

static ssize_t psmx_ec_read(fid_t fid, void *buf, size_t len)
{
	return psmx_ec_readfrom(fid, buf, len, NULL, NULL);
}

static ssize_t psmx_ec_readerr(fid_t fid, void *buf, size_t len, uint64_t flags)
{
	if (len < sizeof(error_ece))
		return -FI_ETOOSMALL;

	*(struct fi_ec_err_entry *)buf = error_ece;
	error_state = 0;

	return 0;
}

static ssize_t psmx_ec_write(fid_t fid, const void *buf, size_t len)
{
	return -ENOSYS;
}

static int psmx_ec_reset(fid_t fid, const void *cond)
{
	return -ENOSYS;
}

static ssize_t psmx_ec_condread(fid_t fid, void *buf, size_t len, const void *cond)
{
	return -ENOSYS;
}

static ssize_t psmx_ec_condreadfrom(fid_t fid, void *buf, size_t len,
				    void *src_addr, size_t *addrlen, const void *cond)
{
	return -ENOSYS;
}

static const char *psmx_ec_strerror(fid_t fid, int prov_errno, const void *prov_data,
				    void *buf, size_t len)
{
	return psm_error_get_string(prov_errno);
}

static int psmx_ec_close(fid_t fid)
{
	struct psmx_fid_ec *fid_ec;

	fid_ec = container_of(fid, struct psmx_fid_ec, ec.fid);
	free(fid_ec);

	return 0;
}

static int psmx_ec_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	return -ENOSYS;
}

static int psmx_ec_sync(fid_t fid, uint64_t flags, void *context)
{
	return -ENOSYS;
}

static int psmx_ec_control(fid_t fid, int command, void *arg)
{
	return -ENOSYS;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_ec_close,
	.bind = psmx_ec_bind,
	.sync = psmx_ec_sync,
	.control = psmx_ec_control,
};

static struct fi_ops_ec psmx_ec_ops = {
	.size = sizeof(struct fi_ops_ec),
	.read = psmx_ec_read,
	.readfrom = psmx_ec_readfrom,
	.readerr = psmx_ec_readerr,
	.write = psmx_ec_write,
	.reset = psmx_ec_reset,
	.condread = psmx_ec_condread,
	.condreadfrom = psmx_ec_condreadfrom,
	.strerror = psmx_ec_strerror,
};

int psmx_ec_open(fid_t fid, struct fi_ec_attr *attr, fid_t *ec, void *context)
{
	struct psmx_fid_domain *fid_domain;
	struct psmx_fid_ec *fid_ec;

	if (attr->domain != FI_EC_DOMAIN_GENERAL && attr->domain != FI_EC_DOMAIN_COMP)
		return -ENOSYS;

	if (attr->type != FI_EC_QUEUE)
		return -ENOSYS;

	if (attr->format != FI_EC_FORMAT_TAGGED && attr->format != FI_EC_FORMAT_UNSPEC)
		return -ENOSYS;

	fid_domain = container_of(fid, struct psmx_fid_domain, domain.fid);
	fid_ec = (struct psmx_fid_ec *) calloc(1, sizeof *fid_ec);
	if (!fid_ec)
		return -ENOMEM;

	fid_ec->domain = fid_domain;
	fid_ec->type = FI_EC_QUEUE;
	fid_ec->format = FI_EC_FORMAT_TAGGED;
	fid_ec->ec.fid.size = sizeof(struct fid_ec);
	fid_ec->ec.fid.fclass = FID_CLASS_EC;
	fid_ec->ec.fid.context = context;
	fid_ec->ec.fid.ops = &psmx_fi_ops;
	fid_ec->ec.ops = &psmx_ec_ops;

	*ec = &fid_ec->ec.fid;
	return 0;
}

