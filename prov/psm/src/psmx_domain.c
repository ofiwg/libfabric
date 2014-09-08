/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

static int psmx_domain_close(fid_t fid)
{
	struct psmx_fid_domain *fid_domain;
	int err;

	fid_domain = container_of(fid, struct psmx_fid_domain, domain.fid);

#if PSMX_USE_AM
	psmx_am_fini(fid_domain);
#endif

	if (fid_domain->ns_thread) {
		pthread_cancel(fid_domain->ns_thread);
		pthread_join(fid_domain->ns_thread, NULL);
	}

#if PSMX_USE_AM
	/* AM messages could arrive after MQ is finalized, causing segfault
	 * when trying to dereference the MQ pointer. There is no mechanism
	 * to properly shutdown AM. The workaround is to keep MQ valid.
	 */
	if (0)
#endif
	psm_mq_finalize(fid_domain->psm_mq);

	err = psm_ep_close(fid_domain->psm_ep, PSM_EP_CLOSE_GRACEFUL,
			   (int64_t) PSMX_TIME_OUT * 1000000000LL);
	if (err != PSM_OK)
		psm_ep_close(fid_domain->psm_ep, PSM_EP_CLOSE_FORCE, 0);

	free(fid_domain);

	return 0;
}

static int psmx_domain_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	return -ENOSYS;
}

static int psmx_domain_sync(fid_t fid, uint64_t flags, void *context)
{
	return -ENOSYS;
}

static int psmx_domain_control(fid_t fid, int command, void *arg)
{
	return -ENOSYS;
}

static int psmx_domain_query(struct fid_domain *domain,
			     struct fi_domain_attr *attr, size_t *attrlen)
{
	return -ENOSYS;
}

static int psmx_if_open(struct fid_domain *domain, const char *name, uint64_t flags,
			struct fid **fif, void *context)
{
	return -ENOSYS;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_domain_close,
	.bind = psmx_domain_bind,
	.sync = psmx_domain_sync,
	.control = psmx_domain_control,
};

static struct fi_ops_domain psmx_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.query = psmx_domain_query,
	.av_open = psmx_av_open,
	.eq_open = psmx_eq_open,
	.endpoint = psmx_ep_open,
	.if_open = psmx_if_open,
	.cntr_open = psmx_cntr_open,
};

int psmx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		     struct fid_domain **domain, void *context)
{
	struct psmx_fid_domain *fid_domain;
	struct psm_ep_open_opts opts;
	int err = -ENOMEM;
	char *s;

	psmx_debug("%s\n", __func__);

	if (!info->domain_attr->name || strncmp(info->domain_attr->name, "psm", 3))
		return -EINVAL;

	psmx_query_mpi();

	fid_domain = (struct psmx_fid_domain *) calloc(1, sizeof *fid_domain);
	if (!fid_domain)
		goto err_out;

	fid_domain->domain.fid.fclass = FID_CLASS_DOMAIN;
	fid_domain->domain.fid.context = context;
	fid_domain->domain.fid.ops = &psmx_fi_ops;
	fid_domain->domain.ops = &psmx_domain_ops;
	fid_domain->domain.mr = &psmx_mr_ops;

	psm_ep_open_opts_get_defaults(&opts);

	err = psm_ep_open(info->auth_key, &opts,
			  &fid_domain->psm_ep, &fid_domain->psm_epid);
	if (err != PSM_OK) {
		fprintf(stderr, "%s: psm_ep_open returns %d, errno=%d\n",
			__func__, err, errno);
		err = psmx_errno(err);
		goto err_out_free_domain;
	}

	err = psm_mq_init(fid_domain->psm_ep, PSM_MQ_ORDERMASK_ALL,
			  NULL, 0, &fid_domain->psm_mq);
	if (err != PSM_OK) {
		fprintf(stderr, "%s: psm_mq_init returns %d, errno=%d\n",
			__func__, err, errno);
		err = psmx_errno(err);
		goto err_out_close_ep;
	}

	fid_domain->ns_port = psmx_uuid_to_port(info->auth_key);

	s = getenv("SFI_PSM_NAME_SERVER");
	if (s && (!strcasecmp(s, "yes") || !strcasecmp(s, "on") || !strcmp(s, "1")))
		err = pthread_create(&fid_domain->ns_thread, NULL, psmx_name_server, (void *)fid_domain);
	else
		err = -1;

	if (err)
		fid_domain->ns_thread = 0;

	if (info->ep_cap & FI_MSG)
		fid_domain->reserved_tag_bits |= PSMX_MSG_BIT;

#if PSMX_USE_AM
	s = getenv("SFI_PSM_AM_MSG");
	if (s && (!strcasecmp(s, "yes") || !strcasecmp(s, "on") || !strcmp(s, "1")))
		psmx_am_msg_enabled = 1;

	s = getenv("SFI_PSM_TAGGED_RMA");
	if (s && (!strcasecmp(s, "yes") || !strcasecmp(s, "on") || !strcmp(s, "1")))
		psmx_am_tagged_rma = 1;

	if (psmx_am_msg_enabled)
		fid_domain->reserved_tag_bits &= ~PSMX_MSG_BIT;

	if ((info->ep_cap & FI_RMA) && psmx_am_tagged_rma)
		fid_domain->reserved_tag_bits |= PSMX_RMA_BIT;

	if ((info->ep_cap & FI_RMA) ||
	    (info->ep_cap & FI_ATOMICS) ||
	    psmx_am_msg_enabled) {
		err = psmx_am_init(fid_domain);
		if (err) {
			if (fid_domain->ns_thread) {
				pthread_cancel(fid_domain->ns_thread);
				pthread_join(fid_domain->ns_thread, NULL);
			}
			psm_mq_finalize(fid_domain->psm_mq);
			goto err_out_close_ep;
		}
	}
#endif
	fid_domain->ep_cap = info->ep_cap;

	*domain = &fid_domain->domain;
	return 0;

err_out_close_ep:
	if (psm_ep_close(fid_domain->psm_ep, PSM_EP_CLOSE_GRACEFUL,
			 (int64_t) PSMX_TIME_OUT * 1000000000LL) != PSM_OK)
		psm_ep_close(fid_domain->psm_ep, PSM_EP_CLOSE_FORCE, 0);

err_out_free_domain:
	free(fid_domain);

err_out:
	return err;
}

