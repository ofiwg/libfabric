/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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

static inline int normalize_core_id(int core_id, int num_cores)
{
	if (core_id < 0)
		core_id += num_cores;

	if (core_id < 0)
		core_id = 0;

	if (core_id >= num_cores)
		core_id = num_cores - 1;

	return core_id;
}

static int psmx_progress_set_affinity(char *affinity)
{
	int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
	int core_id;
	cpu_set_t cpuset;
	char *triplet;
	int n, start, end, stride;
	int set_count = 0;

	if (!affinity) {
		FI_INFO(&psmx_prov, FI_LOG_CORE, "progress thread affinity not set\n");
		return 0;
	}

	CPU_ZERO(&cpuset);

	for (triplet = affinity; triplet; triplet = strchr(triplet, 'c')) {
		if (triplet[0] == ',')
			triplet++;

		stride = 1;
		n = sscanf(triplet, "%d:%d:%d", &start, &end, &stride);
		if (n < 1)
			continue;

		if (n < 2)
			end = start;
	
		if (stride < 1)
			stride = 1;

		start = normalize_core_id(start, num_cores);
		end = normalize_core_id(end, num_cores);

		for (core_id = start; core_id <= end; core_id += stride) {
			CPU_SET(core_id, &cpuset);
			set_count++;
		}

		FI_INFO(&psmx_prov, FI_LOG_CORE,
			"core set [%d:%d:%d] added to progress thread affinity set\n",
			start, end, stride);
	}

	if (set_count)
		pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	else
		FI_INFO(&psmx_prov, FI_LOG_CORE,
			"progress thread affinity not set due to invalid format\n");

	return set_count;
}

static void *psmx_progress_func(void *args)
{
	struct psmx_fid_domain *domain = args;
	int affinity_set;
	int sleep_usec;
	struct timespec ts;

	FI_INFO(&psmx_prov, FI_LOG_CORE, "\n");

	affinity_set = psmx_progress_set_affinity(psmx_env.prog_affinity);

	/* Negative sleep time means let the system choose the default.
	 * If affinity is set, sleep a short time to get better latency.
	 * If affinity is not set, short sleep time doesn't make difference.
	 */
	sleep_usec = psmx_env.prog_interval;
	if (sleep_usec < 0) {
		if (affinity_set)
			sleep_usec = 1;
		else
			sleep_usec = 1000;
	}

	ts.tv_sec = sleep_usec / 1000000;
	ts.tv_nsec = (sleep_usec % 1000000) * 1000;

	while (1) {
		psmx_progress(domain);
		nanosleep(&ts, NULL);
	}

	return NULL;
}

static void psmx_domain_start_progress(struct psmx_fid_domain *domain)
{
	int err;

	err = pthread_create(&domain->progress_thread, NULL,
			     psmx_progress_func, (void *)domain);
	if (err) {
		domain->progress_thread = pthread_self();
		FI_INFO(&psmx_prov, FI_LOG_CORE,
			"pthread_create returns %d\n", err);
	} else {
		FI_INFO(&psmx_prov, FI_LOG_CORE, "progress thread started\n");
	}
}

static void psmx_domain_stop_progress(struct psmx_fid_domain *domain)
{
	int err;
	void *exit_code;

	if (!pthread_equal(domain->progress_thread, pthread_self())) {
		err = pthread_cancel(domain->progress_thread);
		if (err) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"pthread_cancel returns %d\n", err);
		}
		err = pthread_join(domain->progress_thread, &exit_code);
		if (err) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"pthread_join returns %d\n", err);
		}
		else {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"progress thread exited with code %ld (%s)\n",
				(uintptr_t)exit_code,
				(exit_code == PTHREAD_CANCELED) ?
					"PTHREAD_CANCELED" : "?");
		}
	}
}

static int psmx_domain_close(fid_t fid)
{
	struct psmx_fid_domain *domain;
	int err;

	FI_INFO(&psmx_prov, FI_LOG_DOMAIN, "\n");

	domain = container_of(fid, struct psmx_fid_domain, domain.fid);

	if (--domain->refcnt > 0)
		return 0;

	if (domain->progress_thread_enabled)
		psmx_domain_stop_progress(domain);

	psmx_am_fini(domain);

	err = fastlock_destroy(&domain->poll_lock);
	if (err)
		FI_WARN(&psmx_prov, FI_LOG_CORE,
			"pthread_spin_destroy returns %d\n", err);

#if 0
	/* AM messages could arrive after MQ is finalized, causing segfault
	 * when trying to dereference the MQ pointer. There is no mechanism
	 * to properly shutdown AM. The workaround is to keep MQ valid.
	 */
	psm_mq_finalize(domain->psm_mq);
#endif

	/* workaround for:
	 * Assertion failure at psm_ep.c:1059: ep->mctxt_master == ep
	 */
	sleep(psmx_env.delay);

	err = psm_ep_close(domain->psm_ep, PSM_EP_CLOSE_GRACEFUL,
			   (int64_t) psmx_env.timeout * 1000000000LL);
	if (err != PSM_OK)
		psm_ep_close(domain->psm_ep, PSM_EP_CLOSE_FORCE, 0);

	domain->fabric->active_domain = NULL;
	free(domain);

	return 0;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_domain psmx_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = psmx_av_open,
	.cq_open = psmx_cq_open,
	.endpoint = psmx_ep_open,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = psmx_cntr_open,
	.poll_open = psmx_poll_open,
	.stx_ctx = psmx_stx_ctx,
	.srx_ctx = fi_no_srx_context,
};

int psmx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		     struct fid_domain **domain, void *context)
{
	struct psmx_fid_fabric *fabric_priv;
	struct psmx_fid_domain *domain_priv;
	struct psm_ep_open_opts opts;
	int err = -FI_ENOMEM;

	FI_INFO(&psmx_prov, FI_LOG_DOMAIN, "\n");

	fabric_priv = container_of(fabric, struct psmx_fid_fabric, fabric);
	if (fabric_priv->active_domain) {
		fabric_priv->active_domain->refcnt++;
		*domain = &fabric_priv->active_domain->domain;
		return 0;
	}

	if (!info->domain_attr->name || strncmp(info->domain_attr->name, PSMX_DOMAIN_NAME, PSMX_DOMAIN_NAME_LEN))
		return -FI_EINVAL;

	domain_priv = (struct psmx_fid_domain *) calloc(1, sizeof *domain_priv);
	if (!domain_priv)
		goto err_out;

	domain_priv->domain.fid.fclass = FI_CLASS_DOMAIN;
	domain_priv->domain.fid.context = context;
	domain_priv->domain.fid.ops = &psmx_fi_ops;
	domain_priv->domain.ops = &psmx_domain_ops;
	domain_priv->domain.mr = &psmx_mr_ops;
	domain_priv->mr_mode = info->domain_attr->mr_mode;
	domain_priv->mode = info->mode;
	domain_priv->caps = info->caps;
	domain_priv->fabric = fabric_priv;
	domain_priv->progress_thread_enabled =
		(info->domain_attr->data_progress == FI_PROGRESS_AUTO);

	psm_ep_open_opts_get_defaults(&opts);

	FI_INFO(&psmx_prov, FI_LOG_CORE, "uuid: %s\n", psmx_uuid_to_string(fabric_priv->uuid));

	err = psm_ep_open(fabric_priv->uuid, &opts,
			  &domain_priv->psm_ep, &domain_priv->psm_epid);
	if (err != PSM_OK) {
		FI_WARN(&psmx_prov, FI_LOG_CORE,
			"psm_ep_open returns %d, errno=%d\n", err, errno);
		err = psmx_errno(err);
		goto err_out_free_domain;
	}

	FI_INFO(&psmx_prov, FI_LOG_CORE, "epid: 0x%016lx\n", domain_priv->psm_epid);

	err = psm_mq_init(domain_priv->psm_ep, PSM_MQ_ORDERMASK_ALL,
			  NULL, 0, &domain_priv->psm_mq);
	if (err != PSM_OK) {
		FI_WARN(&psmx_prov, FI_LOG_CORE,
			"psm_mq_init returns %d, errno=%d\n", err, errno);
		err = psmx_errno(err);
		goto err_out_close_ep;
	}

	if (psmx_domain_enable_ep(domain_priv, NULL) < 0)
		goto err_out_finalize_mq;

	err = fastlock_init(&domain_priv->poll_lock);
	if (err) {
		FI_WARN(&psmx_prov, FI_LOG_CORE,
			"pthread_spin_init returns %d\n", err);
		goto err_out_finalize_mq;
	}

	if (domain_priv->progress_thread_enabled)
		psmx_domain_start_progress(domain_priv);

	domain_priv->refcnt = 1;
	fabric_priv->active_domain = domain_priv;
	*domain = &domain_priv->domain;
	return 0;

err_out_finalize_mq:
	psm_mq_finalize(domain_priv->psm_mq);

err_out_close_ep:
	if (psm_ep_close(domain_priv->psm_ep, PSM_EP_CLOSE_GRACEFUL,
			 (int64_t) psmx_env.timeout * 1000000000LL) != PSM_OK)
		psm_ep_close(domain_priv->psm_ep, PSM_EP_CLOSE_FORCE, 0);

err_out_free_domain:
	free(domain_priv);

err_out:
	return err;
}

int psmx_domain_check_features(struct psmx_fid_domain *domain, int ep_cap)
{
	if ((domain->caps & ep_cap & ~PSMX_SUB_CAPS) != (ep_cap & ~PSMX_SUB_CAPS)) {
		FI_INFO(&psmx_prov, FI_LOG_CORE,
			"caps mismatch: domain->caps=%llx, ep->caps=%llx, mask=%llx\n",
			domain->caps, ep_cap, ~PSMX_SUB_CAPS);
		return -FI_EOPNOTSUPP;
	}

	if ((ep_cap & FI_TAGGED) && domain->tagged_ep &&
	    fi_recv_allowed(ep_cap))
		return -FI_EBUSY;

	if ((ep_cap & FI_MSG) && domain->msg_ep &&
	    fi_recv_allowed(ep_cap))
		return -FI_EBUSY;

	if ((ep_cap & FI_RMA) && domain->rma_ep &&
	    fi_rma_target_allowed(ep_cap))
		return -FI_EBUSY;

	if ((ep_cap & FI_ATOMICS) && domain->atomics_ep &&
	    fi_rma_target_allowed(ep_cap))
		return -FI_EBUSY;

	return 0;
}

int psmx_domain_enable_ep(struct psmx_fid_domain *domain, struct psmx_fid_ep *ep)
{
	uint64_t ep_cap = 0;

	if (ep)
		ep_cap = ep->caps;

	if ((domain->caps & ep_cap & ~PSMX_SUB_CAPS) != (ep_cap & ~PSMX_SUB_CAPS)) {
		FI_INFO(&psmx_prov, FI_LOG_CORE,
			"caps mismatch: domain->caps=%llx, ep->caps=%llx, mask=%llx\n",
			domain->caps, ep_cap, ~PSMX_SUB_CAPS);
		return -FI_EOPNOTSUPP;
	}

	if (ep_cap & FI_MSG)
		domain->reserved_tag_bits |= PSMX_MSG_BIT;

	if (psmx_env.am_msg)
		domain->reserved_tag_bits &= ~PSMX_MSG_BIT;

	if ((ep_cap & FI_RMA) && psmx_env.tagged_rma)
		domain->reserved_tag_bits |= PSMX_RMA_BIT;

	if (((ep_cap & FI_RMA) || (ep_cap & FI_ATOMICS) || psmx_env.am_msg) &&
	    !domain->am_initialized) {
		int err = psmx_am_init(domain);
		if (err)
			return err;

		domain->am_initialized = 1;
	}

	if ((ep_cap & FI_RMA) && fi_rma_target_allowed(ep_cap))
		domain->rma_ep = ep;

	if ((ep_cap & FI_ATOMICS) && fi_rma_target_allowed(ep_cap))
		domain->atomics_ep = ep;

	if ((ep_cap & FI_TAGGED) && fi_recv_allowed(ep_cap))
		domain->tagged_ep = ep;

	if ((ep_cap & FI_MSG) && fi_recv_allowed(ep_cap))
		domain->msg_ep = ep;

	return 0;
}

void psmx_domain_disable_ep(struct psmx_fid_domain *domain, struct psmx_fid_ep *ep)
{
	if (!ep)
		return;

	if ((ep->caps & FI_RMA) && domain->rma_ep == ep)
		domain->rma_ep = NULL;

	if ((ep->caps & FI_ATOMICS) && domain->atomics_ep == ep)
		domain->atomics_ep = NULL;

	if ((ep->caps & FI_TAGGED) && domain->tagged_ep == ep)
		domain->tagged_ep = NULL;

	if ((ep->caps & FI_MSG) && domain->msg_ep == ep)
		domain->msg_ep = NULL;
}

