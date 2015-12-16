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

#include "psmx2.h"

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

static int psmx2_progress_set_affinity(char *affinity)
{
	int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
	int core_id;
	cpu_set_t cpuset;
	char *triplet;
	int n, start, end, stride;
	int set_count = 0;

	if (!affinity) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"progress thread affinity not set\n");
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

		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"core set [%d:%d:%d] added to progress thread affinity set\n",
			start, end, stride);
	}

	if (set_count)
		pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	else
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"progress thread affinity not set due to invalid format\n");

	return set_count;
}

static void *psmx2_progress_func(void *args)
{
	struct psmx2_fid_domain *domain = args;
	int affinity_set;
	int sleep_usec;
	struct timespec ts;

	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	affinity_set = psmx2_progress_set_affinity(psmx2_env.prog_affinity);

	/* Negative sleep time means let the system choose the default.
	 * If affinity is set, sleep a short time to get better latency.
	 * If affinity is not set, short sleep time doesn't make difference.
	 */
	sleep_usec = psmx2_env.prog_interval;
	if (sleep_usec < 0) {
		if (affinity_set)
			sleep_usec = 1;
		else
			sleep_usec = 1000;
	}

	ts.tv_sec = sleep_usec / 1000000;
	ts.tv_nsec = (sleep_usec % 1000000) * 1000;

	while (1) {
		psmx2_progress(domain);
		nanosleep(&ts, NULL);
	}

	return NULL;
}

static void psmx2_domain_start_progress(struct psmx2_fid_domain *domain)
{
	int err;

	err = pthread_create(&domain->progress_thread, NULL,
			     psmx2_progress_func, (void *)domain);
	if (err) {
		domain->progress_thread = pthread_self();
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"pthread_create returns %d\n", err);
	} else {
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "progress thread started\n");
	}
}

static void psmx2_domain_stop_progress(struct psmx2_fid_domain *domain)
{
	int err;
	void *exit_code;

	if (!pthread_equal(domain->progress_thread, pthread_self())) {
		err = pthread_cancel(domain->progress_thread);
		if (err) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"pthread_cancel returns %d\n", err);
		}
		err = pthread_join(domain->progress_thread, &exit_code);
		if (err) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"pthread_join returns %d\n", err);
		}
		else {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"progress thread exited with code %ld (%s)\n",
				(uintptr_t)exit_code,
				(exit_code == PTHREAD_CANCELED) ?
					"PTHREAD_CANCELED" : "?");
		}
	}
}

void psmx2_domain_release(struct psmx2_fid_domain *domain)
{
	int err;

	FI_INFO(&psmx2_prov, FI_LOG_DOMAIN, "refcnt=%d\n", domain->refcnt);

	if (--domain->refcnt > 0)
		return;

	if (domain->progress_thread_enabled)
		psmx2_domain_stop_progress(domain);

	psmx2_am_fini(domain);

	fastlock_destroy(&domain->poll_lock);
	fastlock_destroy(&domain->vl_lock);
	rbtDelete(domain->mr_map);
	fastlock_destroy(&domain->mr_lock);

#if 0
	/* AM messages could arrive after MQ is finalized, causing segfault
	 * when trying to dereference the MQ pointer. There is no mechanism
	 * to properly shutdown AM. The workaround is to keep MQ valid.
	 */
	psm2_mq_finalize(domain->psm2_mq);
#endif

	/* workaround for:
	 * Assertion failure at psm2_ep.c:1059: ep->mctxt_master == ep
	 */
	sleep(psmx2_env.delay);

	if (psmx2_env.timeout)
		err = psm2_ep_close(domain->psm2_ep, PSM2_EP_CLOSE_GRACEFUL,
				    (int64_t) psmx2_env.timeout * 1000000000LL);
	else
		err = PSM2_EP_CLOSE_TIMEOUT;

	if (err != PSM2_OK)
		psm2_ep_close(domain->psm2_ep, PSM2_EP_CLOSE_FORCE, 0);

	domain->fabric->active_domain = NULL;

	psmx2_fabric_release(domain->fabric);

	free(domain);
}

static int psmx2_domain_close(fid_t fid)
{
	struct psmx2_fid_domain *domain;

	FI_INFO(&psmx2_prov, FI_LOG_DOMAIN, "\n");

	domain = container_of(fid, struct psmx2_fid_domain, domain.fid);

	psmx2_domain_release(domain);

	return 0;
}

static struct fi_ops psmx2_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx2_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_domain psmx2_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = psmx2_av_open,
	.cq_open = psmx2_cq_open,
	.endpoint = psmx2_ep_open,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = psmx2_cntr_open,
	.poll_open = psmx2_poll_open,
	.stx_ctx = psmx2_stx_ctx,
	.srx_ctx = fi_no_srx_context,
};

static int psmx2_key_compare(void *key1, void *key2)
{
	return (key1 < key2) ?  -1 : (key1 > key2);
}

int psmx2_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		      struct fid_domain **domain, void *context)
{
	struct psmx2_fid_fabric *fabric_priv;
	struct psmx2_fid_domain *domain_priv;
	struct psm2_ep_open_opts opts;
	int err;

	FI_INFO(&psmx2_prov, FI_LOG_DOMAIN, "\n");

	fabric_priv = container_of(fabric, struct psmx2_fid_fabric, fabric);
	psmx2_fabric_acquire(fabric_priv);

	if (fabric_priv->active_domain) {
		psmx2_domain_acquire(fabric_priv->active_domain);
		*domain = &fabric_priv->active_domain->domain;
		return 0;
	}

	if (!info->domain_attr->name ||
	    strcmp(info->domain_attr->name, PSMX2_DOMAIN_NAME)) {
		err = -FI_EINVAL;
		goto err_out;
	}

	domain_priv = (struct psmx2_fid_domain *) calloc(1, sizeof *domain_priv);
	if (!domain_priv) {
		err = -FI_ENOMEM;
		goto err_out;
	}

	domain_priv->domain.fid.fclass = FI_CLASS_DOMAIN;
	domain_priv->domain.fid.context = context;
	domain_priv->domain.fid.ops = &psmx2_fi_ops;
	domain_priv->domain.ops = &psmx2_domain_ops;
	domain_priv->domain.mr = &psmx2_mr_ops;
	domain_priv->mr_mode = info->domain_attr->mr_mode;
	domain_priv->mode = info->mode;
	domain_priv->caps = info->caps;
	domain_priv->fabric = fabric_priv;
	domain_priv->progress_thread_enabled =
		(info->domain_attr->data_progress == FI_PROGRESS_AUTO);

	psm2_ep_open_opts_get_defaults(&opts);

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"uuid: %s\n", psmx2_uuid_to_string(fabric_priv->uuid));

	err = psm2_ep_open(fabric_priv->uuid, &opts,
			   &domain_priv->psm2_ep, &domain_priv->psm2_epid);
	if (err != PSM2_OK) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"psm2_ep_open returns %d, errno=%d\n", err, errno);
		err = psmx2_errno(err);
		goto err_out_free_domain;
	}

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"epid: 0x%016lx\n", domain_priv->psm2_epid);

	err = psm2_mq_init(domain_priv->psm2_ep, PSM2_MQ_ORDERMASK_ALL,
			   NULL, 0, &domain_priv->psm2_mq);
	if (err != PSM2_OK) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"psm2_mq_init returns %d, errno=%d\n", err, errno);
		err = psmx2_errno(err);
		goto err_out_close_ep;
	}

	err = fastlock_init(&domain_priv->mr_lock);
	if (err) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"fastlock_init(mr_lock) returns %d\n", err);
		goto err_out_finalize_mq;
	}

	domain_priv->mr_map = rbtNew(&psmx2_key_compare);
	if (!domain_priv->mr_map) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"rbtNew failed\n");
		goto err_out_destroy_mr_lock;
	}

	domain_priv->mr_reserved_key = 1;
	
	err = fastlock_init(&domain_priv->vl_lock);
	if (err) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"fastlock_init(vl_lock) returns %d\n", err);
		goto err_out_delete_mr_map;
	}
	memset(domain_priv->vl_map, 0, sizeof(domain_priv->vl_map));
	domain_priv->vl_alloc = 0;

	err = fastlock_init(&domain_priv->poll_lock);
	if (err) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"fastlock_init(poll_lock) returns %d\n", err);
		goto err_out_destroy_vl_lock;
	}

	/* Set active domain before psmx2_domain_enable_ep() installs the
	 * AM handlers to ensure that psmx2_active_fabric->active_domain
	 * is always non-NULL inside the handlers. Notice that the vlaue
	 * active_domain becomes NULL again only when the domain is closed.
	 * At that time the AM handlers are gone with the PSM endpoint.
	 */
	fabric_priv->active_domain = domain_priv;

	if (psmx2_domain_enable_ep(domain_priv, NULL) < 0)
		goto err_out_reset_active_domain;

	if (domain_priv->progress_thread_enabled)
		psmx2_domain_start_progress(domain_priv);

	domain_priv->refcnt = 1;
	*domain = &domain_priv->domain;
	return 0;

err_out_reset_active_domain:
	fabric_priv->active_domain = NULL;
	fastlock_destroy(&domain_priv->poll_lock);

err_out_destroy_vl_lock:
	fastlock_destroy(&domain_priv->vl_lock);

err_out_delete_mr_map:
	rbtDelete(domain_priv->mr_map);

err_out_destroy_mr_lock:
	fastlock_destroy(&domain_priv->mr_lock);

err_out_finalize_mq:
	psm2_mq_finalize(domain_priv->psm2_mq);

err_out_close_ep:
	if (psm2_ep_close(domain_priv->psm2_ep, PSM2_EP_CLOSE_GRACEFUL,
			  (int64_t) psmx2_env.timeout * 1000000000LL) != PSM2_OK)
		psm2_ep_close(domain_priv->psm2_ep, PSM2_EP_CLOSE_FORCE, 0);

err_out_free_domain:
	free(domain_priv);

err_out:
	psmx2_fabric_release(fabric_priv);
	return err;
}

int psmx2_domain_check_features(struct psmx2_fid_domain *domain, int ep_cap)
{
	if ((domain->caps & ep_cap & ~PSMX2_SUB_CAPS) !=
	    (ep_cap & ~PSMX2_SUB_CAPS)) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"caps mismatch: domain->caps=%llx, "
			"ep->caps=%llx, mask=%llx\n",
			domain->caps, ep_cap, ~PSMX2_SUB_CAPS);
		return -FI_EOPNOTSUPP;
	}

	return 0;
}

int psmx2_domain_enable_ep(struct psmx2_fid_domain *domain,
			   struct psmx2_fid_ep *ep)
{
	uint64_t ep_cap = 0;

	if (ep)
		ep_cap = ep->caps;

	if ((domain->caps & ep_cap & ~PSMX2_SUB_CAPS) !=
	    (ep_cap & ~PSMX2_SUB_CAPS)) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"caps mismatch: domain->caps=%llx, "
			"ep->caps=%llx, mask=%llx\n",
			domain->caps, ep_cap, ~PSMX2_SUB_CAPS);
		return -FI_EOPNOTSUPP;
	}

	if (((ep_cap & FI_RMA) || (ep_cap & FI_ATOMICS)) &&
	    !domain->am_initialized) {
		int err = psmx2_am_init(domain);
		if (err)
			return err;

		domain->am_initialized = 1;
	}

	return 0;
}

