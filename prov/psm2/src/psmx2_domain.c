/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

static int psmx2_trx_ctxt_cnt = 0;

/*
 * Tx/Rx context disconnect protocol:
 *
 * TRX_CTXT disconnect REQ:
 *	args[0].u32w0	cmd
 *
 * Before a PSM2 endpoint is closed, a TRX_CTXT disconnect REQ is sent to
 * all connected peers. Each peer then calls psm2_ep_disconnet() to clean
 * up the local connection state. This allows a future endpoint with the
 * same epid to connect to the same peers.
 */

static psm2_error_t (*psmx2_disconnect_psm2_ep)(psm2_ep_t ep,
						int epaddr_cnt,
						psm2_epaddr_t *epaddrs,
						const int *masks,
						psm2_error_t *errors,
						int64_t timeout);

struct disconnect_args {
	psm2_ep_t	ep;
	psm2_epaddr_t	epaddr;
};

static void *disconnect_func(void *args)
{
	struct disconnect_args *disconn = args;
	psm2_error_t errors;

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"psm2_ep: %p, epaddr: %p\n", disconn->ep, disconn->epaddr);

	(*psmx2_disconnect_psm2_ep)(disconn->ep, 1, &disconn->epaddr, NULL,
				    &errors, 5*1e9);
	free(args);
	return NULL;
}

static int psmx2_peer_match(struct dlist_entry *item, const void *arg)
{
	struct psmx2_epaddr_context *peer;

	peer = container_of(item, struct psmx2_epaddr_context, entry);
	return  (peer->epaddr == arg);
}

int psmx2_am_trx_ctxt_handler_ext(psm2_am_token_t token, psm2_amarg_t *args,
				  int nargs, void *src, uint32_t len,
				  struct psmx2_trx_ctxt *trx_ctxt)
{
	psm2_epaddr_t epaddr;
	int err = 0;
	int cmd;
	struct disconnect_args *disconn;
	pthread_t disconnect_thread;

	psm2_am_get_source(token, &epaddr);
	cmd = PSMX2_AM_GET_OP(args[0].u32w0);

	switch(cmd) {
	case PSMX2_AM_REQ_TRX_CTXT_DISCONNECT:
		if (psmx2_disconnect_psm2_ep) {
			/*
			 * we can't call psm2_ep_disconnect from the AM
			 * handler. instead, create a thread to do the work.
			 * the performance of this operation is not important.
			 */
			disconn = malloc(sizeof(*disconn));
			if (disconn) {
				psmx2_lock(&trx_ctxt->peer_lock, 2);
				dlist_remove_first_match(&trx_ctxt->peer_list,
							 psmx2_peer_match, epaddr);
				psmx2_unlock(&trx_ctxt->peer_lock, 2);
				disconn->ep = trx_ctxt->psm2_ep;
				disconn->epaddr = epaddr;
				pthread_create(&disconnect_thread, NULL,
					       disconnect_func, disconn);
				pthread_detach(disconnect_thread);
			}
		}
		break;

	default:
		err = -FI_EINVAL;
		break;
	}

	return err;
}

void psmx2_trx_ctxt_disconnect_peers(struct psmx2_trx_ctxt *trx_ctxt)
{
	struct dlist_entry *item, *tmp;
	struct psmx2_epaddr_context *peer;
	psm2_amarg_t arg;

	arg.u32w0 = PSMX2_AM_REQ_TRX_CTXT_DISCONNECT;

	psmx2_lock(&trx_ctxt->peer_lock, 2);
	dlist_foreach_safe(&trx_ctxt->peer_list, item, tmp) {
		dlist_remove(item);
		peer = container_of(item, struct psmx2_epaddr_context, entry);
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "epaddr: %p\n", peer->epaddr);
		psm2_am_request_short(peer->epaddr, PSMX2_AM_TRX_CTXT_HANDLER,
				      &arg, 1, NULL, 0, 0, NULL, NULL);
		psm2_epaddr_setctxt(peer->epaddr, NULL);
		free(peer);
	}
	psmx2_unlock(&trx_ctxt->peer_lock, 2);
}

void psmx2_trx_ctxt_free(struct psmx2_trx_ctxt *trx_ctxt)
{
	int err;

	if (!trx_ctxt)
		return;

	psmx2_trx_ctxt_disconnect_peers(trx_ctxt);

	if (trx_ctxt->am_initialized)
		psmx2_am_fini(trx_ctxt);

#if 0
	/* AM messages could arrive after MQ is finalized, causing segfault
	 * when trying to dereference the MQ pointer. There is no mechanism
	 * to properly shutdown AM. The workaround is to keep MQ valid.
	 */
	psm2_mq_finalize(trx_ctxt->psm2_mq);
#endif

	/* workaround for:
	 * Assertion failure at psm2_ep.c:1059: ep->mctxt_master == ep
	 */
	sleep(psmx2_env.delay);

	if (psmx2_env.timeout)
		err = psm2_ep_close(trx_ctxt->psm2_ep, PSM2_EP_CLOSE_GRACEFUL,
				    (int64_t) psmx2_env.timeout * 1000000000LL);
	else
		err = PSM2_EP_CLOSE_TIMEOUT;

	if (err != PSM2_OK)
		psm2_ep_close(trx_ctxt->psm2_ep, PSM2_EP_CLOSE_FORCE, 0);

	util_buf_pool_destroy(trx_ctxt->am_req_pool);
	fastlock_destroy(&trx_ctxt->am_req_pool_lock);
	fastlock_destroy(&trx_ctxt->poll_lock);
	fastlock_destroy(&trx_ctxt->peer_lock);
	free(trx_ctxt);
}

struct psmx2_trx_ctxt *psmx2_trx_ctxt_alloc(struct psmx2_fid_domain *domain,
					    struct psmx2_ep_name *src_addr,
					    int sep_ctxt_idx)
{
	struct psmx2_trx_ctxt *trx_ctxt;
	struct psm2_ep_open_opts opts;
	int should_retry = 0;
	int err;

	if (psmx2_trx_ctxt_cnt >= psmx2_env.max_trx_ctxt) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"number of Tx/Rx contexts exceeds limit (%d).\n",
			psmx2_env.max_trx_ctxt);
		return NULL;
	}

	trx_ctxt = calloc(1, sizeof(*trx_ctxt));
	if (!trx_ctxt) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"failed to allocate trx_ctxt.\n");
		return NULL;
	}

	trx_ctxt->am_req_pool = util_buf_pool_create(
					sizeof(struct psmx2_am_request),
					sizeof(void *),
					0, /* max_cnt: unlimited */
					64); /* chunk_cnt */
	if (!trx_ctxt->am_req_pool) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"failed to allocate am_req_pool.\n");
		goto err_out;
	}

	psm2_ep_open_opts_get_defaults(&opts);
	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"uuid: %s\n", psmx2_uuid_to_string(domain->fabric->uuid));

	opts.unit = src_addr ? src_addr->unit : PSMX2_DEFAULT_UNIT;
	opts.port = src_addr ? src_addr->port : PSMX2_DEFAULT_PORT;
	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"ep_open_opts: unit=%d port=%u\n", opts.unit, opts.port);

	if (opts.unit < 0 && sep_ctxt_idx >= 0) {
		should_retry = 1;
		opts.unit = sep_ctxt_idx % psmx2_env.num_devunits;
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"sep %d: ep_open_opts: unit=%d\n", sep_ctxt_idx, opts.unit);
	}

	err = psm2_ep_open(domain->fabric->uuid, &opts,
			   &trx_ctxt->psm2_ep, &trx_ctxt->psm2_epid);
	if (err != PSM2_OK) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"psm2_ep_open returns %d, errno=%d\n", err, errno);
		if (!should_retry) {
			err = psmx2_errno(err);
			goto err_out_destroy_pool;
		}

		/* When round-robin fails, retry w/o explicit assignment */
		opts.unit = -1;
		err = psm2_ep_open(domain->fabric->uuid, &opts,
				   &trx_ctxt->psm2_ep, &trx_ctxt->psm2_epid);
		if (err != PSM2_OK) {
			FI_WARN(&psmx2_prov, FI_LOG_CORE,
				"psm2_ep_open returns %d, errno=%d\n", err, errno);
			err = psmx2_errno(err);
			goto err_out_destroy_pool;
		}
	}

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"epid: 0x%016lx\n", trx_ctxt->psm2_epid);

	err = psm2_mq_init(trx_ctxt->psm2_ep, PSM2_MQ_ORDERMASK_ALL,
			   NULL, 0, &trx_ctxt->psm2_mq);
	if (err != PSM2_OK) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"psm2_mq_init returns %d, errno=%d\n", err, errno);
		err = psmx2_errno(err);
		goto err_out_close_ep;
	}

	fastlock_init(&trx_ctxt->peer_lock);
	fastlock_init(&trx_ctxt->poll_lock);
	fastlock_init(&trx_ctxt->am_req_pool_lock);
	fastlock_init(&trx_ctxt->rma_queue.lock);
	fastlock_init(&trx_ctxt->trigger_queue.lock);
	dlist_init(&trx_ctxt->peer_list);
	slist_init(&trx_ctxt->rma_queue.list);
	slist_init(&trx_ctxt->trigger_queue.list);
	trx_ctxt->id = psmx2_trx_ctxt_cnt++;
	trx_ctxt->domain = domain;

	return trx_ctxt;

err_out_close_ep:
	if (psm2_ep_close(trx_ctxt->psm2_ep, PSM2_EP_CLOSE_GRACEFUL,
			  (int64_t) psmx2_env.timeout * 1000000000LL) != PSM2_OK)
		psm2_ep_close(trx_ctxt->psm2_ep, PSM2_EP_CLOSE_FORCE, 0);

err_out_destroy_pool:
	util_buf_pool_destroy(trx_ctxt->am_req_pool);

err_out:
	free(trx_ctxt);
	return NULL;
}

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
		psmx2_progress_all(domain);
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
		} else {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"progress thread exited with code %ld (%s)\n",
				(uintptr_t)exit_code,
				(exit_code == PTHREAD_CANCELED) ?
					"PTHREAD_CANCELED" : "?");
		}
	}
}

static int psmx2_domain_close(fid_t fid)
{
	struct psmx2_fid_domain *domain;

	domain = container_of(fid, struct psmx2_fid_domain,
			      util_domain.domain_fid.fid);

	FI_INFO(&psmx2_prov, FI_LOG_DOMAIN, "refcnt=%d\n",
		ofi_atomic_get32(&domain->util_domain.ref));

	psmx2_domain_release(domain);

	if (ofi_domain_close(&domain->util_domain))
		return 0;

	if (domain->progress_thread_enabled)
		psmx2_domain_stop_progress(domain);

	fastlock_destroy(&domain->sep_lock);
	fastlock_destroy(&domain->mr_lock);
	rbtDelete(domain->mr_map);

	domain->fabric->active_domain = NULL;
	free(domain);

	psmx2_atomic_global_fini();
	psmx2_am_global_fini();
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
	.scalable_ep = psmx2_sep_open,
	.cntr_open = psmx2_cntr_open,
	.poll_open = fi_poll_create,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = psmx2_query_atomic,
};

static int psmx2_key_compare(void *key1, void *key2)
{
	return (key1 < key2) ?  -1 : (key1 > key2);
}

static int psmx2_domain_init(struct psmx2_fid_domain *domain,
			     struct psmx2_ep_name *src_addr)
{
	int err;
	void *dlsym(void *handle, const char *symbol);

	psmx2_disconnect_psm2_ep = dlsym(NULL, "psm2_ep_disconnect");
	if (!psmx2_disconnect_psm2_ep)
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"function psm2_ep_disconnect() is missing\n");

	psmx2_am_global_init();
	psmx2_atomic_global_init();

	err = fastlock_init(&domain->mr_lock);
	if (err) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"fastlock_init(mr_lock) returns %d\n", err);
		goto err_out;
	}

	domain->mr_map = rbtNew(&psmx2_key_compare);
	if (!domain->mr_map) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"rbtNew failed\n");
		goto err_out_destroy_mr_lock;
	}

	domain->mr_reserved_key = 1;
	domain->max_atomic_size = INT_MAX;

	ofi_atomic_initialize32(&domain->sep_cnt, 0);
	fastlock_init(&domain->sep_lock);
	dlist_init(&domain->sep_list);
	dlist_init(&domain->trx_ctxt_list);
	fastlock_init(&domain->trx_ctxt_lock);

	domain->fabric->active_domain = domain;

	if (domain->progress_thread_enabled)
		psmx2_domain_start_progress(domain);

	return 0;

err_out_destroy_mr_lock:
	fastlock_destroy(&domain->mr_lock);

err_out:
	return err;
}

int psmx2_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		      struct fid_domain **domain, void *context)
{
	struct psmx2_fid_fabric *fabric_priv;
	struct psmx2_fid_domain *domain_priv;
	struct psmx2_ep_name *src_addr = info->src_addr;
	int mr_mode = (info->domain_attr->mr_mode & FI_MR_BASIC) ? FI_MR_BASIC : 0;
	int err;

	FI_INFO(&psmx2_prov, FI_LOG_DOMAIN, "\n");

	if (!psmx2_env.sep)
		psmx2_domain_ops.scalable_ep = fi_no_scalable_ep;

	fabric_priv = container_of(fabric, struct psmx2_fid_fabric,
				   util_fabric.fabric_fid);

	if (fabric_priv->active_domain) {
		if (mr_mode != fabric_priv->active_domain->mr_mode) {
			FI_INFO(&psmx2_prov, FI_LOG_DOMAIN,
				"mr_mode mismatch: expecting %s\n",
				mr_mode ? "FI_MR_SCALABLE" : "FI_MR_BASIC");
			return -FI_EINVAL;
		}

		psmx2_domain_acquire(fabric_priv->active_domain);
		*domain = &fabric_priv->active_domain->util_domain.domain_fid;
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

	err = ofi_domain_init(fabric, info, &domain_priv->util_domain, context);
	if (err)
		goto err_out_free_domain;

	/* fclass & context are set in ofi_domain_init */
	domain_priv->util_domain.domain_fid.fid.ops = &psmx2_fi_ops;
	domain_priv->util_domain.domain_fid.ops = &psmx2_domain_ops;
	domain_priv->util_domain.domain_fid.mr = &psmx2_mr_ops;
	domain_priv->mr_mode = mr_mode;
	domain_priv->mode = info->mode;
	domain_priv->caps = PSMX2_CAPS | PSMX2_DOM_CAPS;
	domain_priv->fabric = fabric_priv;
	domain_priv->progress_thread_enabled =
		(info->domain_attr->data_progress == FI_PROGRESS_AUTO);
	domain_priv->addr_format = info->addr_format;

	if (info->addr_format == FI_ADDR_STR)
		src_addr = psmx2_string_to_ep_name(info->src_addr);

	err = psmx2_domain_init(domain_priv, src_addr);
	if (info->addr_format == FI_ADDR_STR)
		free(src_addr);
	if (err)
		goto err_out_close_domain;

	/* take the reference to count for multiple domain open calls */
	psmx2_domain_acquire(fabric_priv->active_domain);

	*domain = &domain_priv->util_domain.domain_fid;
	return 0;

err_out_close_domain:
	ofi_domain_close(&domain_priv->util_domain);

err_out_free_domain:
	free(domain_priv);

err_out:
	return err;
}

static int psmx2_domain_check_features(struct psmx2_fid_domain *domain,
				       uint64_t ep_cap)
{
	if ((domain->caps & ep_cap & ~PSMX2_SUB_CAPS) !=
	    (ep_cap & ~PSMX2_SUB_CAPS)) {
		uint64_t mask = ~PSMX2_SUB_CAPS;
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"caps mismatch: domain->caps=%s,\n ep->caps=%s,\n mask=%s\n",
			fi_tostr(&domain->caps, FI_TYPE_CAPS),
			fi_tostr(&ep_cap, FI_TYPE_CAPS),
			fi_tostr(&mask, FI_TYPE_CAPS));
		return -FI_EOPNOTSUPP;
	}

	return 0;
}

int psmx2_domain_enable_ep(struct psmx2_fid_domain *domain,
			   struct psmx2_fid_ep *ep)
{
	int err;

	err = psmx2_domain_check_features(domain, ep->caps);
	if (err)
		return err;

	if ((ep->caps & FI_RMA) || (ep->caps & FI_ATOMICS))
		return psmx2_am_init(ep->trx_ctxt);

	return 0;
}

