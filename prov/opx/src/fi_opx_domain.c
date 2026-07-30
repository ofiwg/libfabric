/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2026 by Cornelis Networks.
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <unistd.h>
#include <limits.h>
#include <fcntl.h>
#include <ctype.h>

#include "rdma/fabric.h"

#include "rdma/opx/fi_opx_domain.h"
#include "rdma/opx/fi_opx_internal.h"
#include "rdma/opx/fi_opx_hfi1.h"
#include "opx_shm_cache.h"

#include <ofi_enosys.h>

#include "rdma/opx/fi_opx.h"
#include "rdma/opx/opx_hfisvc.h"
#include "rdma/opx/opx_hfisvc_poll.h"

#define OPX_DOMAIN_HFISVC_NOT_INITIALIZED (0x7FFFFFFFFFFFFFFEll)

#if HAVE_HFISVC
/* OPX does not ibverbs directly, dlopen/dlsym only */
#include <dlfcn.h>
#endif

static int fi_opx_close_domain(fid_t fid)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_DOMAIN, "close domain\n");

	int		      ret;
	struct fi_opx_domain *opx_domain = container_of(fid, struct fi_opx_domain, domain_fid);

	ret = fi_opx_fid_check(fid, FI_CLASS_DOMAIN, "domain");
	if (ret) {
		return ret;
	}

	/* Release the cached drvchk shm segment on orderly exit (winner only; crash path self-heals). */
	opx_shm_cache_unlink(opx_domain->drvchk_shm_name);

	ret = fi_opx_finalize_mr_ops(&opx_domain->domain_fid);
	if (ret) {
		return ret;
	}

	opx_close_tid_domain(opx_domain->tid_domain, OPX_TID_NO_LOCK_ON_CLEANUP);
	opx_domain->tid_domain = NULL;
#ifdef OPX_HMEM
	opx_hmem_close_domain(opx_domain->hmem_domain, OPX_HMEM_NO_LOCK_ON_CLEANUP);
	opx_domain->hmem_domain = NULL;
#endif

#if HAVE_HFISVC
	if (opx_domain->use_hfisvc) {
		/* Drain outstanding access keys for all contexts */
		for (int i = 0; i < opx_domain->hfisvc.num_ctxs; i++) {
			while (opx_hfisvc_keyset_outstanding(opx_domain->hfisvc.ctxs[i].access_key_set)) {
				opx_domain_hfisvc_poll(opx_domain);
			}
		}
	}
#endif

	if (!slist_empty(&opx_domain->deferred_work_queue)) {
		struct opx_domain_deferred_work *work_item =
			(struct opx_domain_deferred_work *) slist_remove_head(&opx_domain->deferred_work_queue);

		while (work_item) {
			if (work_item->opx_mr && work_item->work_fn == opx_hfisvc_mr_deferred_close) {
				free(work_item->opx_mr);
			}
			OPX_BUF_FREE(work_item);
			work_item =
				(struct opx_domain_deferred_work *) slist_remove_head(&opx_domain->deferred_work_queue);
		}
	}

	ofi_bufpool_destroy(opx_domain->deferred_work_pool);
#if HAVE_HFISVC
	if (opx_domain->use_hfisvc) {
		ret = fi_opx_ref_finalize(&opx_domain->hfisvc.ref_cnt, "hfisvc");
		if (ret) {
			return ret;
		}

		/* Close domain-level MR queues */
		ret = (*opx_domain->hfisvc.completion_queue_close)(&opx_domain->hfisvc.mr_completion_queue);
		if (ret) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"[HFISVC] Failed closing domain MR completion queue, ret=%d\n", ret);
		}

		ret = (*opx_domain->hfisvc.command_queue_close)(&opx_domain->hfisvc.mr_command_queue);
		if (ret) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"[HFISVC] Failed closing domain MR command queue, ret=%d\n", ret);
		}

		/* Close all hfisvc contexts */
		for (int i = 0; i < opx_domain->hfisvc.num_ctxs; i++) {
			if (opx_domain->hfisvc.ctxs[i].ctx == NULL) {
				continue;
			}

			FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] Closing context %d\n", i);

			opx_hfisvc_keyset_free(opx_domain->hfisvc.ctxs[i].access_key_set);
			opx_domain->hfisvc.ctxs[i].access_key_set = 0;
			int finalize_ret = (*opx_domain->hfisvc.finalize)(opx_domain->hfisvc.ctxs[i].ctx);
			if (finalize_ret) {
				FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
					"[HFISVC] Failed finalizing context %d, ret=%d\n", i, finalize_ret);
			}
			opx_hfi1_rdma_context_close(opx_domain->hfisvc.ctxs[i].ctx);
			opx_domain->hfisvc.ctxs[i].ctx = NULL;
		}

		if (opx_domain->hfisvc.libhfi1verbs != NULL) {
			dlclose(opx_domain->hfisvc.libhfi1verbs);
			opx_domain->hfisvc.libhfi1verbs = NULL;
		}
	}
#endif

	/* Close rdma-core lib, the endpoint already closed contexts */
	opx_hfi1_rdma_lib_close();

	ret = fi_opx_ref_finalize(&opx_domain->ref_cnt, "domain");
	if (ret) {
		return ret;
	}

	ofi_atomic_dec32(&opx_domain->fabric->util_fabric.ref);

	FI_OPX_DEBUG_COUNTERS_PRINT(opx_domain->debug_counters);

	free(opx_domain);

	// opx_domain (the object passed in as fid) is now unusable

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_DOMAIN, "domain closed\n");
	return 0;
}

static struct fi_ops fi_opx_fi_ops = {.size	= sizeof(struct fi_ops),
				      .close	= fi_opx_close_domain,
				      .bind	= fi_no_bind,
				      .control	= fi_no_control,
				      .ops_open = fi_no_ops_open};

static struct fi_ops_domain fi_opx_domain_ops = {.size	      = sizeof(struct fi_ops_domain),
						 .av_open     = fi_opx_av_open,
						 .cq_open     = fi_opx_cq_open,
						 .endpoint    = fi_opx_endpoint,
						 .scalable_ep = fi_no_scalable_ep,
						 .cntr_open   = fi_opx_cntr_open,
						 .poll_open   = fi_no_poll_open,
						 .stx_ctx     = fi_no_stx_context,
						 .srx_ctx     = fi_no_srx_context};

static inline void opx_util_domain_cleanup(struct fi_opx_domain *opx_domain)
{
	if (opx_domain->tid_domain) {
		opx_close_tid_domain(opx_domain->tid_domain, OPX_TID_NO_LOCK_ON_CLEANUP);
		opx_domain->tid_domain = NULL;
	}
#ifdef OPX_HMEM
	if (opx_domain->hmem_domain) {
		opx_hmem_close_domain(opx_domain->hmem_domain, OPX_HMEM_NO_LOCK_ON_CLEANUP);
		opx_domain->hmem_domain = NULL;
	}
#endif
}

uint32_t opx_domain_get_ctx_cnt(int hfi, int port)
{
	uint32_t ctx_cnt = opx_hfi_get_port_num_contexts(hfi, port);

	return ctx_cnt ? ctx_cnt : FI_OPX_DEFAULT_DOMAIN_CTX_CNT;
}

uint32_t opx_domain_get_total_ctx_cnt(int port)
{
	const int num_hfis = opx_hfi_get_num_units();
	uint32_t  total	   = 0;

	/* Sum the raw per-unit context count directly (not via
	 * opx_domain_get_ctx_cnt()) so a unit reporting 0 contexts
	 * contributes 0, not a 160-context fallback, to the total. The
	 * fallback below is only applied once, to the aggregate. */
	for (int hfi = 0; hfi < num_hfis; ++hfi) {
		if (opx_hfi_get_unit_active(hfi)) {
			total += opx_hfi_get_port_num_contexts(hfi, port);
		}
	}

	return total ? total : FI_OPX_DEFAULT_DOMAIN_CTX_CNT;
}

static bool opx_parse_env_int(const char *name, int32_t *out)
{
	const char *e = getenv(name);
	if (!e || !*e) {
		return false;
	}
	char *ep;
	long  val = strtol(e, &ep, 10);
	if (ep == e) {
		return false;
	}
	*out = (int32_t) val;
	return true;
}

void opx_query_local_rank_info(int32_t *local_rank_count, int32_t *local_rank)
{
	static const struct {
		const char *count_var;
		const char *id_var;
	} launchers[] = {
		{"MPI_LOCALNRANKS", "MPI_LOCALRANKID"}, {"OMPI_COMM_WORLD_LOCAL_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK"},
		{"LOCAL_WORLD_SIZE", "LOCAL_RANK"},	{"SLURM_NTASKS_PER_NODE", "SLURM_LOCALID"},
		{"CCL_LOCAL_SIZE", "CCL_LOCAL_RANK"},
	};

	if (local_rank_count) {
		*local_rank_count = -1;
	}
	if (local_rank) {
		*local_rank = -1;
	}

	for (size_t i = 0; i < sizeof(launchers) / sizeof(launchers[0]); i++) {
		int32_t count;
		if (opx_parse_env_int(launchers[i].count_var, &count) && count > 0) {
			if (local_rank_count) {
				*local_rank_count = count;
			}
			if (local_rank) {
				opx_parse_env_int(launchers[i].id_var, local_rank);
			}
			return;
		}
	}
}

int fi_opx_alloc_default_domain_attr(struct fi_domain_attr **domain_attr)
{
	struct fi_domain_attr *attr;

	attr = calloc(1, sizeof(*attr));
	if (!attr) {
		goto err;
	}

	int32_t local_rank_count;
	opx_query_local_rank_info(&local_rank_count, NULL);
	const uint32_t ppn	   = local_rank_count > 0 ? (uint32_t) local_rank_count : 1;
	const unsigned ctx_cnt	   = opx_domain_get_ctx_cnt(0, OPX_PORT_NUM_ANY);
	const int      num_hfis	   = opx_hfi_get_num_units();
	const uint32_t ppn_per_hfi = (num_hfis > 1) ? MAX(1, ppn / (uint32_t) num_hfis) : ppn;
	const unsigned tx_ctx_cnt  = ctx_cnt / ppn_per_hfi;
	const unsigned rx_ctx_cnt  = ctx_cnt / ppn_per_hfi;

	attr->domain = NULL;
	attr->name   = strdup(FI_OPX_DOMAIN_NAME);

	attr->threading	       = OPX_THREAD;
	attr->control_progress = FI_PROGRESS_MANUAL;
	attr->data_progress    = FI_PROGRESS_MANUAL;
	attr->resource_mgmt    = FI_RM_DISABLED;
	attr->av_type	       = OPX_AV;
	attr->mr_mode	       = FI_OPX_BASE_MR_MODE;
	attr->caps	       = FI_OPX_DOMAIN_CAPS;
	attr->mr_key_size      = sizeof(uint64_t);
	attr->cq_data_size     = FI_OPX_REMOTE_CQ_DATA_SIZE;
	attr->cq_cnt	       = (size_t) -1;
	attr->ep_cnt	       = ctx_cnt / ppn_per_hfi;
	attr->tx_ctx_cnt       = tx_ctx_cnt;
	attr->rx_ctx_cnt       = rx_ctx_cnt;

	attr->max_ep_tx_ctx = 1;
	attr->max_ep_rx_ctx = 1;

	attr->max_ep_stx_ctx = 0;
	attr->max_ep_srx_ctx = 0;
	attr->mr_iov_limit   = 1;

	*domain_attr = attr;

	return 0;
err:
	*domain_attr = NULL;
	errno	     = FI_ENOMEM;
	return -1;
}

int fi_opx_choose_domain(uint64_t caps, struct fi_domain_attr *domain_attr, struct fi_domain_attr *hints,
			 enum fi_progress progress)
{
	if (!domain_attr) {
		FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN, "missing domain attribute structure\n");
		goto err;
	}

	*domain_attr		   = *fi_opx_global.default_domain_attr;
	domain_attr->name	   = NULL;
	domain_attr->data_progress = progress;

#ifdef OPX_ENABLED
	domain_attr->mr_mode = OPX_MR;
#endif

#ifdef OPX_HMEM
	domain_attr->mr_mode |= FI_MR_HMEM;
#endif

	if (hints) {
		{
			const int opx_modern_mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
			int supported_mr_mode = FI_OPX_BASE_MR_MODE | opx_modern_mr_mode | FI_MR_ENDPOINT | FI_MR_RAW;
#ifdef OPX_HMEM
			supported_mr_mode |= FI_MR_HMEM;
#endif

			if (hints->mr_mode == ~(OFI_MR_BASIC | OFI_MR_SCALABLE)) {
				domain_attr->mr_mode = FI_OPX_BASE_MR_MODE;
#ifdef OPX_HMEM
				domain_attr->mr_mode |= FI_MR_HMEM;
#endif
			} else if (hints->mr_mode == 0) {
				domain_attr->mr_mode = 0;
			} else if (hints->mr_mode & ~supported_mr_mode) {
				FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN,
					"Unsupported mr_mode hint 0x%x, supported 0x%x\n", hints->mr_mode,
					supported_mr_mode);
				goto unavailable;
			} else if (hints->mr_mode & opx_modern_mr_mode) {
				domain_attr->mr_mode = hints->mr_mode & opx_modern_mr_mode;
#ifdef OPX_HMEM
				domain_attr->mr_mode |= hints->mr_mode & FI_MR_HMEM;
#endif
			} else if ((hints->mr_mode & FI_OPX_BASE_MR_MODE) == FI_OPX_BASE_MR_MODE) {
				domain_attr->mr_mode = FI_OPX_BASE_MR_MODE;
#ifdef OPX_HMEM
				domain_attr->mr_mode |= FI_MR_HMEM;
#endif
			} else if (hints->mr_mode & FI_MR_ENDPOINT) {
				domain_attr->mr_mode = 0;
#ifdef OPX_HMEM
				domain_attr->mr_mode |= hints->mr_mode & FI_MR_HMEM;
#endif
			} else {
				FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN,
					"Unsupported mr_mode hint 0x%x, required 0x%x\n", hints->mr_mode,
					FI_OPX_BASE_MR_MODE);
				goto unavailable;
			}
		}

		if (hints->domain) {
			struct fi_opx_domain *opx_domain =
				container_of(hints->domain, struct fi_opx_domain, domain_fid);

			domain_attr->threading	   = opx_domain->threading;
			domain_attr->resource_mgmt = opx_domain->resource_mgmt;
			domain_attr->tx_ctx_cnt	   = fi_opx_domain_get_tx_max(hints->domain);
			domain_attr->rx_ctx_cnt	   = fi_opx_domain_get_rx_max(hints->domain);
			domain_attr->max_ep_tx_ctx = fi_opx_domain_get_tx_max(hints->domain);
			domain_attr->max_ep_rx_ctx = fi_opx_domain_get_rx_max(hints->domain);

		} else {
			if (hints->threading) {
				domain_attr->threading = hints->threading;
			}
			if (hints->resource_mgmt) {
				domain_attr->resource_mgmt = hints->resource_mgmt;
			}
			if (hints->av_type) {
				domain_attr->av_type = hints->av_type;
			}
			if (hints->mr_key_size) {
				domain_attr->mr_key_size = hints->mr_key_size;
			}
			if (hints->cq_data_size) {
				domain_attr->cq_data_size = hints->cq_data_size;
			}
			if (hints->cq_cnt) {
				domain_attr->cq_cnt = hints->cq_cnt;
			}
			if (hints->ep_cnt) {
				domain_attr->ep_cnt = hints->ep_cnt;
			}
			if (hints->tx_ctx_cnt) {
				domain_attr->tx_ctx_cnt = hints->tx_ctx_cnt;
			}
			if (hints->rx_ctx_cnt) {
				domain_attr->rx_ctx_cnt = hints->rx_ctx_cnt;
			}
			if (hints->max_ep_tx_ctx) {
				domain_attr->max_ep_tx_ctx = hints->max_ep_tx_ctx;
			}
			if (hints->max_ep_rx_ctx) {
				domain_attr->max_ep_rx_ctx = hints->max_ep_rx_ctx;
			}
			if (hints->mr_iov_limit) {
				domain_attr->mr_iov_limit = hints->mr_iov_limit;
			}
		}
	}

	domain_attr->name = strdup(FI_OPX_DOMAIN_NAME);

	if (!domain_attr->name) {
		FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN, "no memory\n");
		errno = FI_ENOMEM;
		return -errno;
	}

	domain_attr->cq_data_size = FI_OPX_REMOTE_CQ_DATA_SIZE;

	return 0;
unavailable:
	errno = FI_ENODATA;
	return -errno;
err:
	errno = FI_EINVAL;
	return -errno;
}

int fi_opx_check_domain_attr(struct fi_domain_attr *attr)
{
	if (OFI_UNLIKELY(fi_opx_threading_unknown(attr->threading))) {
		FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN, "incorrect threading level\n");
		goto err;
	}

	if (attr->mr_key_size) {
		if (attr->mr_key_size > sizeof(uint64_t)) {
			FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN, "memory key size too large\n");
			goto err;
		}
	}
	if (attr->cq_data_size) {
		if (attr->cq_data_size > FI_OPX_REMOTE_CQ_DATA_SIZE) {
			FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN, "max cq data supported is %zu\n",
			       FI_OPX_REMOTE_CQ_DATA_SIZE);
			goto err;
		}
	}

	return 0;

err:
	errno = FI_EINVAL;
	return -errno;
}

int fi_opx_validate_affinity_str(char *str)
{
	int  cols      = 0;
	bool recentCol = true;
	int  iter;

	for (iter = 0; iter < strlen(str); iter++) {
		if (!isdigit(str[iter]) && str[iter] != ':') {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"Invalid program affinity. Progress affinity must be a digit or colon.\n");
			return -1;
		}

		if (str[iter] == ':') {
			if (recentCol) {
				FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
					"Progress Affinity improperly formatted. Must be a : separated triplet.\n");
				return -1;
			} else {
				cols += 1;
				recentCol = true;
			}
		} else {
			recentCol = false;
		}
	}

	if (cols != 2) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"Progress Affinity improperly formatted. Must be a : separated triplet.\n");
		return -1;
	}
	return 0;
}

int fi_opx_domain(struct fid_fabric *fabric, struct fi_info *info, struct fid_domain **dom, void *context)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_DOMAIN, "open domain\n");

	int		      ret	      = 0;
	int		      get_param_check = 0;
	struct fi_opx_domain *opx_domain      = NULL;
	struct fi_opx_fabric *opx_fabric      = container_of(fabric, struct fi_opx_fabric, util_fabric.fabric_fid);

	if (!info) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "no info supplied\n");
		errno = FI_EINVAL;
		return -errno;
	}

	ret = fi_opx_fid_check(&fabric->fid, FI_CLASS_FABRIC, "fabric");
	if (ret) {
		return ret;
	}

	opx_domain = calloc(1, sizeof(struct fi_opx_domain));
	if (!opx_domain) {
		errno = FI_ENOMEM;
		goto err;
	}
	FI_OPX_DEBUG_COUNTERS_INIT(opx_domain->debug_counters);
	opx_domain->tid_domain = NULL;
#ifdef OPX_HMEM
	opx_domain->hmem_domain = NULL;
#endif

	if (fi_opx_global.default_domain_attr == NULL) {
		if (fi_opx_alloc_default_domain_attr(&fi_opx_global.default_domain_attr)) {
			FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN,
			       "alloc function could not allocate block of memory\n");
			errno = FI_ENOMEM;
			goto err;
		}
	}

	struct opx_tid_domain *opx_tid_domain;
	struct opx_tid_fabric *opx_tid_fabric = opx_fabric->tid_fabric;

	if (opx_open_tid_domain(opx_tid_fabric, info, &opx_tid_domain)) {
		errno = FI_ENOMEM;
		goto err;
	}
	opx_domain->tid_domain = opx_tid_domain;

#ifdef OPX_HMEM
	struct opx_hmem_domain *opx_hmem_domain;
	struct opx_hmem_fabric *opx_hmem_fabric = opx_fabric->hmem_fabric;

	if (opx_hmem_open_domain(opx_hmem_fabric, info, &opx_hmem_domain)) {
		errno = FI_ENOMEM;
		goto err;
	}
	opx_domain->hmem_domain		    = opx_hmem_domain;
	opx_domain->hmem_domain->opx_domain = opx_domain;

	size_t env_var_threshold;
	get_param_check = fi_param_get_size_t(fi_opx_global.prov, "dev_reg_send_threshold", &env_var_threshold);
	if (get_param_check == FI_SUCCESS) {
		if (env_var_threshold <= OPX_HMEM_DEV_REG_THRESHOLD_MAX) {
			opx_domain->hmem_domain->devreg_copy_from_threshold = env_var_threshold;
		} else {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"FI_OPX_DEV_REG_SEND_THRESHOLD must be an integer >= %u and <= %u. Using default value (%u) instead of %zu\n",
				OPX_HMEM_DEV_REG_THRESHOLD_MIN, OPX_HMEM_DEV_REG_THRESHOLD_MAX,
				OPX_HMEM_DEV_REG_SEND_THRESHOLD_DEFAULT, env_var_threshold);
		}
	}

	get_param_check = fi_param_get_size_t(fi_opx_global.prov, "dev_reg_recv_threshold", &env_var_threshold);
	if (get_param_check == FI_SUCCESS) {
		if (env_var_threshold <= OPX_HMEM_DEV_REG_THRESHOLD_MAX) {
			opx_domain->hmem_domain->devreg_copy_to_threshold = env_var_threshold;
		} else {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"FI_OPX_DEV_REG_RECV_THRESHOLD must be an integer >= %u and <= %u. Using default value (%u) instead of %zu\n",
				OPX_HMEM_DEV_REG_THRESHOLD_MIN, OPX_HMEM_DEV_REG_THRESHOLD_MAX,
				OPX_HMEM_DEV_REG_RECV_THRESHOLD_DEFAULT, env_var_threshold);
		}
	}

	opx_domain->hmem_domain->dmabuf_supported = 0;

#if HAVE_HFISVC_DMABUF
#if HAVE_CUDA
	if (cuda_is_dmabuf_supported()) {
		opx_domain->hmem_domain->dmabuf_supported = 1;
	}
#elif HAVE_ROCR
	if (rocr_hmem_get_dmabuf_fd(NULL, 0, NULL, 0) != -FI_EOPNOTSUPP) {
		opx_domain->hmem_domain->dmabuf_supported = 1;
	}
#endif
#endif
#endif

	/* fill in default domain attributes */
	opx_domain->threading	  = fi_opx_global.default_domain_attr->threading;
	opx_domain->resource_mgmt = fi_opx_global.default_domain_attr->resource_mgmt;
	opx_domain->data_progress = fi_opx_global.default_domain_attr->data_progress;

	if (info->domain_attr) {
		if (info->domain_attr->domain) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "domain cannot be supplied\n");
			goto err;
		}
		ret = fi_opx_check_domain_attr(info->domain_attr);
		if (ret) {
			goto err;
		}
		opx_domain->threading	  = info->domain_attr->threading;
		opx_domain->resource_mgmt = info->domain_attr->resource_mgmt;
		if (fi_opx_global.progress == FI_PROGRESS_UNSPEC) {
			opx_domain->data_progress = info->domain_attr->data_progress;
		}
	}

	opx_domain->fabric = opx_fabric;

	fi_opx_ref_init(&opx_domain->ref_cnt, 0, "domain");

	opx_domain->domain_fid.fid.fclass  = FI_CLASS_DOMAIN;
	opx_domain->domain_fid.fid.context = context;
	opx_domain->domain_fid.fid.ops	   = &fi_opx_fi_ops;
	opx_domain->domain_fid.ops	   = &fi_opx_domain_ops;

	opx_domain->progress_affinity_str = NULL;
	get_param_check = fi_param_get_str(fi_opx_global.prov, "prog_affinity", &opx_domain->progress_affinity_str);
	if (get_param_check == FI_SUCCESS) {
		if (fi_opx_validate_affinity_str(opx_domain->progress_affinity_str) != 0) {
			opx_domain->progress_affinity_str = NULL;
			errno				  = FI_EINVAL;
			goto err;
		}
	}

	// Max UUID consists of 32 hex digits.
	char *env_var_uuid = OPX_DEFAULT_JOB_KEY_STR;
	get_param_check	   = fi_param_get_str(fi_opx_global.prov, "uuid", &env_var_uuid);
	char *impi_uuid	   = getenv("I_MPI_HYDRA_UUID");
	char *slurm_job_id = getenv("SLURM_JOB_ID");

	if (get_param_check == FI_SUCCESS) {
		FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN, "Detected user specified FI_OPX_UUID\n");
	} else if (slurm_job_id) {
		env_var_uuid = slurm_job_id;
		FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN, "Found SLURM_JOB_ID.  Using it for FI_OPX_UUID\n");
	} else if (impi_uuid) {
		env_var_uuid = impi_uuid;
		FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN, "Found I_MPI_HYDRA_UUID.  Using it for FI_OPX_UUID\n");
	} else {
		env_var_uuid = OPX_DEFAULT_JOB_KEY_STR;
	}

	if (strlen(env_var_uuid) >= OPX_JOB_KEY_STR_SIZE) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"UUID too long. UUID must consist of 1-32 hexadecimal digits.  Using default OPX uuid instead\n");
		env_var_uuid = OPX_DEFAULT_JOB_KEY_STR;
	}

	int i;
	for (i = 0; i < OPX_JOB_KEY_STR_SIZE && env_var_uuid[i] != 0; i++) {
		if (!isxdigit(env_var_uuid[i])) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"Invalid UUID. UUID must consist solely of hexadecimal digits.  Using default OPX uuid instead\n");
			env_var_uuid = OPX_DEFAULT_JOB_KEY_STR;
		}
	}

	// Copy the job key and guarantee null termination.
	strncpy(opx_domain->unique_job_key_str, env_var_uuid, OPX_JOB_KEY_STR_SIZE - 1);
	opx_domain->unique_job_key_str[OPX_JOB_KEY_STR_SIZE - 1] = '\0';

	int elements_read = sscanf(
		opx_domain->unique_job_key_str,
		"%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx",
		&opx_domain->unique_job_key[0], &opx_domain->unique_job_key[1], &opx_domain->unique_job_key[2],
		&opx_domain->unique_job_key[3], &opx_domain->unique_job_key[4], &opx_domain->unique_job_key[5],
		&opx_domain->unique_job_key[6], &opx_domain->unique_job_key[7], &opx_domain->unique_job_key[8],
		&opx_domain->unique_job_key[9], &opx_domain->unique_job_key[10], &opx_domain->unique_job_key[11],
		&opx_domain->unique_job_key[12], &opx_domain->unique_job_key[13], &opx_domain->unique_job_key[14],
		&opx_domain->unique_job_key[15]);
	if (elements_read == EOF) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"Error: sscanf encountered an input failure (EOF), unable to parse the unique job key string.\n");
		errno = FI_EINVAL;
		goto err;
	}

	FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN, "Domain unique job key set to %s\n", opx_domain->unique_job_key_str);
	// TODO: Print out a summary of all domain settings wtih FI_INFO

	opx_domain->use_hfisvc = 0;
#if HAVE_HFISVC
	fi_opx_ref_init(&opx_domain->hfisvc.ref_cnt, OPX_DOMAIN_HFISVC_NOT_INITIALIZED, "hfisvc");
#endif

	opx_domain->rx_count = 0;
	opx_domain->tx_count = 0;
	opx_domain->ep_count = 0;

	memset(&opx_domain->hfi_local_info, 0, sizeof(opx_domain->hfi_local_info));
	opx_domain->hfi_local_info.local_lids_size  = 0;
	opx_domain->hfi_local_info.sw_type	    = OPX_HFI1_UNDEF;
	opx_domain->hfi_local_info.hw_type	    = OPX_HFI1_UNDEF;
	opx_domain->hfi_local_info.sim_rctxt_fd	    = -1;
	opx_domain->hfi_local_info.sim_sctxt_fd	    = -1;
	opx_domain->hfi_local_info.lid[0]	    = (opx_lid_t) 0;
	opx_domain->hfi_local_info.lid[1]	    = (opx_lid_t) 0;
	opx_domain->hfi_local_info.hfi_unit[0]	    = (uint8_t) -1U;
	opx_domain->hfi_local_info.hfi_unit[1]	    = (uint8_t) -1U;
	opx_domain->hfi_local_info.sriov	    = false;
	opx_domain->hfi_local_info.port_loopback    = false;
	opx_domain->hfi_local_info.hairpin_loopback = false;
	opx_domain->hfi_local_info.multi_hfi	    = false;
	opx_domain->hfi_local_info.lid_path_mask    = 0;
	opx_domain->hfi_local_info.lid_mask	    = 0xFFFFFFFFu;
	opx_domain->hfi_local_info.neighbor_type    = OPX_NEIGHBOR_UNKNOWN;

	ret = fi_opx_init_mr_ops(&opx_domain->domain_fid, info);
	if (ret) {
		goto err;
	}

	slist_init(&opx_domain->deferred_work_queue);
	ofi_bufpool_create(&opx_domain->deferred_work_pool, sizeof(struct opx_domain_deferred_work), 32, UINT_MAX, 2048,
			   OFI_BUFPOOL_NO_ZERO);

	ofi_atomic_inc32(&opx_fabric->util_fabric.ref);

	*dom = &opx_domain->domain_fid;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_DOMAIN, "domain opened\n");
	return 0;

err:
	if (opx_domain) {
		fi_opx_finalize_mr_ops(&opx_domain->domain_fid);
		opx_util_domain_cleanup(opx_domain);
		free(opx_domain);
		opx_domain = NULL;
	}

	if (fi_opx_global.default_domain_attr != NULL) {
		if (fi_opx_global.default_domain_attr->name != NULL) {
			free(fi_opx_global.default_domain_attr->name);
			fi_opx_global.default_domain_attr->name = NULL;
		}
		free(fi_opx_global.default_domain_attr);
		fi_opx_global.default_domain_attr = NULL;
	}
	return -errno;
}

#if HAVE_HFISVC

extern struct opx_rdma_ops_struct opx_rdma_ops;

int opx_domain_hfisvc_init_ctx(struct fi_opx_domain *domain, int ctx_index)
{
	assert(ctx_index >= 0 && ctx_index < OPX_MAX_TX_CONTEXTS);
	assert(domain->hfisvc.ctxs[ctx_index].ctx != NULL);

	int ret;

	FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] Initializing context %d\n", ctx_index);

	ret = (*domain->hfisvc.initialize)(domain->hfisvc.ctxs[ctx_index].ctx);
	if (ret) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"[HFISVC] hfisvc_client_initialize failed for ctx %d, ret=%d\n", ctx_index, ret);
		return -FI_ENODEV;
	}

	ret = (*domain->hfisvc.get_client_key)(domain->hfisvc.ctxs[ctx_index].ctx,
					       &domain->hfisvc.ctxs[ctx_index].client_key);
	if (ret) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] get_client_key failed for ctx %d, ret=%d\n",
			ctx_index, ret);
		int finalize_ret = (*domain->hfisvc.finalize)(domain->hfisvc.ctxs[ctx_index].ctx);
		if (finalize_ret) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] Failed finalizing context %d, ret=%d\n",
				ctx_index, finalize_ret);
		}
		return -FI_ENODEV;
	}

	OPX_HFISVC_DEBUG_LOG("Initializing hfisvc keyset for ctx %d\n", ctx_index);
	ret = opx_hfisvc_keyset_init(&domain->hfisvc.ctxs[ctx_index].access_key_set);
	if (ret) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] Failed initializing keyset for ctx %d, ret=%d\n",
			ctx_index, ret);
		int finalize_ret = (*domain->hfisvc.finalize)(domain->hfisvc.ctxs[ctx_index].ctx);
		if (finalize_ret) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] Failed finalizing context %d, ret=%d\n",
				ctx_index, finalize_ret);
		}
		return -FI_ENODEV;
	}

	FI_INFO(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] Initialized context %d with client key %u\n", ctx_index,
		domain->hfisvc.ctxs[ctx_index].client_key);

	return FI_SUCCESS;
}

int opx_domain_hfisvc_init(struct fi_opx_domain *domain)
{
	int rc = FI_SUCCESS;

	pthread_mutex_lock(&opx_rdma_ops.lock);

	if (ofi_atomic_get64(&domain->hfisvc.ref_cnt) != OPX_DOMAIN_HFISVC_NOT_INITIALIZED) {
		fi_opx_ref_inc(&domain->hfisvc.ref_cnt, "hfisvc");
		goto done;
	}

	/*
	 * Keep an HFISVC-owned dynamic loader reference for the function
	 * pointers cached below.  The HFI1 direct-verbs path also owns a
	 * libhfi1verbs handle, but endpoint/context close paths may release
	 * that global handle before this domain polls its HFISVC queues.
	 */
	domain->hfisvc.libhfi1verbs = dlopen("libhfi1verbs.so.1", RTLD_LAZY);
	if (domain->hfisvc.libhfi1verbs == NULL) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] Could not dlopen libhfi1verbs: %s\n", dlerror());
		rc = -FI_ENODEV;
		goto done;
	}

	FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] libhfi1verbs found\n");

#define OPX_HFISVC_DLSYM(_member, _symbol)                                                              \
	do {                                                                                            \
		domain->hfisvc._member = dlsym(domain->hfisvc.libhfi1verbs, _symbol);                   \
		if (domain->hfisvc._member == NULL) {                                                   \
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] %s not found\n", _symbol); \
			rc = -FI_ENODEV;                                                                \
			goto done;                                                                      \
		}                                                                                       \
	} while (0)

	OPX_HFISVC_DLSYM(initialize, "hfisvc_client_initialize");
	OPX_HFISVC_DLSYM(finalize, "hfisvc_client_finalize");
	OPX_HFISVC_DLSYM(get_client_key, "hfisvc_client_key");
	OPX_HFISVC_DLSYM(command_queue_open, "hfisvc_client_command_queue_open");
	OPX_HFISVC_DLSYM(command_queue_close, "hfisvc_client_command_queue_close");
	OPX_HFISVC_DLSYM(completion_queue_open, "hfisvc_client_completion_queue_open");
	OPX_HFISVC_DLSYM(completion_queue_close, "hfisvc_client_completion_queue_close");
	OPX_HFISVC_DLSYM(cq_read, "hfisvc_client_cq_read");
	OPX_HFISVC_DLSYM(cmd_dma_access_once_va, "hfisvc_client_cmd_dma_access_once_va");
	OPX_HFISVC_DLSYM(cmd_dma_access_once, "hfisvc_client_cmd_dma_access_once");
	OPX_HFISVC_DLSYM(cmd_rdma_read, "hfisvc_client_cmd_rdma_read");
	OPX_HFISVC_DLSYM(cmd_rdma_read_va, "hfisvc_client_cmd_rdma_read_va");
	OPX_HFISVC_DLSYM(cmd_rdma_write, "hfisvc_client_cmd_rdma_write");
	OPX_HFISVC_DLSYM(cmd_mr_open, "hfisvc_client_cmd_mr_open");
	OPX_HFISVC_DLSYM(cmd_mr_close, "hfisvc_client_cmd_mr_close");
	OPX_HFISVC_DLSYM(cmd_dma_access_enable, "hfisvc_client_cmd_dma_access_enable");
	OPX_HFISVC_DLSYM(cmd_dma_access_disable, "hfisvc_client_cmd_dma_access_disable");
	OPX_HFISVC_DLSYM(doorbell, "hfisvc_client_doorbell");

#undef OPX_HFISVC_DLSYM

	/* Initialize the primary context (ctxs[0]) before opening queues. */
	int ret = opx_domain_hfisvc_init_ctx(domain, 0);
	if (ret) {
		rc = ret;
		goto done;
	}

	/* Open domain-level MR command/completion queues on primary context */
	OPX_HFISVC_DEBUG_LOG("Creating domain MR command queue\n");
	ret = (*domain->hfisvc.command_queue_open)(&domain->hfisvc.mr_command_queue, domain->hfisvc.ctxs[0].ctx);
	if (ret) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "[HFISVC] Failed creating domain MR command queue, ret=%d\n",
			ret);
		opx_hfisvc_keyset_free(domain->hfisvc.ctxs[0].access_key_set);
		domain->hfisvc.ctxs[0].access_key_set = 0;
		int finalize_ret		      = (*domain->hfisvc.finalize)(domain->hfisvc.ctxs[0].ctx);
		if (finalize_ret) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"[HFISVC] Failed finalizing primary context, ret=%d\n", finalize_ret);
		}
		rc = -FI_ENODEV;
		goto done;
	}

	OPX_HFISVC_DEBUG_LOG("Creating domain MR completion queue\n");
	ret = (*domain->hfisvc.completion_queue_open)(&domain->hfisvc.mr_completion_queue, domain->hfisvc.ctxs[0].ctx);
	if (ret) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
			"[HFISVC] Failed creating domain MR completion queue, ret=%d\n", ret);
		(*domain->hfisvc.command_queue_close)(&domain->hfisvc.mr_command_queue);
		opx_hfisvc_keyset_free(domain->hfisvc.ctxs[0].access_key_set);
		domain->hfisvc.ctxs[0].access_key_set = 0;
		int finalize_ret		      = (*domain->hfisvc.finalize)(domain->hfisvc.ctxs[0].ctx);
		if (finalize_ret) {
			FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN,
				"[HFISVC] Failed finalizing primary context, ret=%d\n", finalize_ret);
		}
		rc = -FI_ENODEV;
		goto done;
	}

	domain->hfisvc.num_ctxs = 1;
	ofi_atomic_set64(&domain->hfisvc.ref_cnt, 1);
	domain->use_hfisvc = 1;

done:
	if (rc != FI_SUCCESS) {
		if (domain->hfisvc.libhfi1verbs != NULL) {
			dlclose(domain->hfisvc.libhfi1verbs);
			domain->hfisvc.libhfi1verbs = NULL;
		}
	}
	pthread_mutex_unlock(&opx_rdma_ops.lock);

	return rc;
}
#endif
