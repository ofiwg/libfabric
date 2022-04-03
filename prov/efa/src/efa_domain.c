/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2017-2020 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "config.h"

#include <ofi_util.h>
#include "efa.h"
#include "rxr_cntr.h"

static int efa_domain_close(fid_t fid)
{
	struct efa_domain *domain;
	int ret;

	domain = container_of(fid, struct efa_domain,
			      util_domain.domain_fid.fid);

	if (efa_is_cache_available(domain)) {
		ofi_mr_cache_cleanup(domain->cache);
		free(domain->cache);
		domain->cache = NULL;
	}

	if (domain->ibv_pd) {
		domain->ibv_pd = NULL;
	}

	ret = ofi_domain_close(&domain->util_domain);
	if (ret)
		return ret;

	if (domain->shm_domain) {
		ret = fi_close(&domain->shm_domain->fid);
		if (ret)
			return ret;
	}

	fi_freeinfo(domain->info);
	free(domain->qp_table);
	free(domain);
	return 0;
}

static int efa_open_device_by_name(struct efa_domain *domain, const char *name)
{
	int i, ret = -FI_ENODEV;
	int name_len;

	if (!name)
		return -FI_EINVAL;

	if (domain->type == EFA_DOMAIN_RDM)
		name_len = strlen(name) - strlen(efa_rdm_domain.suffix);
	else
		name_len = strlen(name) - strlen(efa_dgrm_domain.suffix);

	for (i = 0; i < g_device_cnt; i++) {
		ret = strncmp(name, g_device_list[i].ibv_ctx->device->name, name_len);
		if (!ret) {
			domain->device = &g_device_list[i];
			break;
		}
	}

	if (i == g_device_cnt)
		return -FI_ENODEV;

	domain->ibv_pd = domain->device->ibv_pd;
	return 0;
}

static struct fi_ops efa_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_domain efa_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = efa_av_open,
	.cq_open = efa_cq_open,
	.endpoint = efa_ep_open,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = efa_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
	.query_collective = fi_no_query_collective,
};

/* @brief Setup the MR cache.
 *
 * This function enables the MR cache using the util MR cache code. Note that
 * if the call to ofi_mr_cache_init fails, we continue but disable the cache.
 *
 * @param efa_domain The EFA domain where cache ops should be set
 * @param info Validated info struct selected by the user
 * @return 0 on success, fi_errno on failure.
 */
static int efa_mr_cache_init(struct efa_domain *domain, struct fi_info *info)
{
	struct ofi_mem_monitor *memory_monitors[OFI_HMEM_MAX] = {
		[FI_HMEM_SYSTEM] = default_monitor,
		[FI_HMEM_CUDA] = cuda_monitor,
	};
	int ret;

	/* Both Open MPI (and possibly other MPI implementations) and
	 * Libfabric use the same live binary patching to enable memory
	 * monitoring, but the patching technique only allows a single
	 * "winning" patch.  The Libfabric memhooks monitor will not
	 * overwrite a previous patch, but instead return
	 * -FI_EALREADY.  There are three cases of concern, and in all
	 * but one of them, we can avoid changing the default monitor.
	 *
	 * (1) Upper layer does not patch, such as Open MPI 4.0 and
	 * earlier.  In this case, the default monitor will be used,
	 * as the default monitor is either not the memhooks monitor
	 * (because the user specified a different monitor) or the
	 * default monitor is the memhooks monitor, but we were able
	 * to install the patches.  We will use the default monitor in
	 * this case.
	 *
	 * (2) Upper layer does patch, but does not export a memory
	 * monitor, such as Open MPI 4.1.0 and 4.1.1.  In this case,
	 * if the default memory monitor is not memhooks, we will use
	 * the default monitor.  If the default monitor is memhooks,
	 * the patch will fail to apply, and we will change the
	 * requested monitor to UFFD to avoid a broken configuration.
	 * If the user explicitly requested memhooks, we will return
	 * an error, as we can not satisfy that request.
	 *
	 * (3) Upper layer does patch and exports a memory monitor,
	 * such as Open MPI 4.1.2 and later.  In this case, the
	 * default monitor will have been changed from the memhooks
	 * monitor to the imported monitor, so we will use the
	 * imported monitor.
	 *
	 * The only known cases in which we will not use the default
	 * monitor are Open MPI 4.1.0/4.1.1.
	 *
	 * It is possible that this could be better handled at the
	 * mem_monitor level in Libfabric, but so far we have not
	 * reached agreement on how that would work.
	 */
	if (default_monitor == memhooks_monitor) {
		ret = memhooks_monitor->start(memhooks_monitor);
		if (ret == -FI_EALREADY) {
			if (cache_params.monitor) {
				EFA_WARN(FI_LOG_DOMAIN,
					 "Memhooks monitor requested via FI_MR_CACHE_MONITOR, but memhooks failed to\n"
					 "install.  No working monitor availale.\n");
				return -FI_ENOSYS;
			}
			EFA_INFO(FI_LOG_DOMAIN,
				 "Detected potential memhooks monitor conflict. Switching to UFFD.\n");
			memory_monitors[FI_HMEM_SYSTEM] = uffd_monitor;
		}
	} else if (default_monitor == NULL) {
		/* TODO: Fail if we don't find a system monitor.  This
		 * is a debatable decision, as the VERBS provider
		 * falls back to a no-cache mode in this case.  We
		 * fail the domain creation because the rest of the MR
		 * code hasn't been audited to deal with a NULL
		 * monitor.
		 */
		EFA_WARN(FI_LOG_DOMAIN,
			 "No default SYSTEM monitor available.\n");
		return -FI_ENOSYS;
	}

	domain->cache = (struct ofi_mr_cache *)calloc(1, sizeof(struct ofi_mr_cache));
	if (!domain->cache)
		return -FI_ENOMEM;

	if (!efa_mr_max_cached_count)
		efa_mr_max_cached_count = info->domain_attr->mr_cnt *
					  EFA_MR_CACHE_LIMIT_MULT;
	if (!efa_mr_max_cached_size)
		efa_mr_max_cached_size = domain->device->ibv_attr.max_mr_size *
					 EFA_MR_CACHE_LIMIT_MULT;
	/*
	 * XXX: we're modifying a global in the util mr cache? do we need an
	 * API here instead?
	 */
	cache_params.max_cnt = efa_mr_max_cached_count;
	cache_params.max_size = efa_mr_max_cached_size;
	domain->cache->entry_data_size = sizeof(struct efa_mr);
	domain->cache->add_region = efa_mr_cache_entry_reg;
	domain->cache->delete_region = efa_mr_cache_entry_dereg;
	ret = ofi_mr_cache_init(&domain->util_domain, memory_monitors,
				domain->cache);
	if (!ret) {
		domain->util_domain.domain_fid.mr = &efa_domain_mr_cache_ops;
		EFA_INFO(FI_LOG_DOMAIN, "EFA MR cache enabled, max_cnt: %zu max_size: %zu\n",
			 cache_params.max_cnt, cache_params.max_size);
	} else {
		EFA_WARN(FI_LOG_DOMAIN, "EFA MR cache init failed: %s\n",
		         fi_strerror(ret));
		free(domain->cache);
		domain->cache = NULL;
	}

	return 0;
}

/* @brief Allocate a domain, open the device, and set it up based on the hints.
 *
 * This function creates a domain and uses the info struct to configure the
 * domain based on what capabilities are set. Fork support is checked here and
 * the MR cache is also set up here.
 *
 * Note the trickery with rxr_domain where detect whether this endpoint is RDM
 * or DGRAM to set some state in rxr_domain. We can do this as the type field
 * is at the beginning of efa_domain and rxr_domain, and we know efa_domain
 * stored within rxr_domain. This will be removed when rxr_domain_open and
 * efa_domain_open are combined.
 *
 * @param fabric_fid fabric that the domain should be tied to
 * @param info info struct that was validated and returned by fi_getinfo
 * @param domain_fid pointer where newly domain fid should be stored
 * @param context void pointer stored with the domain fid
 * @return 0 on success, fi_errno on error
 */
int efa_domain_open(struct fid_fabric *fabric_fid, struct fi_info *info,
		    struct fid_domain **domain_fid, void *context)
{
	struct efa_domain *domain;
	struct efa_fabric *fabric;
	const struct fi_info *fi;
	size_t qp_table_size;
	bool app_mr_local;
	int ret, err;

	fi = efa_get_efa_info(info->domain_attr->name);
	if (!fi)
		return -FI_EINVAL;

	fabric = container_of(fabric_fid, struct efa_fabric,
			      util_fabric.fabric_fid);
	ret = ofi_check_domain_attr(&efa_prov, fabric_fid->api_version,
				    fi->domain_attr, info);
	if (ret)
		return ret;

	domain = calloc(1, sizeof(*domain));
	if (!domain)
		return -FI_ENOMEM;

	qp_table_size = roundup_power_of_two(info->domain_attr->ep_cnt);
	domain->qp_table_sz_m1 = qp_table_size - 1;
	domain->qp_table = calloc(qp_table_size, sizeof(*domain->qp_table));
	if (!domain->qp_table) {
		ret = -FI_ENOMEM;
		goto err_free_domain;
	}

	ret = ofi_domain_init(fabric_fid, info, &domain->util_domain,
			      context, 0);
	if (ret)
		goto err_free_qp_table;

	domain->info = fi_dupinfo(info);
	if (!domain->info) {
		ret = -FI_ENOMEM;
		goto err_close_domain;
	}

	if (EFA_EP_TYPE_IS_RDM(info)) {
		struct rxr_domain *rxr_domain;
		domain->type = EFA_DOMAIN_RDM;
		rxr_domain = container_of(domain_fid, struct rxr_domain,
					  rdm_domain);
		app_mr_local = rxr_domain->rxr_mr_local;
	} else {
		domain->type = EFA_DOMAIN_DGRAM;
		/* DGRAM always requires FI_MR_LOCAL */
		app_mr_local = true;
	}

	ret = efa_open_device_by_name(domain, info->domain_attr->name);
	if (ret)
		goto err_free_info;

	domain->util_domain.domain_fid.fid.ops = &efa_fid_ops;
	domain->util_domain.domain_fid.ops = &efa_domain_ops;
	/* RMA mr_modes are being removed, since EFA layer
	 * does not have RMA capabilities. Hence, adding FI_MR_VIRT_ADDR
	 * until RMA capabilities are added to EFA layer
	 */
	domain->util_domain.mr_map.mode |= FI_MR_VIRT_ADDR;
	/*
	 * ofi_domain_init() would have stored the EFA mr_modes in the mr_map,
	 * but we need the rbtree insertions and lookups to use EFA provider's
	 * specific key, so unset the FI_MR_PROV_KEY bit for mr_map.
	 */
	domain->util_domain.mr_map.mode &= ~FI_MR_PROV_KEY;
	domain->fab = fabric;

	domain->util_domain.domain_fid.mr = &efa_domain_mr_ops;

	*domain_fid = &domain->util_domain.domain_fid;

	domain->cache = NULL;

	ret = efa_fork_support_enable_if_requested(*domain_fid);
	if (ret) {
		EFA_WARN(FI_LOG_DOMAIN, "Failed to initialize fork support %d", ret);
		goto err_free_info;
	}

	if (EFA_EP_TYPE_IS_RDM(info)) {
		ret = efa_hmem_support_status_update_all(domain->hmem_support_status);
		if (ret) {
			EFA_WARN(FI_LOG_DOMAIN,
				 "efa_check_hmem_support failed: %s\n",
				 fi_strerror(-ret));
			goto err_free_info;
		}
	}

	/*
	 * If FI_MR_LOCAL is set, we do not want to use the MR cache.
	 */
	if (!app_mr_local && efa_mr_cache_enable) {
		ret = efa_mr_cache_init(domain, info);
		if (ret)
			goto err_free_info;
	}

	return 0;
err_free_info:
	fi_freeinfo(domain->info);
err_close_domain:
	err = ofi_domain_close(&domain->util_domain);
	if (err) {
		EFA_WARN(FI_LOG_DOMAIN,
			   "ofi_domain_close fails: %d", err);
	}
err_free_qp_table:
	free(domain->qp_table);
err_free_domain:
	free(domain);
	return ret;
}

