/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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

#include <rdma/fi_errno.h>

#include "efa.h"
#include "efa_prov_info.h"
#include "efa_user_info.h"
#include "ofi_hmem.h"

struct rxr_env rxr_env = {
	.tx_min_credits = RXR_DEF_MIN_TX_CREDITS,
	.tx_queue_size = 0,
	.enable_shm_transfer = 1,
	.use_device_rdma = 0,
	.use_zcpy_rx = 1,
	.zcpy_rx_seed = 0,
	.shm_av_size = 128,
	.shm_max_medium_size = 4096,
	.recvwin_size = RXR_RECVWIN_SIZE,
	.ooo_pool_chunk_size = 64,
	.unexp_pool_chunk_size = 1024,
	.readcopy_pool_size = 256,
	.atomrsp_pool_size = 1024,
	.cq_size = RXR_DEF_CQ_SIZE,
	.max_memcpy_size = 4096,
	.mtu_size = 0,
	.tx_size = 0,
	.rx_size = 0,
	.tx_iov_limit = 0,
	.rx_iov_limit = 0,
	.rx_copy_unexp = 1,
	.rx_copy_ooo = 1,
	.rnr_backoff_wait_time_cap = RXR_DEFAULT_RNR_BACKOFF_WAIT_TIME_CAP,
	.rnr_backoff_initial_wait_time = 0, /* 0 is random wait time  */
	.efa_cq_read_size = 50,
	.shm_cq_read_size = 50,
	.efa_max_medium_msg_size = 65536,
	.efa_min_read_msg_size = 1048576,
	.efa_max_gdrcopy_msg_size = 32768,
	.efa_min_read_write_size = 65536,
	.efa_read_segment_size = 1073741824,
	.rnr_retry = 3, /* Setting this value to EFA_RNR_INFINITE_RETRY makes the firmware retry indefinitey */
	.efa_runt_size = 307200,
};

/* @brief Read and store the FI_EFA_* environment variables.
 */
void rxr_init_env(void)
{
	if (getenv("FI_EFA_SHM_MAX_MEDIUM_SIZE")) {
		fprintf(stderr,
			"FI_EFA_SHM_MAX_MEDIUM_SIZE env variable detected! The use of this variable has been deprecated and as such execution cannot proceed.\n");
		abort();
	};

	fi_param_get_int(&rxr_prov, "tx_min_credits", &rxr_env.tx_min_credits);
	if (rxr_env.tx_min_credits <= 0) {
		FI_WARN(&rxr_prov, FI_LOG_EP_DATA,
			"FI_EFA_TX_MIN_CREDITS was set to %d, which is <= 0."
			"This value will cause EFA communication to deadlock."
			"To avoid that, the variable was reset to %d\n",
			rxr_env.tx_min_credits, RXR_DEF_MIN_TX_CREDITS);
		rxr_env.tx_min_credits = RXR_DEF_MIN_TX_CREDITS;
	}

	fi_param_get_int(&rxr_prov, "tx_queue_size", &rxr_env.tx_queue_size);
	fi_param_get_int(&rxr_prov, "enable_shm_transfer", &rxr_env.enable_shm_transfer);
	fi_param_get_int(&rxr_prov, "use_device_rdma", &rxr_env.use_device_rdma);
	fi_param_get_int(&rxr_prov, "use_zcpy_rx", &rxr_env.use_zcpy_rx);
	fi_param_get_int(&rxr_prov, "zcpy_rx_seed", &rxr_env.zcpy_rx_seed);
	fi_param_get_int(&rxr_prov, "shm_av_size", &rxr_env.shm_av_size);
	fi_param_get_int(&rxr_prov, "recvwin_size", &rxr_env.recvwin_size);
	fi_param_get_int(&rxr_prov, "readcopy_pool_size", &rxr_env.readcopy_pool_size);
	fi_param_get_int(&rxr_prov, "cq_size", &rxr_env.cq_size);
	fi_param_get_size_t(&rxr_prov, "max_memcpy_size",
			    &rxr_env.max_memcpy_size);
	fi_param_get_bool(&rxr_prov, "mr_cache_enable",
			  &efa_mr_cache_enable);
	fi_param_get_size_t(&rxr_prov, "mr_max_cached_count",
			    &efa_mr_max_cached_count);
	fi_param_get_size_t(&rxr_prov, "mr_max_cached_size",
			    &efa_mr_max_cached_size);
	fi_param_get_size_t(&rxr_prov, "mtu_size",
			    &rxr_env.mtu_size);
	fi_param_get_size_t(&rxr_prov, "tx_size", &rxr_env.tx_size);
	fi_param_get_size_t(&rxr_prov, "rx_size", &rxr_env.rx_size);
	fi_param_get_size_t(&rxr_prov, "tx_iov_limit", &rxr_env.tx_iov_limit);
	fi_param_get_size_t(&rxr_prov, "rx_iov_limit", &rxr_env.rx_iov_limit);
	fi_param_get_bool(&rxr_prov, "rx_copy_unexp",
			  &rxr_env.rx_copy_unexp);
	fi_param_get_bool(&rxr_prov, "rx_copy_ooo",
			  &rxr_env.rx_copy_ooo);

	fi_param_get_int(&rxr_prov, "max_timeout", &rxr_env.rnr_backoff_wait_time_cap);
	if (rxr_env.rnr_backoff_wait_time_cap > RXR_MAX_RNR_BACKOFF_WAIT_TIME_CAP)
		rxr_env.rnr_backoff_wait_time_cap = RXR_MAX_RNR_BACKOFF_WAIT_TIME_CAP;

	fi_param_get_int(&rxr_prov, "timeout_interval",
			 &rxr_env.rnr_backoff_initial_wait_time);
	fi_param_get_size_t(&rxr_prov, "efa_cq_read_size",
			 &rxr_env.efa_cq_read_size);
	fi_param_get_size_t(&rxr_prov, "shm_cq_read_size",
			 &rxr_env.shm_cq_read_size);
	fi_param_get_size_t(&rxr_prov, "inter_max_medium_message_size",
			    &rxr_env.efa_max_medium_msg_size);
	fi_param_get_size_t(&rxr_prov, "inter_min_read_message_size",
			    &rxr_env.efa_min_read_msg_size);
	fi_param_get_size_t(&rxr_prov, "inter_min_read_write_size",
			    &rxr_env.efa_min_read_write_size);
	fi_param_get_size_t(&rxr_prov, "inter_read_segment_size",
			    &rxr_env.efa_read_segment_size);
	fi_param_get_size_t(&rxr_prov, "inter_max_gdrcopy_message_size",
			    &rxr_env.efa_max_gdrcopy_msg_size);
	fi_param_get_size_t(&rxr_prov, "runt_size",
			    &rxr_env.efa_runt_size);
	efa_fork_support_request_initialize();
}

/*
 * Used to set tx/rx attributes that are characteristic of the device for the
 * two endpoint types and not emulated in software.
 */
void rxr_set_rx_tx_size(struct fi_info *info,
			const struct fi_info *core_info)
{
	if (rxr_env.tx_size > 0)
		info->tx_attr->size = rxr_env.tx_size;
	else
		info->tx_attr->size = core_info->tx_attr->size;

	if (rxr_env.rx_size > 0)
		info->rx_attr->size = rxr_env.rx_size;
	else
		info->rx_attr->size = core_info->rx_attr->size;

	if (rxr_env.tx_iov_limit > 0)
		info->tx_attr->iov_limit = rxr_env.tx_iov_limit;

	if (rxr_env.rx_iov_limit > 0)
		info->rx_attr->iov_limit = rxr_env.rx_iov_limit;
}

static int rxr_dgram_info_to_rxr(uint32_t version,
				 const struct fi_info *core_info,
				 struct fi_info *info) {
	rxr_set_rx_tx_size(info, core_info);
	return 0;
}

static int rxr_info_to_rxr(uint32_t version,
			   struct fi_info *info, const struct fi_info *hints)
{
	uint64_t atomic_ordering;

	/*
	 * Do not advertise FI_HMEM capabilities when the core can not support
	 * it or when the application passes NULL hints (given this is a primary
	 * cap). The logic for device-specific checks pertaining to HMEM comes
	 * further along this path.
	 */
	if (!hints) {
		info->caps &= ~FI_HMEM;
	}

	/*
	 * Handle user-provided hints and adapt the info object passed back up
	 * based on EFA-specific constraints.
	 */
	if (hints) {
		if (hints->tx_attr) {
			atomic_ordering = FI_ORDER_ATOMIC_RAR | FI_ORDER_ATOMIC_RAW |
					  FI_ORDER_ATOMIC_WAR | FI_ORDER_ATOMIC_WAW;
			if (!(hints->tx_attr->msg_order & atomic_ordering)) {			
				info->ep_attr->max_order_raw_size = 0;
			}
		}

		/* We only support manual progress for RMA operations */
		if (hints->caps & FI_RMA) {
			info->domain_attr->control_progress = FI_PROGRESS_MANUAL;
			info->domain_attr->data_progress = FI_PROGRESS_MANUAL;
		}


#if HAVE_CUDA || HAVE_NEURON
		/* If the application requires HMEM support, we will add
		 * FI_MR_HMEM to mr_mode, because we need application to
		 * provide descriptor for cuda or neuron buffer. Note we did
		 * not add FI_MR_LOCAL here because according to FI_MR man
		 * page:
		 *
		 *     "If FI_MR_HMEM is set, but FI_MR_LOCAL is unset,
		 *      only device buffers must be registered when used locally.
		 *      "
		 * which means FI_MR_HMEM implies FI_MR_LOCAL for cuda or neuron buffer.
		 */
		if (hints->caps & FI_HMEM) {
			if (ofi_hmem_p2p_disabled()) {
				FI_WARN(&rxr_prov, FI_LOG_CORE,
					"FI_HMEM capability currently requires peer to peer support, which is disabled.\n");
				return -FI_ENODATA;
			}
			//TODO: remove the rdma checks once FI_HMEM w/o p2p is supported

			if (!efa_device_support_rdma_read()) {
				FI_WARN(&rxr_prov, FI_LOG_CORE,
				        "FI_HMEM capability requires RDMA, which this device does not support.\n");
				return -FI_ENODATA;

			}

			if (!rxr_env.use_device_rdma) {
				FI_WARN(&rxr_prov, FI_LOG_CORE,
				        "FI_HMEM capability requires RDMA, which is turned off. You can turn it on by set environment variable FI_EFA_USE_DEVICE_RDMA to 1.\n");
				return -FI_ENODATA;
			}

			if (hints->domain_attr &&
			    !(hints->domain_attr->mr_mode & FI_MR_HMEM)) {
				FI_WARN(&rxr_prov, FI_LOG_CORE,
				        "FI_HMEM capability requires device registrations (FI_MR_HMEM)\n");
				return -FI_ENODATA;
			}

			info->domain_attr->mr_mode |= FI_MR_HMEM;

		} else {
			/*
			 * FI_HMEM is a primary capability. Providers should
			 * only enable it if requested by applications.
			 */
			info->caps &= ~FI_HMEM;
		}
#endif
		/*
		 * The provider does not force applications to register buffers
		 * with the device, but if an application is able to, reuse
		 * their registrations and avoid the bounce buffers.
		 */
		if (hints->domain_attr && hints->domain_attr->mr_mode & FI_MR_LOCAL)
			info->domain_attr->mr_mode |= FI_MR_LOCAL;

		/*
		 * Same goes for prefix mode, where the protocol does not
		 * absolutely need a prefix before receive buffers, but it can
		 * use it when available to optimize transfers with endpoints
		 * having the following profile:
		 *	- Requires FI_MSG and not FI_TAGGED/FI_ATOMIC/FI_RMA
		 *	- Can handle registrations (FI_MR_LOCAL)
		 *	- No need for FI_DIRECTED_RECV
		 *	- Guaranteed to send msgs smaller than info->nic->link_attr->mtu
		 */
		if (hints->mode & FI_MSG_PREFIX) {
			FI_INFO(&rxr_prov, FI_LOG_CORE,
				"FI_MSG_PREFIX supported by application.\n");
			info->mode |= FI_MSG_PREFIX;
			info->tx_attr->mode |= FI_MSG_PREFIX;
			info->rx_attr->mode |= FI_MSG_PREFIX;
			info->ep_attr->msg_prefix_size = RXR_MSG_PREFIX_SIZE;
			FI_INFO(&rxr_prov, FI_LOG_CORE,
				"FI_MSG_PREFIX size = %ld\n", info->ep_attr->msg_prefix_size);
		}
	}

	/* Use a table for AV if the app has no strong requirement */
	if (!hints || !hints->domain_attr ||
	    hints->domain_attr->av_type == FI_AV_UNSPEC)
		info->domain_attr->av_type = FI_AV_TABLE;
	else
		info->domain_attr->av_type = hints->domain_attr->av_type;

	if (!hints || !hints->domain_attr ||
	    hints->domain_attr->resource_mgmt == FI_RM_UNSPEC)
		info->domain_attr->resource_mgmt = FI_RM_ENABLED;
	else
		info->domain_attr->resource_mgmt = hints->domain_attr->resource_mgmt;

	return 0;
}

static int rxr_dgram_getinfo(uint32_t version, const char *node,
			     const char *service, uint64_t flags,
			     const struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *core_info, *util_info, *cur, *tail;
	int ret;

	core_info = NULL;

	ret = efa_getinfo(version, node, service, flags, hints, &core_info);

	if (ret) {
		*info = NULL;
		return ret;
	}

	ret = -FI_ENODATA;

	*info = NULL;
	tail = NULL;
	for (cur = core_info; cur; cur = cur->next) {
		/* Skip non DGRAM info structs */
		if (cur->ep_attr->type != FI_EP_DGRAM)
			continue;

		ret = 0;

		util_info = fi_dupinfo(cur);
		if (!util_info) {
			ret = -FI_ENOMEM;
			fi_freeinfo(*info);
			goto out;
		}

		rxr_dgram_info_to_rxr(version, cur, util_info);

		if (!*info)
			*info = util_info;
		else
			tail->next = util_info;
		tail = util_info;
	}

out:
	fi_freeinfo(core_info);
	return ret;
}

static
int rxr_rdm_getinfo(uint32_t version, const char *node,
		    const char *service, uint64_t flags,
		    const struct fi_info *hints, struct fi_info **info)
{
	const struct fi_info *prov_info_rxr;
	struct fi_info *dupinfo, *tail;
	int ret;

	ret = efa_user_info_check_hints_addr(node, service, flags, hints);
	if (ret)
		return ret;

	if (hints) {
		ret = ofi_prov_check_info(&rxr_util_prov, version, hints);
		if (ret)
			return ret;
	}

	*info = tail = NULL;
	for (prov_info_rxr = rxr_util_prov.info;
	     prov_info_rxr;
	     prov_info_rxr = prov_info_rxr->next) {
		ret = efa_prov_info_compare_src_addr(node, flags, hints, prov_info_rxr);
		if (ret)
			continue;

		dupinfo = fi_dupinfo(prov_info_rxr);
		if (!dupinfo) {
			ret = -FI_ENOMEM;
			goto free_info;
		}

		ret = efa_user_info_set_dest_addr(node, service, flags, hints, *info);
		if (ret)
			goto free_info;

		dupinfo->fabric_attr->api_version = version;

		ret = rxr_info_to_rxr(version, dupinfo, hints);
		if (ret)
			goto free_info;

		ofi_alter_info(dupinfo, hints, version);

		/* If application asked for FI_REMOTE_COMM but not FI_LOCAL_COMM, it
		 * does not want to use shm. In this case, we honor the request by
		 * unsetting the FI_LOCAL_COMM flag in info. This way rxr_endpoint()
		 * should disable shm transfer for the endpoint
		 */
		if (hints && hints->caps & FI_REMOTE_COMM && !(hints->caps & FI_LOCAL_COMM))
			dupinfo->caps &= ~FI_LOCAL_COMM;

		if (!*info)
			*info = dupinfo;
		else
			tail->next = dupinfo;
		tail = dupinfo;
	}

	return 0;
free_info:
	fi_freeinfo(dupinfo);
	fi_freeinfo(*info);
	*info = NULL;
	return ret;
}

int rxr_getinfo(uint32_t version, const char *node,
		const char *service, uint64_t flags,
		const struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *dgram_info_list, *rdm_info_list;
	int err;
	static bool shm_info_initialized = false;

	/*
	 * efa_shm_info_initialize() initializes the global variable g_shm_info.
	 * Ideally it should be called during provider initialization. However,
	 * At the time of EFA provider initialization, shm provider has not been
	 * initialized yet, therefore g_shm_info cannot be initialized. As a workaround,
	 * we initialize g_shm_info when the rxr_getinfo() is called 1st time,
	 * at this point all the providers have been initialized.
	 */
	if (!shm_info_initialized) {
		efa_shm_info_initialize(hints);
		shm_info_initialized = true;
	}

	if (hints && hints->ep_attr && hints->ep_attr->type == FI_EP_DGRAM)
		return rxr_dgram_getinfo(version, node, service, flags, hints, info);

	if (hints && hints->ep_attr && hints->ep_attr->type == FI_EP_RDM)
		return rxr_rdm_getinfo(version, node, service, flags, hints, info);

	if (hints && hints->ep_attr && hints->ep_attr->type != FI_EP_UNSPEC) {
		EFA_WARN(FI_LOG_DOMAIN, "unsupported endpoint type: %d\n",
			 hints->ep_attr->type);
		return -FI_ENODATA;
	}

	err = rxr_dgram_getinfo(version, node, service, flags, hints, &dgram_info_list);
	if (err && err != -FI_ENODATA) {
		return err;
	}

	err = rxr_rdm_getinfo(version, node, service, flags, hints, &rdm_info_list);
	if (err && err != -FI_ENODATA) {
		fi_freeinfo(dgram_info_list);
		return err;
	}

	if (rdm_info_list && dgram_info_list) {
		struct fi_info *tail;

		tail = rdm_info_list;
		while (tail->next)
			tail = tail->next;

		tail->next = dgram_info_list;
		*info = rdm_info_list;
		return 0;
	}

	if (rdm_info_list) {
		assert(!dgram_info_list);
		*info = rdm_info_list;
		return 0;
	}

	if (dgram_info_list) {
		assert(!rdm_info_list);
		*info = dgram_info_list;
		return 0;
	}

	*info = NULL;
	return -FI_ENODATA;
}

void rxr_define_env()
{
	fi_param_define(&rxr_prov, "tx_min_credits", FI_PARAM_INT,
			"Defines the minimum number of credits a sender requests from a receiver (Default: 32).");
	fi_param_define(&rxr_prov, "tx_queue_size", FI_PARAM_INT,
			"Defines the maximum number of unacknowledged sends with the NIC.");
	fi_param_define(&rxr_prov, "enable_shm_transfer", FI_PARAM_INT,
			"Enable using SHM provider to perform TX operations between processes on the same system. (Default: 1)");
	fi_param_define(&rxr_prov, "use_device_rdma", FI_PARAM_INT,
			"whether to use device's RDMA functionality for one-sided and two-sided transfer.");
	fi_param_define(&rxr_prov, "use_zcpy_rx", FI_PARAM_INT,
			"Enables the use of application's receive buffers in place of bounce-buffers when feasible. (Default: 1)");
	fi_param_define(&rxr_prov, "zcpy_rx_seed", FI_PARAM_INT,
			"Defines the number of bounce-buffers the provider will prepost during EP initialization.  (Default: 0)");
	fi_param_define(&rxr_prov, "shm_av_size", FI_PARAM_INT,
			"Defines the maximum number of entries in SHM provider's address vector (Default 128).");
	fi_param_define(&rxr_prov, "recvwin_size", FI_PARAM_INT,
			"Defines the size of sliding receive window. (Default: 16384)");
	fi_param_define(&rxr_prov, "readcopy_pool_size", FI_PARAM_INT,
			"Defines the size of readcopy packet pool size. (Default: 256)");
	fi_param_define(&rxr_prov, "cq_size", FI_PARAM_INT,
			"Define the size of completion queue. (Default: 8192)");
	fi_param_define(&rxr_prov, "mr_cache_enable", FI_PARAM_BOOL,
			"Enables using the mr cache and in-line registration instead of a bounce buffer for iov's larger than max_memcpy_size. Defaults to true. When disabled, only uses a bounce buffer.");
	fi_param_define(&rxr_prov, "mr_max_cached_count", FI_PARAM_SIZE_T,
			"Sets the maximum number of memory registrations that can be cached at any time.");
	fi_param_define(&rxr_prov, "mr_max_cached_size", FI_PARAM_SIZE_T,
			"Sets the maximum amount of memory that cached memory registrations can hold onto at any time.");
	fi_param_define(&rxr_prov, "max_memcpy_size", FI_PARAM_SIZE_T,
			"Threshold size switch between using memory copy into a pre-registered bounce buffer and memory registration on the user buffer. (Default: 4096)");
	fi_param_define(&rxr_prov, "mtu_size", FI_PARAM_SIZE_T,
			"Override the MTU size of the device.");
	fi_param_define(&rxr_prov, "tx_size", FI_PARAM_SIZE_T,
			"Set the maximum number of transmit operations before the provider returns -FI_EAGAIN. For only the RDM endpoint, this parameter will cause transmit operations to be queued when this value is set higher than the default and the transmit queue is full.");
	fi_param_define(&rxr_prov, "rx_size", FI_PARAM_SIZE_T,
			"Set the maximum number of receive operations before the provider returns -FI_EAGAIN.");
	fi_param_define(&rxr_prov, "tx_iov_limit", FI_PARAM_SIZE_T,
			"Maximum transmit iov_limit.");
	fi_param_define(&rxr_prov, "rx_iov_limit", FI_PARAM_SIZE_T,
			"Maximum receive iov_limit.");
	fi_param_define(&rxr_prov, "rx_copy_unexp", FI_PARAM_BOOL,
			"Enables the use of a separate pool of bounce-buffers to copy unexpected messages out of the pre-posted receive buffers. (Default: 1)");
	fi_param_define(&rxr_prov, "rx_copy_ooo", FI_PARAM_BOOL,
			"Enables the use of a separate pool of bounce-buffers to copy out-of-order RTM packets out of the pre-posted receive buffers. (Default: 1)");
	fi_param_define(&rxr_prov, "max_timeout", FI_PARAM_INT,
			"Set the maximum timeout (us) for backoff to a peer after a receiver not ready error. (Default: 1000000)");
	fi_param_define(&rxr_prov, "timeout_interval", FI_PARAM_INT,
			"Set the time interval (us) for the base timeout to use for exponential backoff to a peer after a receiver not ready error. (Default: 0 [random])");
	fi_param_define(&rxr_prov, "efa_cq_read_size", FI_PARAM_SIZE_T,
			"Set the number of EFA completion entries to read for one loop for one iteration of the progress engine. (Default: 50)");
	fi_param_define(&rxr_prov, "shm_cq_read_size", FI_PARAM_SIZE_T,
			"Set the number of SHM completion entries to read for one loop for one iteration of the progress engine. (Default: 50)");
	fi_param_define(&rxr_prov, "inter_max_medium_message_size", FI_PARAM_INT,
			"The maximum message size for inter EFA medium message protocol (Default 65536).");
	fi_param_define(&rxr_prov, "inter_min_read_message_size", FI_PARAM_INT,
			"The minimum message size for inter EFA read message protocol. If instance support RDMA read, messages whose size is larger than this value will be sent by read message protocol (Default 1048576).");
	fi_param_define(&rxr_prov, "inter_max_gdrcopy_message_size", FI_PARAM_INT,
			"The maximum message size to use gdrcopy. If instance support gdrcopy, messages whose size is smaller than this value will be sent by eager/longcts protocol (Default 32768).");
	fi_param_define(&rxr_prov, "inter_min_read_write_size", FI_PARAM_INT,
			"The mimimum message size for inter EFA write to use read write protocol. If firmware support RDMA read, and FI_EFA_USE_DEVICE_RDMA is 1, write requests whose size is larger than this value will use the read write protocol (Default 65536).");
	fi_param_define(&rxr_prov, "inter_read_segment_size", FI_PARAM_INT,
			"Calls to RDMA read is segmented using this value.");
	fi_param_define(&rxr_prov, "fork_safe", FI_PARAM_BOOL,
			"Enables fork support and disables internal usage of huge pages. Has no effect on kernels which set copy-on-fork for registered pages, generally 5.13 and later. (Default: false)");
	fi_param_define(&rxr_prov, "runt_size", FI_PARAM_INT,
			"The part of message that will be eagerly sent of a runting protocol (Default 0).");
}

