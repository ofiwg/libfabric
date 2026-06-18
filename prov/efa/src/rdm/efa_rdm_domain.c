/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <assert.h>
#include <ofi_util.h>

#include "config.h"
#include "efa.h"
#include "efa_av.h"
#include "efa_domain_util.h"
#include "efa_rdm_cntr.h"
#include "efa_rdm_cq.h"
#include "efa_rdm_atomic.h"
#include "efa_rdm_domain.h"
#include "efa_rdm_mr.h"

static int efa_rdm_domain_close(fid_t fid);

static int efa_rdm_domain_ops_open(struct fid *fid, const char *ops_name,
				uint64_t flags, void **ops, void *context);

static struct fi_ops efa_ops_domain_fid_rdm = {
	.size = sizeof(struct fi_ops),
	.close = efa_rdm_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = efa_rdm_domain_ops_open,
};

static struct fi_ops_domain efa_domain_ops_rdm = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = efa_av_open,
	.cq_open = efa_rdm_cq_open,
	.endpoint = efa_rdm_ep_open,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = efa_rdm_cntr_open,
	.poll_open = fi_poll_create,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = efa_rdm_atomic_query,
	.query_collective = fi_no_query_collective,
};

static int efa_rdm_domain_init(struct efa_rdm_domain *rdm_domain, struct fi_info *info)
{
	struct efa_domain *efa_domain = &rdm_domain->efa_domain;
	struct fi_info *shm_info = NULL;
	int err;

	assert(EFA_INFO_TYPE_IS_RDM(info));

	/*
	 * Open the MR cache if application did not set FI_MR_LOCAL
	 * and the cache is enabled
	 *
	 * Explicit memory registrations from external application
	 * should never go in the MR cache
	 */
	rdm_domain->cache = NULL;
	if (!efa_domain->mr_local && efa_mr_cache_enable) {
		err = efa_rdm_mr_cache_open(&rdm_domain->cache,
						    efa_domain);
		if (err)
			return err;
	}
	efa_domain->util_domain.domain_fid.mr = &efa_rdm_domain_mr_ops;

	efa_shm_info_create(info, &shm_info);
	if (shm_info && !efa_domain->fabric->shm_fabric) {
		err = fi_fabric(shm_info->fabric_attr,
				&efa_domain->fabric->shm_fabric,
				efa_domain->fabric->util_fabric.fabric_fid.fid.context);
		if (err) {
			EFA_WARN(FI_LOG_DOMAIN, 
				 "Failed to create shm_fabric: %s\n",
				 fi_strerror(-err));
			return err;
		}
	}

	if (efa_domain->fabric->shm_fabric) {
		err = fi_domain(efa_domain->fabric->shm_fabric, shm_info,
				&rdm_domain->shm_domain, NULL);
		if (err)
			return err;
	}

	rdm_domain->mtu_size = efa_domain->device->ibv_port_attr.max_msg_sz;
	rdm_domain->addrlen = (info->src_addr) ? info->src_addrlen : info->dest_addrlen;
	rdm_domain->rdm_cq_size = MAX(info->rx_attr->size + info->tx_attr->size,
				  efa_env.cq_size);
	rdm_domain->num_read_msg_in_flight = 0;

	dlist_init(&rdm_domain->ope_queued_list);
	dlist_init(&rdm_domain->ope_longcts_send_list);
	dlist_init(&rdm_domain->peer_backoff_list);
	dlist_init(&rdm_domain->handshake_queued_peer_list);
	dlist_init(&rdm_domain->ah_lru_list);

	if (shm_info)
		fi_freeinfo(shm_info);

	return 0;
}

/* @brief Allocate an efa-rdm domain.
 *
 * This function creates a domain and uses the info struct to configure
 * the domain based on what capabilities are set. Allocates the extended
 * struct efa_rdm_domain (MR cache, SHM, peer tracking, op-entry lists).
 * The "efa" fabric also serves DGRAM, which uses the base struct;
 * DGRAM requests are forwarded to efa_domain_open. Fork support is
 * checked here and the MR cache is also set up here.
 *
 * @param fabric_fid fabric that the domain should be tied to
 * @param info info struct that was validated and returned by fi_getinfo
 * @param domain_fid pointer where newly domain fid should be stored
 * @param context void pointer stored with the domain fid
 * @return 0 on success, fi_errno on error
 */
int efa_rdm_domain_open(struct fid_fabric *fabric_fid, struct fi_info *info,
			struct fid_domain **domain_fid, void *context)
{
	struct efa_rdm_domain *rdm_domain;
	struct efa_domain *efa_domain;
	int ret = 0, err;
	bool use_lock;

	/* DGRAM is also served by the "efa" fabric but uses the base
	 * struct and base ops. Forward to the base path.
	 */
	if (EFA_INFO_TYPE_IS_DGRAM(info))
		return efa_domain_open(fabric_fid, info, domain_fid, context);

	rdm_domain = calloc(1, sizeof(struct efa_rdm_domain));
	if (!rdm_domain) {
		*domain_fid = NULL;
		return -FI_ENOMEM;
	}
	efa_domain = &rdm_domain->efa_domain;

	/* Initialize srx_lock first so efa_rdm_domain_close can always destroy it */
	use_lock = info->domain_attr &&
		   ofi_thread_level(info->domain_attr->threading) <= ofi_thread_level(FI_THREAD_COMPLETION);
	err = ofi_genlock_init(&rdm_domain->srx_lock, use_lock ? OFI_LOCK_MUTEX : OFI_LOCK_NOOP);
	if (err) {
		EFA_WARN(FI_LOG_DOMAIN, "srx lock init failed! err: %d\n", err);
		free(rdm_domain);
		*domain_fid = NULL;
		return err;
	}

	if (!EFA_INFO_TYPE_IS_RDM(info)) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "efa_rdm_domain_open called with non-rdm info\n");
		ofi_genlock_destroy(&rdm_domain->srx_lock);
		free(rdm_domain);
		*domain_fid = NULL;
		return -FI_EINVAL;
	}
	efa_domain->info_type = EFA_INFO_RDM;

	err = efa_domain_init_base(efa_domain, fabric_fid, info, context);
	if (err) {
		ret = err;
		goto err_free;
	}

	err = efa_mr_pool_create(efa_domain, sizeof(struct efa_rdm_mr));
	if (err) {
		ret = err;
		goto err_free;
	}

	*domain_fid = &efa_domain->util_domain.domain_fid;

	efa_domain->util_domain.domain_fid.fid.ops = &efa_ops_domain_fid_rdm;

	err = efa_rdm_domain_init(rdm_domain, info);
	if (err) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "efa_rdm_domain_init failed. err: %d\n",
			 -err);
		ret = err;
		goto err_free;
	}
	efa_domain->util_domain.domain_fid.ops = &efa_domain_ops_rdm;

	err = efa_domain_finalize_base(efa_domain);
	if (err) {
		ret = err;
		goto err_free;
	}

	return 0;

err_free:
	assert(efa_domain);
	/* TODO: efa_rdm_domain_close called here on a partially-constructed domain
	 * is unsafe when ofi_domain_init or ofi_genlock_init failed since
	 * efa_rdm_domain_close takes the possibly uninitialized lock. The proper
	 * fix will involve fixing ofi_domain_init() to clean up properly
	 * on failure path
	 */
	err = efa_rdm_domain_close(&efa_domain->util_domain.domain_fid.fid);
	if (err) {
		EFA_WARN(FI_LOG_DOMAIN, "When handling error (%d), domain resource was being released. "
			 "During the release process, an additional error (%d) was encountered\n",
			 -ret, -err);
	}

	*domain_fid = NULL;
	return ret;
}

static int efa_rdm_domain_close(fid_t fid)
{
	struct efa_domain *efa_domain;
	struct efa_rdm_domain *rdm_domain;
	int ret;

	efa_domain = container_of(fid, struct efa_domain,
				  util_domain.domain_fid.fid);
	rdm_domain = (struct efa_rdm_domain *) efa_domain;

	if (rdm_domain->cache) {
		ofi_mr_cache_cleanup(rdm_domain->cache);
		free(rdm_domain->cache);
		rdm_domain->cache = NULL;
	}

	if (rdm_domain->shm_domain) {
		ret = fi_close(&rdm_domain->shm_domain->fid);
		if (ret)
			EFA_WARN(FI_LOG_DOMAIN, "Failed to close shm_domain: %d\n", ret);
		rdm_domain->shm_domain = NULL;
	}

	efa_domain_destruct(efa_domain);

	ofi_genlock_destroy(&rdm_domain->srx_lock);
	free(rdm_domain);
	return 0;
}

static int
efa_rdm_domain_ops_open(struct fid *fid, const char *ops_name, uint64_t flags,
			void **ops, void *context)
{
	int ret = FI_SUCCESS;

	if (strcmp(ops_name, FI_EFA_DOMAIN_OPS) == 0) {
		*ops = &efa_ops_domain;
		return ret;
	}
	if (strcmp(ops_name, FI_EFA_GDA_OPS) == 0) {
		EFA_WARN(FI_LOG_DOMAIN, "Only efa direct supports FI_EFA_GDA_OPS\n");
		return -FI_EOPNOTSUPP;
	}

	EFA_WARN(FI_LOG_DOMAIN, "Unknown ops name: %s\n", ops_name);
	ret = -FI_EINVAL;

	return ret;
}

void efa_rdm_domain_progress_peers_and_queues(struct efa_rdm_domain *rdm_domain)
{
	struct efa_rdm_peer *peer;
	struct dlist_entry *tmp;
	struct efa_rdm_ope *ope;
	int ret;

	assert(rdm_domain->efa_domain.info->ep_attr->type == FI_EP_RDM);

	/* Update timers for peers that are in backoff list*/
	dlist_foreach_container_safe(&rdm_domain->peer_backoff_list, struct efa_rdm_peer,
				     peer, rnr_backoff_entry, tmp) {
		if (ofi_gettime_us() >= peer->rnr_backoff_begin_ts +
					peer->rnr_backoff_wait_time) {
			peer->flags &= ~EFA_RDM_PEER_IN_BACKOFF;
			dlist_remove(&peer->rnr_backoff_entry);
		}
	}

	/*
	 * Resend handshake packet for any peers where the first
	 * handshake send failed.
	 */
	dlist_foreach_container_safe(&rdm_domain->handshake_queued_peer_list,
				     struct efa_rdm_peer, peer,
				     handshake_queued_entry, tmp) {
		if (peer->flags & EFA_RDM_PEER_IN_BACKOFF)
			continue;

		ret = efa_rdm_ep_post_handshake(peer->ep, peer);
		if (ret == -FI_EAGAIN)
			continue;

		if (OFI_UNLIKELY(ret)) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "Failed to post HANDSHAKE to peer fi_addr: "
				 "%ld implicit fi_addr: %ld. %s\n",
				 peer->conn->fi_addr,
				 peer->conn->implicit_fi_addr,
				 fi_strerror(-ret));
			efa_base_ep_write_eq_error(&peer->ep->base_ep, -ret, FI_EFA_ERR_PEER_HANDSHAKE);
			continue;
		}

		dlist_remove(&peer->handshake_queued_entry);
		peer->flags &= ~EFA_RDM_PEER_HANDSHAKE_QUEUED;
		peer->flags |= EFA_RDM_PEER_HANDSHAKE_SENT;
	}

	/*
	 * Repost pkts for all queued op entries
	 */
	dlist_foreach_container_safe(&rdm_domain->ope_queued_list,
				     struct efa_rdm_ope,
				     ope, queued_entry, tmp) {

		peer = ope->peer;
		if (peer && (peer->flags & EFA_RDM_PEER_IN_BACKOFF))
			continue;

		if (efa_rdm_ope_process_queued_ope(ope, EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE))
			continue;
		if (efa_rdm_ope_process_queued_ope(ope, EFA_RDM_OPE_QUEUED_RNR))
			continue;
		if (efa_rdm_ope_process_queued_ope(ope, EFA_RDM_OPE_QUEUED_CTRL))
			continue;
		if (efa_rdm_ope_process_queued_ope(ope, EFA_RDM_OPE_QUEUED_READ))
			continue;
	}
	/*
	 * Send data packets until window or data queue is exhausted.
	 */
	dlist_foreach_container_safe(&rdm_domain->ope_longcts_send_list,
				     struct efa_rdm_ope,
				     ope, entry, tmp) {
		peer = ope->peer;
		assert(peer);
		if (peer->flags & EFA_RDM_PEER_IN_BACKOFF)
			continue;

		/*
		 * Do not send DATA packet until we received HANDSHAKE packet from the peer,
		 * this is because endpoint does not know whether peer need connid in header
		 * until it get the HANDSHAKE packet.
		 *
		 * We only do this for DATA packet because other types of packets always
		 * has connid in there packet header. If peer does not make use of the connid,
		 * the connid can be safely ignored.
		 *
		 * DATA packet is different because for DATA packet connid is an optional
		 * header inserted between the mandatory header and the application data.
		 * Therefore if peer does not use/understand connid, it will take connid
		 * as application data thus cause data corruption.
		 *
		 * This will not cause deadlock because peer will send a HANDSHAKE packet
		 * back upon receiving 1st packet from the endpoint, and in all 3 sub0protocols
		 * (long-CTS message, emulated long-CTS write and emulated long-CTS read)
		 * where DATA packet is used, endpoint will send other types of packet to
		 * peer before sending DATA packets. The workflow of the 3 sub-protocol
		 * can be found in protocol v4 document chapter 3.
		 */
		if (!(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED))
			continue;

		if (ope->window > 0) {
			if (efa_rdm_mr_gen_check_ope(ope))
				ret = efa_rdm_ope_post_send(ope, EFA_RDM_CTSDATA_PKT);
			else
				ret = -FI_ECANCELED;

			if (OFI_UNLIKELY(ret)) {
				if (ret != -FI_EAGAIN)
					efa_rdm_txe_handle_error(ope, -ret, FI_EFA_ERR_PKT_POST);
			}
		}
	}
}
