/*
 * Copyright (c) 2018-2019 Cray Inc. All rights reserved.
 * Copyright (c) 2018-2021 System Fabric Works, Inc. All rights reserved.
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
#include "fi_verbs.h"

void vrb_set_xrc_cm_data(struct vrb_xrc_cm_data *local, uint16_t port,
			 uint32_t tgt_qpn, uint32_t srqn)
{
	local->version = VRB_XRC_VERSION;
	local->port = htons(port);
	local->tgt_qpn = htonl(tgt_qpn);
	local->srqn = htonl(srqn);
}

int vrb_verify_xrc_cm_data(struct vrb_xrc_cm_data *remote,
			      int private_data_len)
{
	if (sizeof(*remote) > private_data_len) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "XRC MSG EP CM data length mismatch\n");
		return -FI_EINVAL;
	}

	if (remote->version != VRB_XRC_VERSION) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "XRC MSG EP connection protocol mismatch "
			   "(local %"PRIu8", remote %"PRIu8")\n",
			   VRB_XRC_VERSION, remote->version);
		return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

static void vrb_log_ep_conn(struct vrb_xrc_ep *ep, char *desc)
{
	struct sockaddr *addr;
	char buf[OFI_ADDRSTRLEN];
	size_t len;

	if (!fi_log_enabled(&vrb_prov, FI_LOG_INFO, FI_LOG_EP_CTRL))
		return;

	VERBS_INFO(FI_LOG_EP_CTRL, "EP %p, %s\n", (void *) ep, desc);
	VERBS_INFO(FI_LOG_EP_CTRL,
		  "EP %p, CM ID %p, TGT CM ID %p, SRQN %d Peer SRQN %d\n",
		  (void*) ep, (void *) ep->base_ep.id, (void *) ep->tgt_id,
		  ep->srqn, ep->peer_srqn);


	if (ep->base_ep.id) {
		addr = rdma_get_local_addr(ep->base_ep.id);
		len = sizeof(buf);
		ofi_straddr(buf, &len, ep->base_ep.info_attr.addr_format, addr);
		VERBS_INFO(FI_LOG_EP_CTRL, "EP %p src_addr: %s\n",
			   (void *) ep, buf);

		addr = rdma_get_peer_addr(ep->base_ep.id);
		len = sizeof(buf);
		ofi_straddr(buf, &len, ep->base_ep.info_attr.addr_format, addr);
		VERBS_INFO(FI_LOG_EP_CTRL, "EP %p dst_addr: %s\n",
			   (void *) ep, buf);
	}

	if (ep->base_ep.ibv_qp) {
		VERBS_INFO(FI_LOG_EP_CTRL, "EP %p, INI QP Num %d\n",
			   (void *) ep, ep->base_ep.ibv_qp->qp_num);
		VERBS_INFO(FI_LOG_EP_CTRL, "EP %p, Remote TGT QP Num %d\n",
			   (void *) ep, ep->ini_conn->tgt_qpn);
	}
	if (ep->tgt_ibv_qp)
		VERBS_INFO(FI_LOG_EP_CTRL, "EP %p, TGT QP Num %d\n",
			   (void *) ep, ep->tgt_ibv_qp->qp_num);
}

int vrb_create_xrc_cm_event(struct vrb_xrc_ep *ep, uint32_t event)
{
	struct fi_eq_cm_entry entry = {
		.fid = &ep->base_ep.util_ep.ep_fid.fid,
	};
	struct vrb_eq_entry *eq_entry;

	assert(fastlock_held(&ep->base_ep.eq->lock));
	eq_entry = vrb_eq_alloc_entry(event, &entry, sizeof(entry));
	if (!eq_entry) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to create %d EQ Event\n",
			   event);
		return -FI_ENOMEM;
	}
	dlistfd_insert_tail(&eq_entry->item, &ep->base_ep.eq->list_head);

	return FI_SUCCESS;
}

/* Caller must hold eq:lock */
void vrb_free_xrc_conn_setup(struct vrb_xrc_ep *ep, int disconnect)
{
	/* A disconnect is requested on the RX (target) side to release
	 * CM ID resources, it does not tear down the connection. */
	if (disconnect) {
		assert(ep->tgt_id);
		assert(!ep->tgt_id->qp);

		if (ep->tgt_id->ps == RDMA_PS_UDP) {
			rdma_destroy_id(ep->tgt_id);
			ep->tgt_id = NULL;
		} else {
			rdma_disconnect(ep->tgt_id);
			ep->tgt_disconnect_sent = true;
		}
	}

	if (!disconnect) {
		free(ep->conn_setup);
		ep->conn_setup = NULL;
		free(ep->base_ep.info_attr.src_addr);
		ep->base_ep.info_attr.src_addr = NULL;
		ep->base_ep.info_attr.src_addrlen = 0;
	}
}

int vrb_connect_xrc(struct vrb_xrc_ep *ep, struct sockaddr *addr,
		    void *param, size_t paramlen)
{
	struct vrb_domain *domain = vrb_ep_to_domain(&ep->base_ep);
	int ret;

	assert(fastlock_held(&ep->base_ep.eq->lock));
	assert(!ep->base_ep.id && !ep->base_ep.ibv_qp && !ep->ini_conn);

	domain->xrc.lock_acquire(&domain->xrc.ini_lock);
	ret = vrb_get_shared_ini_conn(ep, &ep->ini_conn);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Get of shared XRC INI connection failed %d\n", ret);
		free(ep->conn_setup);
		ep->conn_setup = NULL;
		domain->xrc.lock_release(&domain->xrc.ini_lock);
		return ret;
	}

	vrb_add_pending_ini_conn(ep, param, paramlen);
	vrb_sched_ini_conn(ep->ini_conn);
	domain->xrc.lock_release(&domain->xrc.ini_lock);

	return FI_SUCCESS;
}

void vrb_ep_ini_conn_done(struct vrb_xrc_ep *ep, uint32_t tgt_qpn)
{
	struct vrb_domain *domain = vrb_ep_to_domain(&ep->base_ep);

	assert(fastlock_held(&ep->base_ep.eq->lock));
	assert(ep->base_ep.id && ep->ini_conn);

	domain->xrc.lock_acquire(&domain->xrc.ini_lock);
	assert(ep->ini_conn->state == VRB_INI_QP_CONNECTING ||
	       ep->ini_conn->state == VRB_INI_QP_CONNECTED);

	/* If this was a physical INI/TGT QP connection, remove the QP
	 * from control of the RDMA CM. We don't want the shared INI QP
	 * to be destroyed if this endpoint closes. */
	if (ep->base_ep.id == ep->ini_conn->phys_conn_id) {
		ep->ini_conn->phys_conn_id = NULL;
		ep->ini_conn->state = VRB_INI_QP_CONNECTED;
		ep->ini_conn->tgt_qpn = tgt_qpn;
		ep->base_ep.id->qp = NULL;
		VERBS_DBG(FI_LOG_EP_CTRL,
			  "Set INI Conn QP %d remote TGT QP %d\n",
			  ep->ini_conn->ini_qp->qp_num,
			  ep->ini_conn->tgt_qpn);
	}

	vrb_log_ep_conn(ep, "INI/TX Connection Done");
	vrb_sched_ini_conn(ep->ini_conn);
	domain->xrc.lock_release(&domain->xrc.ini_lock);
}

void vrb_ep_ini_conn_rejected(struct vrb_xrc_ep *ep)
{
	assert(fastlock_held(&ep->base_ep.eq->lock));
	assert(ep->base_ep.id && ep->ini_conn);

	vrb_log_ep_conn(ep, "INI/TX Connection Rejected");
	vrb_put_shared_ini_conn(ep);
}

void vrb_ep_tgt_conn_done(struct vrb_xrc_ep *ep)
{
	vrb_log_ep_conn(ep, "TGT/RX Connection Done\n");

	if (ep->tgt_id->qp) {
		assert(ep->tgt_ibv_qp == ep->tgt_id->qp);
		ep->tgt_id->qp = NULL;
	}
}

int vrb_resend_shared_accept_xrc(struct vrb_xrc_ep *ep,
				 struct vrb_connreq *connreq,
				 struct rdma_cm_id *id)
{
	struct rdma_conn_param conn_param = { 0 };
	struct vrb_xrc_cm_data *cm_data = ep->accept_param_data;

	assert(fastlock_held(&ep->base_ep.eq->lock));
	assert(cm_data && ep->tgt_ibv_qp);
	assert(ep->tgt_ibv_qp->qp_num == connreq->xrc.tgt_qpn);
	assert(ep->peer_srqn == connreq->xrc.peer_srqn);

	vrb_set_xrc_cm_data(cm_data, connreq->xrc.port,
			    0, ep->srqn);
	conn_param.private_data = cm_data;
	conn_param.private_data_len = ep->accept_param_len;

	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.rnr_retry_count = 7;
	if (ep->base_ep.srq_ep)
		conn_param.srq = 1;
	conn_param.qp_num = ep->tgt_ibv_qp->qp_num;

	return rdma_accept(id, &conn_param);
}

static int
vrb_xrc_connreq_init(struct vrb_ep *ep, struct vrb_connreq *connreq)
{
	struct vrb_xrc_ep *xrc_ep = container_of(ep, struct vrb_xrc_ep,
						 base_ep);
	int ret;

	assert(ep->info_attr.src_addr);
	assert(ep->info_attr.dest_addr);

	xrc_ep->tgt_id = connreq->id;
	xrc_ep->tgt_id->context = &ep->util_ep.ep_fid.fid;
	if (xrc_ep->base_ep.eq) {
		if (rdma_migrate_id(xrc_ep->tgt_id,
				    xrc_ep->base_ep.eq->channel)) {
			ret = -errno;
			FI_WARN(&vrb_prov, FI_LOG_EP_CTRL,
				"rdma_migrate_id error: %s (%d)\n",
				strerror(-ret), -ret);
			return ret;
		}
	}
	return FI_SUCCESS;
}

int vrb_accept_xrc(struct vrb_xrc_ep *ep, struct fi_info *info,
		   void *param, size_t paramlen)
{
	struct sockaddr *addr;
	struct vrb_connreq *connreq;
	struct rdma_conn_param conn_param = { 0 };
	struct vrb_xrc_cm_data *cm_data = param;
	int ret;

	assert(fastlock_held(&ep->base_ep.eq->lock));
	connreq = container_of(info->handle, struct vrb_connreq,
				       handle);
	ret = vrb_xrc_connreq_init(&ep->base_ep, connreq);
	if (ret)
		return ret;

	addr = rdma_get_local_addr(ep->tgt_id);
	if (addr)
		ofi_straddr_dbg(&vrb_prov, FI_LOG_EP_CTRL, "src_addr", addr);
	addr = rdma_get_peer_addr(ep->tgt_id);
	if (addr)
		ofi_straddr_dbg(&vrb_prov, FI_LOG_EP_CTRL, "dest_addr", addr);

	ret = vrb_ep_create_tgt_qp(ep, connreq->xrc.tgt_qpn);
	if (ret)
		return ret;

	ep->remote_pep_port = connreq->xrc.port;
	vrb_set_xrc_cm_data(cm_data, connreq->xrc.port, 0, ep->srqn);
	conn_param.private_data = cm_data;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.rnr_retry_count = 7;
	if (ep->base_ep.srq_ep)
		conn_param.srq = 1;

	if (!ep->tgt_id->qp)
		conn_param.qp_num = ep->tgt_ibv_qp->qp_num;

	ret = rdma_accept(ep->tgt_id, &conn_param);
	if (ret) {
		ret = -errno;
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "XRC TGT, rdma_accept error %d\n", ret);
		return ret;
	}
	free(connreq);

	if (ep->tgt_id->ps == RDMA_PS_TCP)
		return ret;

	VERBS_DBG(FI_LOG_EP_CTRL, "Saving SIDR accept response\n");
	if (vrb_eq_add_sidr_conn(ep, cm_data, paramlen))
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "SIDR connection accept not added to map\n");

	/* No RTU message in SIDR message sequence, generate the CM
	 * connect. The passive side accept implements only the RX
	 * side of a simplex connection so this is ok */
	vrb_ep_tgt_conn_done(ep);

	return vrb_create_xrc_cm_event(ep, FI_CONNECTED);
}

