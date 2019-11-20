/*
 * Copyright (c) 2018 Cray Inc. All rights reserved.
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

void fi_ibv_next_xrc_conn_state(struct fi_ibv_xrc_ep *ep)
{
	switch (ep->conn_state) {
	case FI_IBV_XRC_UNCONNECTED:
		ep->conn_state = FI_IBV_XRC_ORIG_CONNECTING;
		break;
	case FI_IBV_XRC_ORIG_CONNECTING:
		ep->conn_state = FI_IBV_XRC_ORIG_CONNECTED;
		break;
	case FI_IBV_XRC_ORIG_CONNECTED:
		ep->conn_state = FI_IBV_XRC_RECIP_CONNECTING;
		break;
	case FI_IBV_XRC_RECIP_CONNECTING:
		ep->conn_state = FI_IBV_XRC_CONNECTED;
		break;
	case FI_IBV_XRC_CONNECTED:
	case FI_IBV_XRC_ERROR:
		break;
	default:
		assert(0);
		VERBS_WARN(FI_LOG_EP_CTRL, "Unkown XRC connection state %d\n",
			   ep->conn_state);
	}
}

void fi_ibv_prev_xrc_conn_state(struct fi_ibv_xrc_ep *ep)
{
	switch (ep->conn_state) {
	case FI_IBV_XRC_UNCONNECTED:
		break;
	case FI_IBV_XRC_ORIG_CONNECTING:
		ep->conn_state = FI_IBV_XRC_UNCONNECTED;
		break;
	case FI_IBV_XRC_ORIG_CONNECTED:
		ep->conn_state = FI_IBV_XRC_ORIG_CONNECTING;
		break;
	case FI_IBV_XRC_RECIP_CONNECTING:
		ep->conn_state = FI_IBV_XRC_ORIG_CONNECTED;
		break;
	case FI_IBV_XRC_CONNECTED:
		ep->conn_state = FI_IBV_XRC_RECIP_CONNECTING;
		break;
	case FI_IBV_XRC_ERROR:
		break;
	default:
		assert(0);
		VERBS_WARN(FI_LOG_EP_CTRL, "Unkown XRC connection state %d\n",
			   ep->conn_state);
	}
}

void fi_ibv_save_priv_data(struct fi_ibv_xrc_ep *ep, const void *data,
			   size_t len)
{
	ep->conn_setup->event_len = MIN(sizeof(ep->conn_setup->event_data),
					len);
	memcpy(ep->conn_setup->event_data, data, ep->conn_setup->event_len);
}

void fi_ibv_set_xrc_cm_data(struct fi_ibv_xrc_cm_data *local, int reciprocal,
			    uint32_t conn_tag, uint16_t port, uint32_t tgt_qpn,
			    uint32_t srqn)
{
	local->version = FI_IBV_XRC_VERSION;
	local->reciprocal = reciprocal ? 1 : 0;
	local->port = htons(port);
	local->conn_tag = htonl(conn_tag);
	local->tgt_qpn = htonl(tgt_qpn);
	local->srqn = htonl(srqn);
}

int fi_ibv_verify_xrc_cm_data(struct fi_ibv_xrc_cm_data *remote,
			      int private_data_len)
{
	if (sizeof(*remote) > private_data_len) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "XRC MSG EP CM data length mismatch\n");
		return -FI_EINVAL;
	}

	if (remote->version != FI_IBV_XRC_VERSION) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "XRC MSG EP connection protocol mismatch "
			   "(local %"PRIu8", remote %"PRIu8")\n",
			   FI_IBV_XRC_VERSION, remote->version);
		return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

void fi_ibv_log_ep_conn(struct fi_ibv_xrc_ep *ep, char *desc)
{
	struct sockaddr *addr;
	char buf[OFI_ADDRSTRLEN];
	size_t len = sizeof(buf);

	if (!fi_log_enabled(&fi_ibv_prov, FI_LOG_INFO, FI_LOG_EP_CTRL))
		return;

	VERBS_INFO(FI_LOG_EP_CTRL, "EP %p, %s\n", ep, desc);
	VERBS_INFO(FI_LOG_EP_CTRL,
		  "EP %p, CM ID %p, TGT CM ID %p, SRQN %d Peer SRQN %d\n",
		  ep, ep->base_ep.id, ep->tgt_id, ep->srqn, ep->peer_srqn);


	if (ep->base_ep.id) {
		addr = rdma_get_local_addr(ep->base_ep.id);
		if (addr) {
			ofi_straddr(buf, &len, ep->base_ep.info->addr_format,
				    addr);
			VERBS_INFO(FI_LOG_EP_CTRL, "EP %p src_addr: %s\n",
				   ep, buf);
		}
		addr = rdma_get_peer_addr(ep->base_ep.id);
		if (addr) {
			len = sizeof(buf);
			ofi_straddr(buf, &len, ep->base_ep.info->addr_format,
				    addr);
			VERBS_INFO(FI_LOG_EP_CTRL, "EP %p dst_addr: %s\n",
				   ep, buf);
		}
	}

	if (ep->base_ep.ibv_qp) {
		VERBS_INFO(FI_LOG_EP_CTRL, "EP %p, INI QP Num %d\n",
			  ep, ep->base_ep.ibv_qp->qp_num);
		VERBS_INFO(FI_LOG_EP_CTRL, "EP %p, Remote TGT QP Num %d\n", ep,
			  ep->ini_conn->tgt_qpn);
	}
	if (ep->tgt_ibv_qp)
		VERBS_INFO(FI_LOG_EP_CTRL, "EP %p, TGT QP Num %d\n",
			  ep, ep->tgt_ibv_qp->qp_num);
}

/* Caller must hold eq:lock */
void fi_ibv_free_xrc_conn_setup(struct fi_ibv_xrc_ep *ep, int disconnect)
{
	assert(ep->conn_setup);

	/* If a disconnect is requested then the XRC bidirectional connection
	 * has completed and a disconnect sequence is started (the XRC INI QP
	 * side disconnect is initiated when the remote target disconnect is
	 * received). */
	if (disconnect) {
		assert(ep->tgt_id);
		assert(!ep->tgt_id->qp);

		if (ep->conn_setup->tgt_connected) {
			if (ep->tgt_id->ps == RDMA_PS_UDP) {
				rdma_destroy_id(ep->tgt_id);
				ep->tgt_id = NULL;
			} else {
				rdma_disconnect(ep->tgt_id);
			}
			ep->conn_setup->tgt_connected = 0;
		}

		if (ep->base_ep.id->ps == RDMA_PS_UDP) {
			rdma_destroy_id(ep->base_ep.id);
			ep->base_ep.id = NULL;
		}
	}

	fi_ibv_eq_clear_xrc_conn_tag(ep);
	if (!disconnect) {
		free(ep->conn_setup);
		ep->conn_setup = NULL;
	}
}

/* Caller must hold the eq:lock */
int fi_ibv_connect_xrc(struct fi_ibv_xrc_ep *ep, struct sockaddr *addr,
		       int reciprocal, void *param, size_t paramlen)
{
	int ret;

	assert(!ep->base_ep.id && !ep->base_ep.ibv_qp && !ep->ini_conn);

	ret = fi_ibv_get_shared_ini_conn(ep, &ep->ini_conn);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Get of shared XRC INI connection failed %d\n", ret);
		if (!reciprocal) {
			free(ep->conn_setup);
			ep->conn_setup = NULL;
		}
		return ret;
	}

	fi_ibv_eq_set_xrc_conn_tag(ep);
	fi_ibv_add_pending_ini_conn(ep, reciprocal, param, paramlen);
	fi_ibv_sched_ini_conn(ep->ini_conn);

	return FI_SUCCESS;
}

/* Caller must hold the eq:lock */
void fi_ibv_ep_ini_conn_done(struct fi_ibv_xrc_ep *ep, uint32_t tgt_qpn)
{
	assert(ep->base_ep.id && ep->ini_conn);

	assert(ep->ini_conn->state == FI_IBV_INI_QP_CONNECTING ||
	       ep->ini_conn->state == FI_IBV_INI_QP_CONNECTED);

	/* If this was a physical INI/TGT QP connection, remove the QP
	 * from control of the RDMA CM. We don't want the shared INI QP
	 * to be destroyed if this endpoint closes. */
	if (ep->base_ep.id == ep->ini_conn->phys_conn_id) {
		ep->ini_conn->phys_conn_id = NULL;
		ep->ini_conn->state = FI_IBV_INI_QP_CONNECTED;
		ep->ini_conn->tgt_qpn = tgt_qpn;
		ep->base_ep.id->qp = NULL;
		VERBS_DBG(FI_LOG_EP_CTRL,
			  "Set INI Conn QP %d remote TGT QP %d\n",
			  ep->ini_conn->ini_qp->qp_num,
			  ep->ini_conn->tgt_qpn);
	}

	ep->conn_setup->ini_connected = 1;
	fi_ibv_log_ep_conn(ep, "INI Connection Done");
	fi_ibv_sched_ini_conn(ep->ini_conn);
}

/* Caller must hold the eq:lock */
void fi_ibv_ep_ini_conn_rejected(struct fi_ibv_xrc_ep *ep)
{
	assert(ep->base_ep.id && ep->ini_conn);

	fi_ibv_log_ep_conn(ep, "INI Connection Rejected");
	fi_ibv_put_shared_ini_conn(ep);
	ep->conn_state = FI_IBV_XRC_ERROR;
}

void fi_ibv_ep_tgt_conn_done(struct fi_ibv_xrc_ep *ep)
{
	fi_ibv_log_ep_conn(ep, "TGT Connection Done\n");

	if (ep->tgt_id->qp) {
		assert(ep->tgt_ibv_qp == ep->tgt_id->qp);
		ep->tgt_id->qp = NULL;
	}
	ep->conn_setup->tgt_connected = 1;
}

/* Caller must hold the eq:lock */
int fi_ibv_accept_xrc(struct fi_ibv_xrc_ep *ep, int reciprocal,
		      void *param, size_t paramlen)
{
	struct sockaddr *addr;
	struct fi_ibv_connreq *connreq;
	struct rdma_conn_param conn_param = { 0 };
	struct fi_ibv_xrc_cm_data *cm_data = param;
	struct fi_ibv_xrc_cm_data connect_cm_data;
	int ret;

	addr = rdma_get_local_addr(ep->tgt_id);
	if (addr)
		ofi_straddr_dbg(&fi_ibv_prov, FI_LOG_CORE, "src_addr", addr);

	addr = rdma_get_peer_addr(ep->tgt_id);
	if (addr)
		ofi_straddr_dbg(&fi_ibv_prov, FI_LOG_CORE, "dest_addr", addr);

	connreq = container_of(ep->base_ep.info->handle,
			       struct fi_ibv_connreq, handle);
	ret = fi_ibv_ep_create_tgt_qp(ep, connreq->xrc.tgt_qpn);
	if (ret)
		return ret;

	ep->peer_srqn = connreq->xrc.peer_srqn;
	fi_ibv_set_xrc_cm_data(cm_data, connreq->xrc.is_reciprocal,
			       connreq->xrc.conn_tag, connreq->xrc.port,
			       0, ep->srqn);
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

	ep->conn_setup->remote_conn_tag = connreq->xrc.conn_tag;

	assert(ep->conn_state == FI_IBV_XRC_UNCONNECTED ||
	       ep->conn_state == FI_IBV_XRC_ORIG_CONNECTED);
	fi_ibv_next_xrc_conn_state(ep);

	ret = rdma_accept(ep->tgt_id, &conn_param);
	if (OFI_UNLIKELY(ret)) {
		ret = -errno;
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "XRC TGT, rdma_accept error %d\n", ret);
		fi_ibv_prev_xrc_conn_state(ep);
	} else
		free(connreq);

	/* The passive side of the initial shared connection using
	 * SIDR is complete, initiate reciprocal connection */
	if (ep->tgt_id->ps == RDMA_PS_UDP && !reciprocal) {
		fi_ibv_next_xrc_conn_state(ep);
		fi_ibv_ep_tgt_conn_done(ep);
		ret = fi_ibv_connect_xrc(ep, NULL, FI_IBV_RECIP_CONN,
					 &connect_cm_data,
					 sizeof(connect_cm_data));
		if (ret) {
			VERBS_WARN(FI_LOG_EP_CTRL,
				   "XRC reciprocal connect error %d\n", ret);
			fi_ibv_prev_xrc_conn_state(ep);
			ep->tgt_id->qp = NULL;
		}
	}

	return ret;
}

int fi_ibv_process_xrc_connreq(struct fi_ibv_ep *ep,
			       struct fi_ibv_connreq *connreq)
{
	struct fi_ibv_xrc_ep *xrc_ep = container_of(ep, struct fi_ibv_xrc_ep,
						    base_ep);

	assert(ep->info->src_addr);
	assert(ep->info->dest_addr);

	xrc_ep->conn_setup = calloc(1, sizeof(*xrc_ep->conn_setup));
	if (!xrc_ep->conn_setup) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			  "Unable to allocate connection setup memory\n");
		return -FI_ENOMEM;
	}
	xrc_ep->conn_setup->conn_tag = VERBS_CONN_TAG_INVALID;

	/* This endpoint was created on the passive side of a connection
	 * request. The reciprocal connection request will go back to the
	 * passive port indicated by the active side */
	ofi_addr_set_port(ep->info->src_addr, 0);
	ofi_addr_set_port(ep->info->dest_addr, connreq->xrc.port);
	xrc_ep->tgt_id = connreq->id;
	xrc_ep->tgt_id->context = &ep->util_ep.ep_fid.fid;

	return FI_SUCCESS;
}
