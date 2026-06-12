#include "verbs_ofi.h"

#define VRB_UDCM_RETRY_MAX		8
#define VRB_UDCM_RETRY_INTERVAL_NS	(2 * 1000000000ULL)
#define VRB_UDCM_RETRY_JITTER_PCT	25

#define VRB_UDCM_QKEY			0x11111112
#define VRB_UDCM_GRH_SIZE		40
#define VRB_UDCM_VERSION		2
#define VRB_UDCM_PACKET_LIFETIME	16
#define VRB_UDCM_DEFAULT_MAX_RD		16
#define VRB_UDCM_DEFAULT_TIMEOUT	17

enum vrb_udcm_msg_type {
	VRB_UDCM_MSG_CONN_REQ = 1,
	VRB_UDCM_MSG_CONN_REP = 2,
	VRB_UDCM_MSG_CONN_RTU = 3,
	VRB_UDCM_MSG_CONN_REJ = 4,
	VRB_UDCM_MSG_DISC_REQ = 5,
};

struct vrb_udcm_msg_hdr {
	uint8_t		version;
	uint8_t		msg_type;
	uint16_t	reserved;
	uint32_t	conn_id;
} __attribute__((packed));

struct vrb_udcm_conn_req {
	struct vrb_udcm_msg_hdr		hdr;
	struct ofi_addr_ib_ud		src_name;
	uint32_t			src_rc_qpn;
	uint8_t				src_addr_len;
	union ofi_sock_ip		src_addr;
	uint16_t			priv_data_len;
	uint8_t				priv_data[VERBS_CM_DATA_SIZE];
} __attribute__((packed));

struct vrb_udcm_conn_rep {
	struct vrb_udcm_msg_hdr		hdr;
	struct ofi_addr_ib_ud		dst_name;
	uint32_t			dst_rc_qpn;
	uint32_t			resp_conn_id;
	uint16_t			priv_data_len;
	uint8_t				priv_data[VERBS_CM_DATA_SIZE];
} __attribute__((packed));

struct vrb_udcm_conn_rtu {
	struct vrb_udcm_msg_hdr		hdr;
} __attribute__((packed));

struct vrb_udcm_conn_rej {
	struct vrb_udcm_msg_hdr		hdr;
	uint16_t			priv_data_len;
	uint8_t				priv_data[VERBS_CM_DATA_SIZE];
} __attribute__((packed));

struct vrb_udcm_disc_req {
	struct vrb_udcm_msg_hdr		hdr;
} __attribute__((packed));

static const char *vrb_udcm_state_str(enum vrb_udcm_state state)
{
	switch (state) {
	case VRB_UDCM_IDLE:
		return "IDLE";
	case VRB_UDCM_REQ_SENT:
		return "REQ_SENT";
	case VRB_UDCM_REQ_RCVD:
		return "REQ_RCVD";
	case VRB_UDCM_REP_SENT:
		return "REP_SENT";
	case VRB_UDCM_RTU_SENT:
		return "RTU_SENT";
	case VRB_UDCM_CONNECTED:
		return "CONNECTED";
	case VRB_UDCM_REJECTING:
		return "REJECTING";
	case VRB_UDCM_DISCONNECTED:
		return "DISCONNECTED";
	default:
		return "UNKNOWN";
	}
}

static void vrb_udcm_trace_state_change(struct vrb_udcm_ep_ctx *ep_ctx,
					 enum vrb_udcm_state new_state,
					 const char *reason)
{
	enum vrb_udcm_state old_state = ep_ctx->state;

	if (old_state == new_state)
		return;

	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: state %s -> %s conn_id=%u peer_conn_id=%u reason=%s\n",
		 vrb_udcm_state_str(old_state), vrb_udcm_state_str(new_state),
		 ep_ctx->conn_id, ep_ctx->peer_conn_id, reason);
}

static inline uint64_t
vrb_udcm_jittered_deadline(struct vrb_udcm_domain_ctx *ctx, uint64_t now)
{
	uint32_t r = ofi_xorshift_random_r(&ctx->retry_seed);
	uint64_t jitter_range = VRB_UDCM_RETRY_INTERVAL_NS *
				VRB_UDCM_RETRY_JITTER_PCT / 100;
	int64_t jitter = (int64_t)(r % (2 * jitter_range + 1)) -
			 (int64_t)jitter_range;

	return now + VRB_UDCM_RETRY_INTERVAL_NS + jitter;
}

static int vrb_udcm_ud_qp_to_rts(struct ibv_qp *qp, uint8_t port_num,
				  uint16_t pkey_index)
{
	struct ibv_qp_attr attr = {
		.qp_state	= IBV_QPS_INIT,
		.pkey_index	= pkey_index,
		.port_num	= port_num,
		.qkey		= VRB_UDCM_QKEY,
	};
	int ret;

	ret = ibv_modify_qp(qp, &attr,
			    IBV_QP_STATE | IBV_QP_PKEY_INDEX |
			    IBV_QP_PORT | IBV_QP_QKEY);
	if (ret)
		return -errno;

	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTR;
	ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
	if (ret)
		return -errno;

	memset(&attr, 0, sizeof(attr));
	attr.qp_state	= IBV_QPS_RTS;
	attr.sq_psn	= 0;
	ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
	if (ret)
		return -errno;

	return 0;
}

static int vrb_udcm_populate_local_name(struct vrb_udcm_ib_ctx *ib)
{
	struct ibv_port_attr port_attr;
	union ibv_gid gid;
	uint16_t pkey;
	int gid_idx;

	gid_idx = vrb_resolve_gid_idx_for_device(ib->verbs);
	if (gid_idx < 0)
		return gid_idx;
	ib->gid_idx = gid_idx;

	if (ibv_query_gid(ib->verbs, ib->port_num, ib->gid_idx, &gid))
		return -errno;
	if (ibv_query_pkey(ib->verbs, ib->port_num, ib->pkey_index, &pkey))
		return -errno;
	if (ibv_query_port(ib->verbs, ib->port_num, &port_attr))
		return -errno;

	ib->local_name.gid	= *(union ofi_ib_gid *)&gid;
	ib->local_name.qpn	= ib->ud_qp->qp_num;
	ib->local_name.lid	= port_attr.lid;
	ib->local_name.pkey	= pkey;
	ib->local_name.sl	= port_attr.sm_sl;

	return 0;
}

static int vrb_udcm_post_recvs(struct vrb_udcm_ib_ctx *ib)
{
	struct ibv_recv_wr wr, *bad_wr;
	struct ibv_sge sge;
	int i, ret;

	for (i = 0; i < VRB_UDCM_RECV_WR; i++) {
		sge.addr	= (uintptr_t)(ib->recv_bufs + i * VRB_UDCM_BUF_SIZE);
		sge.length	= VRB_UDCM_BUF_SIZE;
		sge.lkey	= ib->recv_mr->lkey;

		wr.wr_id	= (uintptr_t)i;
		wr.next		= NULL;
		wr.sg_list	= &sge;
		wr.num_sge	= 1;

		ret = ibv_post_recv(ib->ud_qp, &wr, &bad_wr);
		if (ret)
			return -errno;
	}
	return 0;
}

static void vrb_udcm_repost_recv(struct vrb_udcm_ib_ctx *ib, uint64_t idx)
{
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge sge = {
		.addr	= (uintptr_t)(ib->recv_bufs + idx * VRB_UDCM_BUF_SIZE),
		.length	= VRB_UDCM_BUF_SIZE,
		.lkey	= ib->recv_mr->lkey,
	};

	wr.wr_id	= idx;
	wr.next		= NULL;
	wr.sg_list	= &sge;
	wr.num_sge	= 1;
	ibv_post_recv(ib->ud_qp, &wr, &bad);
}

static struct ibv_device *vrb_udcm_find_ib_device(const char *name)
{
	struct ibv_device **list, *dev = NULL;
	int i, n;

	list = ibv_get_device_list(&n);
	if (!list)
		return NULL;
	for (i = 0; i < n; i++) {
		if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
			dev = list[i];
			break;
		}
	}
	ibv_free_device_list(list);
	return dev;
}

static struct vrb_udcm_ib_ctx *
vrb_udcm_ib_ctx_create(struct ibv_device *dev)
{
	struct vrb_udcm_ib_ctx *ib;
	struct ibv_device_attr dev_attr;
	struct ibv_port_attr port_attr;
	uint16_t pkey;
	uint8_t port_num;
	int i, ret;
	struct ibv_qp_init_attr qp_attr = {
		.qp_type = IBV_QPT_UD,
		.cap = {
			.max_send_wr  = VRB_UDCM_SEND_WR,
			.max_recv_wr  = VRB_UDCM_RECV_WR,
			.max_send_sge = 1,
			.max_recv_sge = 1,
		},
	};

	ib = calloc(1, sizeof(*ib));
	if (!ib)
		return NULL;

	ofi_spin_init(&ib->send_lock);
	ofi_atomic_initialize32(&ib->ref, 1);

	ib->verbs = ibv_open_device(dev);
	if (!ib->verbs)
		goto err;

	if (ibv_query_device(ib->verbs, &dev_attr))
		goto err;
	for (port_num = 1; port_num <= dev_attr.phys_port_cnt; port_num++) {
		if (ibv_query_port(ib->verbs, port_num, &port_attr) == 0 &&
		    port_attr.state == IBV_PORT_ACTIVE)
			break;
	}
	if (port_num > dev_attr.phys_port_cnt)
		goto err;
	ib->port_num = port_num;

	ib->pkey_index = 0;
	for (i = 0; i < port_attr.pkey_tbl_len; i++) {
		if (ibv_query_pkey(ib->verbs, port_num, i, &pkey))
			break;
		if (ntohs(pkey) == 0xFFFF) {
			ib->pkey_index = i;
			break;
		}
	}

	ib->pd = ibv_alloc_pd(ib->verbs);
	if (!ib->pd)
		goto err;

	ib->comp_channel = ibv_create_comp_channel(ib->verbs);
	if (!ib->comp_channel)
		goto err;
	fi_fd_nonblock(ib->comp_channel->fd);

	ib->ud_recv_cq = ibv_create_cq(ib->verbs, VRB_UDCM_RECV_WR,
					NULL, ib->comp_channel, 0);
	if (!ib->ud_recv_cq)
		goto err;
	ibv_req_notify_cq(ib->ud_recv_cq, 0);

	ib->ud_send_cq = ibv_create_cq(ib->verbs, VRB_UDCM_SEND_WR,
					NULL, NULL, 0);
	if (!ib->ud_send_cq)
		goto err;

	qp_attr.send_cq = ib->ud_send_cq;
	qp_attr.recv_cq = ib->ud_recv_cq;
	ib->ud_qp = ibv_create_qp(ib->pd, &qp_attr);
	if (!ib->ud_qp)
		goto err;

	ret = vrb_udcm_ud_qp_to_rts(ib->ud_qp, ib->port_num, ib->pkey_index);
	if (ret)
		goto err;

	ret = vrb_udcm_populate_local_name(ib);
	if (ret)
		goto err;

	ib->recv_bufs = malloc(VRB_UDCM_RECV_WR * VRB_UDCM_BUF_SIZE);
	if (!ib->recv_bufs)
		goto err;

	ib->recv_mr = ibv_reg_mr(ib->pd, ib->recv_bufs,
				 VRB_UDCM_RECV_WR * VRB_UDCM_BUF_SIZE,
				 IBV_ACCESS_LOCAL_WRITE);
	if (!ib->recv_mr)
		goto err;

	ret = vrb_udcm_post_recvs(ib);
	if (ret)
		goto err;

	ib->send_buf = malloc(VRB_UDCM_BUF_SIZE);
	if (!ib->send_buf)
		goto err;

	ib->send_mr = ibv_reg_mr(ib->pd, ib->send_buf, VRB_UDCM_BUF_SIZE, 0);
	if (!ib->send_mr)
		goto err;

	return ib;

err:
	if (ib->send_mr)
		ibv_dereg_mr(ib->send_mr);
	free(ib->send_buf);
	if (ib->recv_mr)
		ibv_dereg_mr(ib->recv_mr);
	free(ib->recv_bufs);
	if (ib->ud_qp)
		ibv_destroy_qp(ib->ud_qp);
	if (ib->ud_send_cq)
		ibv_destroy_cq(ib->ud_send_cq);
	if (ib->ud_recv_cq)
		ibv_destroy_cq(ib->ud_recv_cq);
	if (ib->comp_channel)
		ibv_destroy_comp_channel(ib->comp_channel);
	if (ib->pd)
		ibv_dealloc_pd(ib->pd);
	if (ib->verbs)
		ibv_close_device(ib->verbs);
	ofi_spin_destroy(&ib->send_lock);
	free(ib);
	return NULL;
}

static void vrb_udcm_ib_ctx_ref(struct vrb_udcm_ib_ctx *ib)
{
	ofi_atomic_inc32(&ib->ref);
}

static void vrb_udcm_ib_ctx_put(struct vrb_udcm_ib_ctx *ib)
{
	if (!ib || ofi_atomic_dec32(&ib->ref))
		return;

	if (ib->eq) {
		ofi_epoll_del(ib->eq->epollfd, ib->comp_channel->fd);
		ib->eq = NULL;
	}

	ibv_dereg_mr(ib->send_mr);
	free(ib->send_buf);
	ibv_dereg_mr(ib->recv_mr);
	free(ib->recv_bufs);
	ibv_destroy_qp(ib->ud_qp);
	ibv_destroy_cq(ib->ud_send_cq);
	ibv_destroy_cq(ib->ud_recv_cq);
	ibv_destroy_comp_channel(ib->comp_channel);
	ibv_dealloc_pd(ib->pd);
	ibv_close_device(ib->verbs);
	ofi_spin_destroy(&ib->send_lock);
	free(ib);
}

static int vrb_udcm_send_msg(struct vrb_udcm_ib_ctx *ib,
			     const struct ofi_addr_ib_ud *dst,
			     const void *msg, size_t len)
{
	struct ibv_send_wr wr = {0}, *bad_wr;
	struct ibv_sge sge;
	struct ibv_ah *ah;
	struct ibv_wc wc;
	int ret = 0, n;
	struct ibv_ah_attr ah_attr = {
		.dlid		= dst->lid,
		.sl		= dst->sl,
		.port_num	= ib->port_num,
		.is_global	= 1,
		.grh = {
			.dgid		= *(union ibv_gid *)&dst->gid,
			.sgid_index	= ib->gid_idx,
			.hop_limit	= 64,
		},
	};

	assert(len <= VRB_UDCM_BUF_SIZE);

	ofi_spin_lock(&ib->send_lock);
	ah = ibv_create_ah(ib->pd, &ah_attr);
	if (!ah) {
		ofi_spin_unlock(&ib->send_lock);
		return -errno;
	}

	memcpy(ib->send_buf, msg, len);
	sge.addr	= (uintptr_t)ib->send_buf;
	sge.length	= len;
	sge.lkey	= ib->send_mr->lkey;

	wr.opcode		= IBV_WR_SEND;
	wr.send_flags		= IBV_SEND_SIGNALED;
	wr.sg_list		= &sge;
	wr.num_sge		= 1;
	wr.wr.ud.ah		= ah;
	wr.wr.ud.remote_qpn	= dst->qpn;
	wr.wr.ud.remote_qkey	= VRB_UDCM_QKEY;

	ret = ibv_post_send(ib->ud_qp, &wr, &bad_wr);
	if (ret) {
		ret = -errno;
		goto out;
	}

	do {
		n = ibv_poll_cq(ib->ud_send_cq, 1, &wc);
		if (n < 0) {
			ret = -errno;
			goto out;
		}
	} while (!n);

	if (wc.status != IBV_WC_SUCCESS)
		ret = -FI_EIO;

out:
	ibv_destroy_ah(ah);
	ofi_spin_unlock(&ib->send_lock);
	return ret;
}

static struct vrb_udcm_ep_ctx *
vrb_udcm_ep_lookup(struct vrb_udcm_domain_ctx *ctx, uint32_t conn_id)
{
	struct vrb_udcm_ep_ctx *ep_ctx;

	dlist_foreach_container(&ctx->ep_list, struct vrb_udcm_ep_ctx,
				ep_ctx, list_entry) {
		if (ep_ctx->conn_id == conn_id)
			return ep_ctx;
	}
	return NULL;
}

static struct vrb_udcm_ep_ctx *
vrb_udcm_ep_lookup_by_peer(struct vrb_udcm_domain_ctx *ctx,
			   uint32_t peer_conn_id, uint32_t peer_qpn)
{
	struct vrb_udcm_ep_ctx *ep_ctx;

	dlist_foreach_container(&ctx->ep_list, struct vrb_udcm_ep_ctx,
				ep_ctx, list_entry) {
		if (ep_ctx->peer_conn_id == peer_conn_id &&
		    ep_ctx->peer_name.qpn == peer_qpn)
			return ep_ctx;
	}
	return NULL;
}

static int vrb_udcm_rc_qp_to_init(struct ibv_qp *qp, uint8_t port_num,
				   uint16_t pkey_index)
{
	struct ibv_qp_attr attr = {
		.qp_state	  = IBV_QPS_INIT,
		.pkey_index	  = pkey_index,
		.port_num	  = port_num,
		.qp_access_flags  = IBV_ACCESS_REMOTE_WRITE |
				    IBV_ACCESS_REMOTE_READ  |
				    IBV_ACCESS_REMOTE_ATOMIC,
	};
	return ibv_modify_qp(qp, &attr,
			     IBV_QP_STATE | IBV_QP_PKEY_INDEX |
			     IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS) ? -errno : 0;
}

static int vrb_udcm_rc_qp_to_rtr(struct ibv_qp *qp, uint8_t port_num,
				  const struct ofi_addr_ib_ud *peer,
				  uint32_t peer_rc_qpn)
{
	struct ibv_qp_attr attr;
	struct ibv_port_attr port_attr;
	struct ibv_device_attr dev_attr;
	enum ibv_mtu mtu = IBV_MTU_1024;
	uint8_t max_rd = 16;
	int gid_idx;

	gid_idx = vrb_resolve_gid_idx_for_device(qp->context);
	if (gid_idx < 0)
		return gid_idx;

	if (ibv_query_port(qp->context, port_num, &port_attr) == 0)
		mtu = port_attr.active_mtu;
	if (ibv_query_device(qp->context, &dev_attr) == 0 &&
	    dev_attr.max_qp_rd_atom > 0)
		max_rd = dev_attr.max_qp_rd_atom;

	attr = (struct ibv_qp_attr) {
		.qp_state	    = IBV_QPS_RTR,
		.path_mtu	    = mtu,
		.dest_qp_num	    = peer_rc_qpn,
		.rq_psn		    = 0,
		.max_dest_rd_atomic = max_rd,
		.min_rnr_timer	    = 12,
		.ah_attr = {
			.dlid		= peer->lid,
			.sl		= peer->sl,
			.port_num	= port_num,
			.is_global	= 1,
			.grh = {
				.dgid		= *(union ibv_gid*)&peer->gid,
				.sgid_index	= gid_idx,
				.hop_limit	= 64,
			},
		},
	};
	return ibv_modify_qp(qp, &attr,
			     IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
			     IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
			     IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) ?
	       -errno : 0;
}

static uint8_t vrb_udcm_ack_timeout(uint8_t ca_ack_delay,
				    uint8_t packet_life_time)
{
	int t = packet_life_time + 1;

	if (t >= ca_ack_delay)
		t += (ca_ack_delay >= (t - 1));
	else
		t = ca_ack_delay + (t >= (ca_ack_delay - 1));

	return MIN(31, t);
}

static int vrb_udcm_rc_qp_to_rts(struct ibv_qp *qp, uint8_t port_num)
{
	struct ibv_qp_attr attr;
	struct ibv_device_attr dev_attr;
	uint8_t max_rd = VRB_UDCM_DEFAULT_MAX_RD;
	uint8_t timeout = VRB_UDCM_DEFAULT_TIMEOUT;

	if (ibv_query_device(qp->context, &dev_attr) == 0) {
		if (dev_attr.max_qp_rd_atom > 0)
			max_rd = dev_attr.max_qp_rd_atom;

		timeout = vrb_udcm_ack_timeout(dev_attr.local_ca_ack_delay,
					       VRB_UDCM_PACKET_LIFETIME);
	}

	attr = (struct ibv_qp_attr) {
		.qp_state      = IBV_QPS_RTS,
		.sq_psn        = 0,
		.timeout       = timeout,
		.retry_cnt     = 15,
		.rnr_retry     = 7,
		.max_rd_atomic = max_rd,
	};
	return ibv_modify_qp(qp, &attr,
			     IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
			     IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
			     IBV_QP_MAX_QP_RD_ATOMIC) ? -errno : 0;
}

static void vrb_udcm_flush_pending_rtu(struct vrb_udcm_domain_ctx *ctx)
{
	struct vrb_udcm_ep_ctx *ep_ctx;
	struct vrb_udcm_conn_rtu rtu;
	struct ofi_addr_ib_ud dst;
	bool found;

	if (!ctx->ib)
		return;

	do {
		found = false;
		ofi_spin_lock(&ctx->lock);
		dlist_foreach_container(&ctx->ep_list, struct vrb_udcm_ep_ctx,
					ep_ctx, list_entry) {
			if (ep_ctx->pending_rtu) {
				ep_ctx->pending_rtu = false;
				found = true;
				memset(&rtu, 0, sizeof(rtu));
				rtu.hdr.version  = VRB_UDCM_VERSION;
				rtu.hdr.msg_type = VRB_UDCM_MSG_CONN_RTU;
				rtu.hdr.conn_id  = htonl(ep_ctx->peer_conn_id);
				dst = ep_ctx->peer_name;
				break;
			}
		}
		ofi_spin_unlock(&ctx->lock);
		if (found) {
			VRB_INFO(FI_LOG_EP_CTRL,
				 "udcm: sending RTU to QPN %u LID %u conn_id=%u\n",
				 dst.qpn, dst.lid, ntohl(rtu.hdr.conn_id));
			vrb_udcm_send_msg(ctx->ib, &dst, &rtu, sizeof(rtu));
		}
	} while (found);
}

static void vrb_udcm_post_connreq(struct vrb_udcm_conn_req *req,
				   struct vrb_udcm_msg_hdr *hdr,
				   struct vrb_pep *pep,
				   struct vrb_domain *domain,
				   struct vrb_eq *eq)
{
	uint8_t evbuf[sizeof(struct fi_eq_cm_entry) + VERBS_CM_DATA_SIZE];
	struct fi_eq_cm_entry *entry = (struct fi_eq_cm_entry *)evbuf;
	struct vrb_connreq *connreq;
	size_t datalen;

	connreq = calloc(1, sizeof(*connreq));
	if (!connreq)
		return;

	connreq->handle.fclass    = FI_CLASS_CONNREQ;
	connreq->udcm_peer_name   = req->src_name;
	connreq->udcm_peer_rc_qpn = ntohl(req->src_rc_qpn);
	connreq->udcm_conn_id     = ntohl(hdr->conn_id);
	connreq->udcm_domain      = domain;
	if (req->src_addr_len > 0 &&
	    req->src_addr_len <= sizeof(connreq->udcm_peer_addr)) {
		memcpy(&connreq->udcm_peer_addr, &req->src_addr,
		       req->src_addr_len);
		connreq->udcm_peer_addr_len = req->src_addr_len;
	}

	memset(evbuf, 0, sizeof(evbuf));
	entry->info = fi_dupinfo(pep->info);
	if (!entry->info) {
		free(connreq);
		return;
	}
	entry->info->handle = &connreq->handle;
	free(entry->info->dest_addr);
	entry->info->dest_addr = NULL;
	entry->info->dest_addrlen = 0;
	if (req->src_addr_len > 0) {
		entry->info->dest_addr = malloc(req->src_addr_len);
		if (entry->info->dest_addr) {
			memcpy(entry->info->dest_addr, &req->src_addr,
			       req->src_addr_len);
			entry->info->dest_addrlen = req->src_addr_len;
		}
	}

	entry->fid = &pep->pep_fid.fid;
	datalen = MIN(ntohs(req->priv_data_len), VERBS_CM_DATA_SIZE);
	if (datalen)
		memcpy(entry->data, req->priv_data, datalen);

	vrb_eq_write_event(eq, FI_CONNREQ, entry,
			   sizeof(*entry) + datalen);
}

static void vrb_udcm_dispatch_connreq(struct vrb_udcm_domain_ctx *ctx,
				      struct vrb_udcm_msg_hdr *hdr,
				      struct vrb_eq *eq)
{
	struct vrb_udcm_conn_req *req = (struct vrb_udcm_conn_req *)hdr;
	struct vrb_udcm_pep_ctx *pep_ctx;
	struct vrb_pep *pep = NULL, *tmp_pep;
	struct vrb_udcm_ep_ctx *dup;

	assert(ofi_spin_held(&ctx->lock));

	if (!eq)
		return;

	dup = vrb_udcm_ep_lookup_by_peer(ctx, ntohl(hdr->conn_id),
					 req->src_name.qpn);
	if (dup) {
		if (dup->state == VRB_UDCM_REP_SENT) {
			VRB_INFO(FI_LOG_EP_CTRL,
				 "udcm: duplicate CONN_REQ resets retry conn_id=%u "
				 "peer_conn_id=%u peer_qpn=%u\n",
				 dup->conn_id, dup->peer_conn_id, req->src_name.qpn);
			dup->retry_count = 0;
			dup->retry_deadline_ns = ofi_gettime_ns();
		}
		return;
	}

	dlist_foreach_container(&eq->pep_list, struct vrb_pep,
				tmp_pep, eq_entry) {
		pep_ctx = tmp_pep->cm_ctx;
		if (pep_ctx && pep_ctx->listening) {
			pep = tmp_pep;
			break;
		}
	}
	if (!pep)
		return;

	vrb_udcm_post_connreq(req, hdr, pep, ctx->domain, eq);
}

static void vrb_udcm_handle_conn_req(struct vrb_udcm_domain_ctx *dctx,
				     struct vrb_eq *eq,
				     struct vrb_udcm_msg_hdr *hdr)
{
	struct vrb_udcm_conn_req *req = (struct vrb_udcm_conn_req *)hdr;
	struct vrb_udcm_pep_ctx *pep_ctx;
	struct vrb_pep *pep = NULL, *tmp_pep;

	if (!eq)
		return;

	if (dctx) {
		vrb_udcm_dispatch_connreq(dctx, hdr, eq);
		return;
	}

	dlist_foreach_container(&eq->pep_list, struct vrb_pep,
				tmp_pep, eq_entry) {
		pep_ctx = tmp_pep->cm_ctx;
		if (pep_ctx && pep_ctx->listening) {
			pep = tmp_pep;
			break;
		}
	}
	if (pep) {
		VRB_INFO(FI_LOG_EP_CTRL,
			 "udcm: PEP received CONN_REQ from QPN %u LID %u\n",
			 req->src_name.qpn, req->src_name.lid);
		vrb_udcm_post_connreq(req, hdr, pep, NULL, eq);
	}
}

static void vrb_udcm_handle_conn_rep(struct vrb_udcm_ib_ctx *ib,
				     struct vrb_udcm_domain_ctx *dctx,
				     struct vrb_udcm_msg_hdr *hdr)
{
	struct vrb_udcm_conn_rep *rep = (struct vrb_udcm_conn_rep *)hdr;
	struct vrb_udcm_ep_ctx *ep_ctx;
	struct vrb_udcm_conn_rtu rtu;
	uint8_t evbuf[sizeof(struct fi_eq_cm_entry) + VERBS_CM_DATA_SIZE];
	struct fi_eq_cm_entry *entry = (struct fi_eq_cm_entry *)evbuf;
	struct ofi_addr_ib_ud peer_name;
	uint32_t peer_rc_qpn;
	size_t datalen;

	if (!dctx)
		return;

	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: domain recv CONN_REP conn_id=%u\n",
		 ntohl(hdr->conn_id));
	ep_ctx = vrb_udcm_ep_lookup(dctx, ntohl(hdr->conn_id));
	if (!ep_ctx || (ep_ctx->state != VRB_UDCM_REQ_SENT &&
			ep_ctx->state != VRB_UDCM_CONNECTED)) {
		VRB_WARN(FI_LOG_EP_CTRL,
			 "udcm: CONN_REP bad state (ep_ctx=%p)\n",
			 ep_ctx);
		return;
	}

	if (ep_ctx->state == VRB_UDCM_CONNECTED) {
		VRB_INFO(FI_LOG_EP_CTRL,
			 "udcm: re-sending RTU for conn_id=%u "
			 "(peer lost our RTU)\n",
			 ep_ctx->peer_conn_id);
		memset(&rtu, 0, sizeof(rtu));
		rtu.hdr.version  = VRB_UDCM_VERSION;
		rtu.hdr.msg_type = VRB_UDCM_MSG_CONN_RTU;
		rtu.hdr.conn_id  = htonl(ep_ctx->peer_conn_id);
		vrb_udcm_send_msg(dctx->ib, &ep_ctx->peer_name,
				  &rtu, sizeof(rtu));
		return;
	}

	peer_name = rep->dst_name;
	peer_rc_qpn = ntohl(rep->dst_rc_qpn);

	if (vrb_udcm_rc_qp_to_rtr(ep_ctx->ep->ibv_qp, ib->port_num,
				  &peer_name, peer_rc_qpn) ||
	    vrb_udcm_rc_qp_to_rts(ep_ctx->ep->ibv_qp, ib->port_num))
		return;

	vrb_set_rnr_timer(ep_ctx->ep->ibv_qp);

	ep_ctx->peer_name = peer_name;
	ep_ctx->peer_conn_id = ntohl(rep->resp_conn_id);
	vrb_udcm_trace_state_change(ep_ctx, VRB_UDCM_CONNECTED,
				    "recv CONN_REP");
	ep_ctx->state = VRB_UDCM_CONNECTED;
	ep_ctx->ep->state = VRB_CONNECTED;
	ep_ctx->retry_deadline_ns = 0;
	ep_ctx->pending_rtu = true;

	memset(evbuf, 0, sizeof(evbuf));
	entry->fid = &ep_ctx->ep->util_ep.ep_fid.fid;
	datalen = MIN(ntohs(rep->priv_data_len), VERBS_CM_DATA_SIZE);
	if (datalen)
		memcpy(entry->data, rep->priv_data, datalen);
	if (ep_ctx->ep->eq)
		vrb_eq_write_event(ep_ctx->ep->eq, FI_CONNECTED,
				   entry, sizeof(*entry) + datalen);
	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: FI_CONNECTED (client) conn_id=%u\n",
		 ep_ctx->conn_id);
}

static void vrb_udcm_handle_conn_rtu(struct vrb_udcm_domain_ctx *dctx,
				     struct vrb_udcm_msg_hdr *hdr)
{
	struct vrb_udcm_ep_ctx *ep_ctx;
	uint8_t evbuf[sizeof(struct fi_eq_cm_entry)];
	struct fi_eq_cm_entry *entry = (struct fi_eq_cm_entry *)evbuf;

	if (!dctx)
		return;

	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: domain recv CONN_RTU conn_id=%u\n",
		 ntohl(hdr->conn_id));
	ep_ctx = vrb_udcm_ep_lookup(dctx, ntohl(hdr->conn_id));
	if (!ep_ctx || ep_ctx->state != VRB_UDCM_REP_SENT) {
		VRB_WARN(FI_LOG_EP_CTRL,
			 "udcm: CONN_RTU lookup miss or wrong state (ep_ctx=%p)\n",
			 ep_ctx);
		return;
	}

	vrb_udcm_trace_state_change(ep_ctx, VRB_UDCM_CONNECTED,
				    "recv CONN_RTU");
	ep_ctx->state = VRB_UDCM_CONNECTED;
	ep_ctx->ep->state = VRB_CONNECTED;
	ep_ctx->retry_deadline_ns = 0;

	memset(evbuf, 0, sizeof(evbuf));
	entry->fid = &ep_ctx->ep->util_ep.ep_fid.fid;
	if (ep_ctx->ep->eq)
		vrb_eq_write_event(ep_ctx->ep->eq, FI_CONNECTED,
				   entry, sizeof(*entry));
	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: dispatched FI_CONNECTED (server) conn_id=%u\n",
		 ep_ctx->conn_id);
}

static void vrb_udcm_handle_conn_rej(struct vrb_udcm_domain_ctx *dctx,
				     struct vrb_udcm_msg_hdr *hdr)
{
	struct vrb_udcm_conn_rej *rej = (struct vrb_udcm_conn_rej *)hdr;
	const struct vrb_cm_data_hdr *cm_hdr;
	struct vrb_udcm_ep_ctx *ep_ctx;
	struct vrb_eq *ep_eq;
	const void *payload;
	size_t raw_len, payload_len;

	if (!dctx)
		return;

	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: domain recv CONN_REJ conn_id=%u\n",
		 ntohl(hdr->conn_id));
	ep_ctx = vrb_udcm_ep_lookup(dctx, ntohl(hdr->conn_id));
	if (!ep_ctx || ep_ctx->state != VRB_UDCM_REQ_SENT) {
		VRB_DBG(FI_LOG_EP_CTRL,
			"udcm: CONN_REJ bad state (ep_ctx=%p state=%d)\n",
			ep_ctx, ep_ctx ? ep_ctx->state : -1);
		return;
	}
	ep_eq = ep_ctx->ep->eq;
	if (!ep_eq)
		return;

	raw_len = MIN(ntohs(rej->priv_data_len), VERBS_CM_DATA_SIZE);
	if (raw_len > sizeof(*cm_hdr)) {
		cm_hdr = (const struct vrb_cm_data_hdr *)rej->priv_data;
		payload = cm_hdr->data;
		payload_len = cm_hdr->size;
		if (payload_len > raw_len - sizeof(*cm_hdr))
			payload_len = raw_len - sizeof(*cm_hdr);
	} else {
		payload = rej->priv_data;
		payload_len = raw_len;
	}

	ofi_mutex_lock(&ep_eq->lock);
	ep_eq->err.err = ECONNREFUSED;
	ep_eq->err.prov_errno = 0;
	ep_eq->err.fid = &ep_ctx->ep->util_ep.ep_fid.fid;
	free(ep_eq->err.err_data);
	ep_eq->err.err_data = NULL;
	ep_eq->err.err_data_size = 0;
	if (payload_len) {
		ep_eq->err.err_data = malloc(payload_len);
		if (ep_eq->err.err_data) {
			memcpy(ep_eq->err.err_data, payload, payload_len);
			ep_eq->err.err_data_size = payload_len;
		}
	}
	ofi_mutex_unlock(&ep_eq->lock);

	vrb_udcm_trace_state_change(ep_ctx, VRB_UDCM_IDLE, "recv CONN_REJ");
	ep_ctx->state = VRB_UDCM_IDLE;
	dlist_remove(&ep_ctx->list_entry);
	dlist_init(&ep_ctx->list_entry);
}

static void vrb_udcm_handle_disc_req(struct vrb_udcm_domain_ctx *dctx,
				     struct vrb_udcm_msg_hdr *hdr)
{
	struct vrb_udcm_ep_ctx *ep_ctx;
	uint8_t evbuf[sizeof(struct fi_eq_cm_entry)];
	struct fi_eq_cm_entry *entry = (struct fi_eq_cm_entry *)evbuf;

	if (!dctx)
		return;

	ep_ctx = vrb_udcm_ep_lookup(dctx, ntohl(hdr->conn_id));
	if (!ep_ctx || ep_ctx->state != VRB_UDCM_CONNECTED)
		return;

	vrb_udcm_trace_state_change(ep_ctx, VRB_UDCM_DISCONNECTED,
				    "recv DISC_REQ");
	ep_ctx->state = VRB_UDCM_DISCONNECTED;
	ep_ctx->ep->state = VRB_DISCONNECTED;

	memset(evbuf, 0, sizeof(evbuf));
	entry->fid = &ep_ctx->ep->util_ep.ep_fid.fid;
	if (ep_ctx->ep->eq)
		vrb_eq_write_event(ep_ctx->ep->eq, FI_SHUTDOWN,
				   entry, sizeof(*entry));
}

static void vrb_udcm_dispatch_recv(struct vrb_udcm_ib_ctx *ib,
				   struct vrb_udcm_domain_ctx *dctx,
				   struct vrb_eq *eq,
				   struct vrb_udcm_msg_hdr *hdr)
{
	switch (hdr->msg_type) {
	case VRB_UDCM_MSG_CONN_REQ:
		vrb_udcm_handle_conn_req(dctx, eq, hdr);
		break;
	case VRB_UDCM_MSG_CONN_REP:
		vrb_udcm_handle_conn_rep(ib, dctx, hdr);
		break;
	case VRB_UDCM_MSG_CONN_RTU:
		vrb_udcm_handle_conn_rtu(dctx, hdr);
		break;
	case VRB_UDCM_MSG_CONN_REJ:
		vrb_udcm_handle_conn_rej(dctx, hdr);
		break;
	case VRB_UDCM_MSG_DISC_REQ:
		vrb_udcm_handle_disc_req(dctx, hdr);
		break;
	default:
		FI_WARN_ONCE(&vrb_prov, FI_LOG_EP_CTRL,
			     "unknown udcm msg_type %u\n", hdr->msg_type);
		break;
	}
}

static int vrb_udcm_drain_cq(struct vrb_udcm_ib_ctx *ib,
			       struct vrb_udcm_domain_ctx *dctx,
			       struct vrb_eq *eq)
{
	struct ibv_cq *ev_cq;
	void *ev_ctx;
	struct ibv_wc wc;
	uint8_t *buf;
	struct vrb_udcm_msg_hdr *hdr;
	int ret;

	if (ibv_get_cq_event(ib->comp_channel, &ev_cq, &ev_ctx) == 0) {
		ibv_ack_cq_events(ev_cq, 1);
		ret = ibv_req_notify_cq(ib->ud_recv_cq, 0);
		if (ret)
			return -errno;
	}

	if (dctx)
		ofi_spin_lock(&dctx->lock);
	while (ibv_poll_cq(ib->ud_recv_cq, 1, &wc) > 0) {
		if (wc.status != IBV_WC_SUCCESS)
			goto repost;

		buf = ib->recv_bufs + wc.wr_id * VRB_UDCM_BUF_SIZE;
		hdr = (struct vrb_udcm_msg_hdr *)(buf + VRB_UDCM_GRH_SIZE);
		if (hdr->version != VRB_UDCM_VERSION) {
			FI_WARN_ONCE(&vrb_prov, FI_LOG_EP_CTRL,
				"unknown udcm version %u\n", hdr->version);
			goto repost;
		}

		vrb_udcm_dispatch_recv(ib, dctx, eq, hdr);
repost:
		vrb_udcm_repost_recv(ib, wc.wr_id);
	}
	if (dctx)
		ofi_spin_unlock(&dctx->lock);

	if (dctx)
		vrb_udcm_flush_pending_rtu(dctx);
	return 0;
}

static void vrb_udcm_check_retries(struct vrb_udcm_domain_ctx *ctx,
				   struct vrb_eq *eq)
{
	struct vrb_udcm_ep_ctx *ep_ctx;
	struct dlist_entry *tmp;
	uint64_t now = ofi_gettime_ns();
	enum { NONE, RETRY, TIMEOUT } action;
	enum vrb_udcm_state state;
	uint32_t conn_id, peer_conn_id, rc_qpn;
	size_t priv_data_len;
	uint8_t priv_data[VERBS_CM_DATA_SIZE];
	union ofi_sock_ip src_addr;
	uint8_t src_addr_len;
	struct ofi_addr_ib_ud dst;
	struct vrb_ep *tep;
	struct vrb_udcm_conn_req retry_req = {0};
	struct vrb_udcm_conn_rep retry_rep = {0};

	if (!ctx->ib)
		return;

	do {
		action = NONE;
		tep = NULL;

		ofi_spin_lock(&ctx->lock);
		dlist_foreach_container_safe(&ctx->ep_list,
				struct vrb_udcm_ep_ctx, ep_ctx, list_entry, tmp) {
			if (!ep_ctx->retry_deadline_ns ||
			    now < ep_ctx->retry_deadline_ns)
				continue;

			if (ep_ctx->retry_count >= VRB_UDCM_RETRY_MAX) {
				ep_ctx->retry_deadline_ns = 0;
				state = ep_ctx->state;
				conn_id = ep_ctx->conn_id;
				dst = ep_ctx->peer_name;
				VRB_INFO(FI_LOG_EP_CTRL,
					 "udcm: retry max reached state=%s conn_id=%u "
					 "peer_conn_id=%u retries=%d\n",
					 vrb_udcm_state_str(ep_ctx->state),
					 ep_ctx->conn_id, ep_ctx->peer_conn_id,
					 ep_ctx->retry_count);
				vrb_udcm_trace_state_change(ep_ctx,
							VRB_UDCM_DISCONNECTED,
							"retry_max_timeout");
				ep_ctx->state = VRB_UDCM_DISCONNECTED;
				ep_ctx->ep->state = VRB_DISCONNECTED;
				tep    = ep_ctx->ep;
				action = TIMEOUT;
			} else {
				ep_ctx->retry_count++;
				ep_ctx->retry_deadline_ns =
					vrb_udcm_jittered_deadline(ctx, now);
				VRB_DBG(FI_LOG_EP_CTRL,
					"udcm: retry state=%s conn_id=%u peer_conn_id=%u "
					"retry_count=%d next_deadline_ns=%llu\n",
					vrb_udcm_state_str(ep_ctx->state),
					ep_ctx->conn_id, ep_ctx->peer_conn_id,
					ep_ctx->retry_count,
					(unsigned long long) ep_ctx->retry_deadline_ns);
				state = ep_ctx->state;
				conn_id = ep_ctx->conn_id;
				peer_conn_id = ep_ctx->peer_conn_id;
				rc_qpn = ep_ctx->ep->ibv_qp ?
					 ep_ctx->ep->ibv_qp->qp_num : 0;
				priv_data_len = ep_ctx->priv_data_len;
				dst = ep_ctx->peer_name;
				src_addr = ep_ctx->src_addr;
				src_addr_len  = ep_ctx->src_addr_len;
				if (priv_data_len)
					memcpy(priv_data, ep_ctx->priv_data,
					       priv_data_len);
				action = RETRY;
			}
			break;
		}
		ofi_spin_unlock(&ctx->lock);

		if (action == TIMEOUT) {
			FI_WARN(&vrb_prov, FI_LOG_EP_CTRL,
				"udcm: connect timed out after %d retries\n",
				VRB_UDCM_RETRY_MAX);
			if (eq) {
				ofi_mutex_lock(&eq->lock);
				eq->err.err = ETIMEDOUT;
				eq->err.prov_errno = 0;
				eq->err.fid = &tep->util_ep.ep_fid.fid;
				ofi_mutex_unlock(&eq->lock);
				return;
			}
		} else if (action == RETRY) {
			if (state == VRB_UDCM_REQ_SENT) {
				retry_req.hdr.version = VRB_UDCM_VERSION;
				retry_req.hdr.msg_type = VRB_UDCM_MSG_CONN_REQ;
				retry_req.hdr.conn_id = htonl(conn_id);
				retry_req.src_name = ctx->ib->local_name;
				retry_req.src_rc_qpn = htonl(rc_qpn);
				retry_req.priv_data_len = htons((uint16_t)priv_data_len);
				if (priv_data_len)
					memcpy(retry_req.priv_data, priv_data,
					       priv_data_len);
				if (src_addr_len) {
					memcpy(&retry_req.src_addr, &src_addr,
					       src_addr_len);
					retry_req.src_addr_len = src_addr_len;
				}
				vrb_udcm_send_msg(ctx->ib, &dst, &retry_req,
						  sizeof(retry_req));
			} else if (state == VRB_UDCM_REP_SENT) {
				retry_rep.hdr.version = VRB_UDCM_VERSION;
				retry_rep.hdr.msg_type = VRB_UDCM_MSG_CONN_REP;
				retry_rep.hdr.conn_id = htonl(peer_conn_id);
				retry_rep.resp_conn_id = htonl(conn_id);
				retry_rep.dst_name = ctx->ib->local_name;
				retry_rep.dst_rc_qpn = htonl(rc_qpn);
				retry_rep.priv_data_len = htons((uint16_t)priv_data_len);
				if (priv_data_len)
					memcpy(retry_rep.priv_data, priv_data,
					       priv_data_len);
				vrb_udcm_send_msg(ctx->ib, &dst, &retry_rep,
						  sizeof(retry_rep));
			}
		}
	} while (action != NONE);
}

static ssize_t vrb_udcm_progress(struct vrb_eq *eq, uint32_t *event,
				 void *buf, size_t len)
{
	struct vrb_udcm_eq_ctx *eq_ctx = eq->cm_ctx;
	struct vrb_udcm_domain_ctx *dctx = NULL;
	struct vrb_pep *pep;
	struct vrb_udcm_pep_ctx *pctx;
	int ret = 0;

	if (eq_ctx && eq_ctx->domain && eq_ctx->domain->cm_ctx)
		dctx = eq_ctx->domain->cm_ctx;

	/* Drain domain's CQ first — CONN_REPs arrive here */
	if (dctx && dctx->ib) {
		ret = vrb_udcm_drain_cq(dctx->ib, dctx, eq);
		if (ret)
			return ret;
	}

	dlist_foreach_container(&eq->pep_list, struct vrb_pep, pep, eq_entry) {
		pctx = pep->cm_ctx;
		if (!pctx || !pctx->ib)
			return 0;

		if (!dctx || pctx->ib != dctx->ib) {
			ret = vrb_udcm_drain_cq(pctx->ib, dctx, pep->eq);
			if (ret)
				return ret;
		}
	}

	if (dctx)
		vrb_udcm_check_retries(dctx, eq);

	return 0;
}

static int vrb_udcm_connect(struct vrb_ep *ep, const void *addr,
			    const void *param, size_t paramlen)
{
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_udcm_domain_ctx *ctx = domain->cm_ctx;
	struct vrb_udcm_ep_ctx *ep_ctx = ep->cm_ctx;
	struct vrb_udcm_conn_req req = {0};
	const struct ofi_sockaddr_ib *sib;
	struct ofi_addr_ib_ud *peer_ud;
	struct sockaddr *sa = (struct sockaddr*)addr;
	struct ofi_addr_ib_ud *ud = (struct ofi_addr_ib_ud *)addr;
	char ip_str[INET6_ADDRSTRLEN];
	int svc;
	struct util_ns ns = {0};

	if (!ctx || !ctx->ib || !ep->ibv_qp)
		return -FI_EOPBADSTATE;
	if (paramlen > VERBS_CM_DATA_SIZE)
		return -FI_EINVAL;

	if (ep->info_attr.addr_format == FI_ADDR_IB_UD && ud->qpn != 0) {
		ep_ctx->peer_name = *ud;
		VRB_INFO(FI_LOG_EP_CTRL,
			 "udcm: addr — QPN %u LID %u\n",
			 ep_ctx->peer_name.qpn, ep_ctx->peer_name.lid);
	} else if (ep->info_attr.addr_format == FI_ADDR_IB_UD) {
		if (ud->gid.raw[10] == 0xff && ud->gid.raw[11] == 0xff) {
			if (!inet_ntop(AF_INET, &ud->gid.raw[12],
				       ip_str, sizeof(ip_str)))
				return -errno;
		} else {
			if (!inet_ntop(AF_INET6, ud->gid.raw,
				       ip_str, sizeof(ip_str)))
				return -errno;
		}
		svc = ud->service;

		ns.port = (svc > 1) ? svc : vrb_gl_data.dgram.name_server_port;
		ns.name_len = sizeof(*peer_ud);
		ns.service_len = sizeof(svc);
		ns.service_cmp = vrb_dgram_ns_service_cmp;
		ns.is_service_wildcard = vrb_dgram_ns_is_service_wildcard;

		ofi_ns_init(&ns);
		peer_ud = ofi_ns_resolve_name(&ns, ip_str, &svc);
		if (!peer_ud) {
			VRB_WARN(FI_LOG_EP_CTRL,
				 "udcm: failed to resolve server UD CM address at %s:%d\n",
				 ip_str, svc);
			return -FI_ENODATA;
		}

		ep_ctx->peer_name = *peer_ud;
		free(peer_ud);
	} else {
		ns.name_len = sizeof(*peer_ud);
		ns.service_len = sizeof(svc);
		ns.service_cmp = vrb_dgram_ns_service_cmp;
		ns.is_service_wildcard = vrb_dgram_ns_is_service_wildcard;

		switch (sa->sa_family) {
		case AF_INET:
			if (!inet_ntop(AF_INET,
				       &((const struct sockaddr_in *)sa)->sin_addr,
				       ip_str, sizeof(ip_str)))
				return -errno;
			svc = ntohs(((const struct sockaddr_in *)sa)->sin_port);
			break;
		case AF_INET6:
			if (!inet_ntop(AF_INET6,
				       &((const struct sockaddr_in6 *)sa)->sin6_addr,
				       ip_str, sizeof(ip_str)))
				return -errno;
			svc = ntohs(((const struct sockaddr_in6 *)sa)->sin6_port);
			break;
		case AF_IB:
			sib = (const struct ofi_sockaddr_ib *)sa;
			if (!inet_ntop(AF_INET6, sib->sib_addr,
				       ip_str, sizeof(ip_str)))
				return -errno;
			svc = (int)(ntohll(sib->sib_sid) & 0xFFFF);
			break;
		default:
			VRB_WARN(FI_LOG_EP_CTRL,
				 "udcm: unsupported dest addr family %d\n",
				 sa->sa_family);
			return -FI_ENOSYS;
		}

		ns.port = (svc > 1) ? svc : vrb_gl_data.dgram.name_server_port;

		ofi_ns_init(&ns);
		peer_ud = ofi_ns_resolve_name(&ns, ip_str, &svc);
		if (!peer_ud) {
			VRB_WARN(FI_LOG_EP_CTRL,
				 "udcm: failed to resolve server UD CM address at %s:%d\n",
				 ip_str, svc);
			return -FI_ENODATA;
		}

		ep_ctx->peer_name = *peer_ud;
		free(peer_ud);
	}

	ep_ctx->conn_id = (uint32_t)ofi_atomic_inc32(&ctx->next_conn_id);

	if (param && paramlen) {
		memcpy(ep_ctx->priv_data, param, paramlen);
		ep_ctx->priv_data_len = paramlen;
	}

	req.hdr.version = VRB_UDCM_VERSION;
	req.hdr.msg_type = VRB_UDCM_MSG_CONN_REQ;
	req.hdr.conn_id = htonl(ep_ctx->conn_id);
	req.src_name = ctx->ib->local_name;
	req.src_rc_qpn = htonl(ep->ibv_qp->qp_num);
	req.priv_data_len = htons((uint16_t)paramlen);
	if (paramlen)
		memcpy(req.priv_data, param, paramlen);

	if (ep->info_attr.src_addr && ep->info_attr.src_addrlen &&
	    ep->info_attr.src_addrlen <= sizeof(req.src_addr)) {
		memcpy(&req.src_addr, ep->info_attr.src_addr,
		       ep->info_attr.src_addrlen);
		req.src_addr_len = (uint8_t)ep->info_attr.src_addrlen;
		memcpy(&ep_ctx->src_addr, ep->info_attr.src_addr,
		       ep->info_attr.src_addrlen);
		ep_ctx->src_addr_len = (uint8_t)ep->info_attr.src_addrlen;
	}

	ofi_spin_lock(&ctx->lock);
	vrb_udcm_trace_state_change(ep_ctx, VRB_UDCM_REQ_SENT,
				    "connect() send CONN_REQ");
	ep_ctx->state = VRB_UDCM_REQ_SENT;
	ep_ctx->retry_count = 0;
	ep_ctx->retry_deadline_ns = vrb_udcm_jittered_deadline(ctx, ofi_gettime_ns());
	dlist_insert_tail(&ep_ctx->list_entry, &ctx->ep_list);
	ofi_spin_unlock(&ctx->lock);

	return vrb_udcm_send_msg(ctx->ib, &ep_ctx->peer_name, &req, sizeof(req));
}

static int vrb_udcm_accept(struct vrb_ep *ep, const void *param,
			   size_t paramlen)
{
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_udcm_domain_ctx *ctx = domain->cm_ctx;
	struct vrb_udcm_ep_ctx *ep_ctx = ep->cm_ctx;
	struct vrb_udcm_conn_rep rep = {0};
	struct vrb_connreq *connreq;
	int ret;

	if (!ctx || !ctx->ib || !ep->ibv_qp)
		return -FI_EOPBADSTATE;
	if (paramlen > VERBS_CM_DATA_SIZE)
		return -FI_EINVAL;

	ret = vrb_udcm_rc_qp_to_rts(ep->ibv_qp, ctx->ib->port_num);
	if (ret)
		return ret;

	vrb_set_rnr_timer(ep->ibv_qp);

	if (ep->info_attr.handle &&
	    ((struct fid *)ep->info_attr.handle)->fclass == FI_CLASS_CONNREQ) {
		connreq = container_of(ep->info_attr.handle,
				       struct vrb_connreq, handle);
		free(connreq);
		ep->info_attr.handle = NULL;
	}

	rep.hdr.version = VRB_UDCM_VERSION;
	rep.hdr.msg_type = VRB_UDCM_MSG_CONN_REP;
	rep.hdr.conn_id = htonl(ep_ctx->peer_conn_id);
	rep.resp_conn_id = htonl(ep_ctx->conn_id);
	rep.dst_name = ctx->ib->local_name;
	rep.dst_rc_qpn = htonl(ep->ibv_qp->qp_num);
	rep.priv_data_len = htons((uint16_t)paramlen);
	if (paramlen)
		memcpy(rep.priv_data, param, paramlen);

	vrb_udcm_trace_state_change(ep_ctx, VRB_UDCM_REP_SENT,
				    "accept() send CONN_REP");
	ep_ctx->state = VRB_UDCM_REP_SENT;
	ep_ctx->retry_count = 0;
	ep_ctx->retry_deadline_ns = vrb_udcm_jittered_deadline(ctx, ofi_gettime_ns());
	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: sending CONN_REP to QPN %u LID %u conn_id=%u\n",
		 ep_ctx->peer_name.qpn, ep_ctx->peer_name.lid, ep_ctx->conn_id);
	return vrb_udcm_send_msg(ctx->ib, &ep_ctx->peer_name, &rep, sizeof(rep));
}

static int vrb_udcm_shutdown(struct vrb_ep *ep, uint64_t flags)
{
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_udcm_domain_ctx *ctx = domain->cm_ctx;
	struct vrb_udcm_ep_ctx *ep_ctx = ep->cm_ctx;
	struct vrb_udcm_disc_req disc = {0};

	if (!ctx || !ctx->ib || !ep_ctx ||
	    ep_ctx->state != VRB_UDCM_CONNECTED)
		return 0;

	vrb_udcm_trace_state_change(ep_ctx, VRB_UDCM_DISCONNECTED,
				    "shutdown() send DISC_REQ");
	ep_ctx->state = VRB_UDCM_DISCONNECTED;
	ep->state = VRB_DISCONNECTED;

	disc.hdr.version = VRB_UDCM_VERSION;
	disc.hdr.msg_type = VRB_UDCM_MSG_DISC_REQ;
	disc.hdr.conn_id = htonl(ep_ctx->peer_conn_id);
	vrb_udcm_send_msg(ctx->ib, &ep_ctx->peer_name, &disc, sizeof(disc));
	return 0;
}

static int vrb_udcm_reject(struct vrb_pep *pep, struct vrb_connreq *connreq,
			   const void *param, size_t paramlen)
{
	struct vrb_udcm_domain_ctx *dctx;
	struct vrb_udcm_pep_ctx *pctx;
	struct vrb_udcm_ib_ctx *ib = NULL;
	struct vrb_udcm_conn_rej rej = {0};

	if (connreq->udcm_domain && connreq->udcm_domain->cm_ctx) {
		dctx = connreq->udcm_domain->cm_ctx;
		ib = dctx->ib;
	}
	if (!ib && pep && pep->cm_ctx) {
		pctx = pep->cm_ctx;
		ib = pctx->ib;
	}
	if (!ib) {
		FI_WARN(&vrb_prov, FI_LOG_EP_CTRL,
			"udcm: REJ DROPPED — no ib_ctx available!\n");
		return 0;
	}

	rej.hdr.version = VRB_UDCM_VERSION;
	rej.hdr.msg_type = VRB_UDCM_MSG_CONN_REJ;
	rej.hdr.conn_id = htonl(connreq->udcm_conn_id);
	rej.priv_data_len = htons((uint16_t)paramlen);
	if (paramlen)
		memcpy(rej.priv_data, param, paramlen);

	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: REJ → peer QPN %u conn_id %u\n",
		 connreq->udcm_peer_name.qpn, connreq->udcm_conn_id);

	return vrb_udcm_send_msg(ib, &connreq->udcm_peer_name,
				 &rej, sizeof(rej));
}

static int vrb_udcm_eq_open(struct vrb_eq *eq)
{
	struct vrb_udcm_eq_ctx *ctx;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return -FI_ENOMEM;

	eq->cm_ctx = ctx;
	eq->cm_ops = &vrb_udcm_ops;
	return 0;
}

static int vrb_udcm_eq_close(struct vrb_eq *eq)
{
	struct vrb_udcm_eq_ctx *eq_ctx = eq->cm_ctx;
	struct vrb_udcm_domain_ctx *dctx;
	struct vrb_udcm_ib_ctx *ib = NULL;
	struct vrb_udcm_pep_ctx *pep_ctx;
	struct vrb_pep *pep;

	if (eq_ctx->domain) {
		dctx = eq_ctx->domain->cm_ctx;
		if (dctx && dctx->ib && dctx->ib->eq == eq) {
			ofi_epoll_del(eq->epollfd,
				      dctx->ib->comp_channel->fd);
			dctx->ib->eq = NULL;
		}
	}

	dlist_foreach_container(&eq->pep_list, struct vrb_pep, pep, eq_entry) {
		pep_ctx = pep->cm_ctx;
		if (pep_ctx && pep_ctx->ib && pep_ctx->ib->eq == eq) {
			ib = pep_ctx->ib;
			break;
		}
	}

	if (ib) {
		ofi_epoll_del(eq->epollfd, ib->comp_channel->fd);
		ib->eq = NULL;
	}

	free(eq->cm_ctx);
	eq->cm_ctx = NULL;
	return 0;
}

static int vrb_udcm_domain_get(struct vrb_domain *domain)
{
	struct vrb_udcm_domain_ctx *ctx;

	ofi_genlock_lock(&domain->util_domain.lock);
	ctx = domain->cm_ctx;
	if (ctx) {
		ofi_atomic_inc32(&ctx->ref);
		ofi_genlock_unlock(&domain->util_domain.lock);
		return 0;
	}

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx) {
		ofi_genlock_unlock(&domain->util_domain.lock);
		return -FI_ENOMEM;
	}

	ctx->domain = domain;
	ofi_spin_init(&ctx->lock);
	ofi_atomic_initialize32(&ctx->ref, 1);
	ofi_atomic_initialize32(&ctx->next_conn_id, 1);
	ctx->retry_seed = ofi_generate_seed();
	dlist_init(&ctx->ep_list);

	domain->cm_ctx = ctx;
	ofi_genlock_unlock(&domain->util_domain.lock);
	return 0;
}

static void vrb_udcm_domain_put(struct vrb_domain *domain)
{
	struct vrb_udcm_domain_ctx *ctx = domain->cm_ctx;

	if (!ctx || ofi_atomic_dec32(&ctx->ref))
		return;

	domain->cm_ctx = NULL;

	vrb_udcm_ib_ctx_put(ctx->ib);
	ctx->ib = NULL;

	ofi_spin_destroy(&ctx->lock);
	free(ctx);
}

static int vrb_udcm_domain_init(struct vrb_domain *domain)
{
	struct vrb_fabric *fab = container_of(domain->util_domain.fabric,
					      struct vrb_fabric, util_fabric);
	int ns_port;

	ns_port = vrb_gl_data.msg.udcm_ns_port;
	if (ns_port <= 0)
		ns_port = vrb_gl_data.dgram.name_server_port;

	if (!fab->name_server.is_initialized) {
		fab->name_server.port = ns_port;
		fab->name_server.name_len = sizeof(struct ofi_addr_ib_ud);
		fab->name_server.service_len = sizeof(int);
		fab->name_server.service_cmp = vrb_dgram_ns_service_cmp;
		fab->name_server.is_service_wildcard = vrb_dgram_ns_is_service_wildcard;
		ofi_ns_init(&fab->name_server);
	}
	ofi_ns_start_server(&fab->name_server);
	return 0;
}

static int vrb_udcm_domain_close(struct vrb_domain *domain)
{
	struct vrb_fabric *fab = container_of(domain->util_domain.fabric,
					      struct vrb_fabric, util_fabric);

	if (fab->name_server.is_initialized)
		ofi_ns_stop_server(&fab->name_server);

	vrb_udcm_domain_put(domain);
	return 0;
}

static int vrb_udcm_pep_init(struct vrb_pep *pep)
{
	struct vrb_udcm_pep_ctx *ctx;
	struct ofi_addr_ib_ud *ud;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return -FI_ENOMEM;

	if (pep->info && pep->info->src_addr) {
		if (pep->info->addr_format == FI_ADDR_IB_UD) {
			ud = (struct ofi_addr_ib_ud *)pep->info->src_addr;
			ctx->service = (int)ud->service;
		} else {
			ctx->service = (int)ofi_addr_get_port(
						pep->info->src_addr);
		}
	}

	pep->cm_ctx = ctx;
	return 0;
}

static int vrb_udcm_pep_close(struct vrb_pep *pep)
{
	struct vrb_udcm_pep_ctx *ctx = pep->cm_ctx;
	struct vrb_fabric *fab = pep->fabric;

	if (!ctx)
		return 0;

	if (fab && fab->name_server.is_initialized && ctx->listening)
		ofi_ns_del_local_name(&fab->name_server,
				      &ctx->service, &ctx->ib->local_name);

	vrb_udcm_ib_ctx_put(ctx->ib);
	ctx->ib = NULL;

	free(ctx);
	pep->cm_ctx = NULL;
	return 0;
}

static uint16_t vrb_udcm_alloc_port(const struct sockaddr_in *template)
{
	struct sockaddr_in sin = *template;
	socklen_t len = sizeof(sin);
	int fd;

	sin.sin_port = 0;
	fd = socket(AF_INET, SOCK_DGRAM, 0);
	if (fd < 0)
		return 0;
	if (bind(fd, (struct sockaddr *)&sin, sizeof(sin)) ||
	    getsockname(fd, (struct sockaddr *)&sin, &len)) {
		ofi_close_socket(fd);
		return 0;
	}
	ofi_close_socket(fd);
	return ntohs(sin.sin_port);
}

static int vrb_udcm_listen(struct vrb_pep *pep)
{
	struct vrb_udcm_pep_ctx *ctx = pep->cm_ctx;
	struct vrb_fabric *fab = pep->fabric;
	struct ofi_addr_ib_ud *ud_addr;
	struct sockaddr_in any = {0};
	struct ibv_device *dev;
	uint16_t port;
	int ret;

	if (!ctx)
		return -FI_EOPBADSTATE;

	dev = vrb_udcm_find_ib_device(pep->info->domain_attr->name);
	if (!dev) {
		VRB_WARN(FI_LOG_EP_CTRL, "udcm: IB device '%s' not found\n",
			 pep->info->domain_attr->name);
		return -FI_ENODATA;
	}

	ctx->ib = vrb_udcm_ib_ctx_create(dev);
	if (!ctx->ib)
		return -FI_ENOMEM;

	if (pep->info->src_addr && pep->info->addr_format == FI_ADDR_IB_UD)
		port = ((struct ofi_addr_ib_ud *)pep->info->src_addr)->service;
	else if (pep->info->src_addr)
		port = ofi_addr_get_port(pep->info->src_addr);
	else
		port = 0;
	if (!port) {
		any.sin_family = AF_INET;
		any.sin_addr.s_addr = INADDR_ANY;
		port = vrb_udcm_alloc_port(&any);
		if (!port) {
			VRB_WARN(FI_LOG_EP_CTRL,
				 "udcm: failed to allocate ephemeral port\n");
			ret = -FI_EADDRNOTAVAIL;
			goto err;
		}
	}

	if (pep->info->addr_format == FI_ADDR_IB_UD) {
		ud_addr = calloc(1, sizeof(*ud_addr));
		if (!ud_addr) {
			ret = -FI_ENOMEM;
			goto err;
		}
		*ud_addr = ctx->ib->local_name;
		ud_addr->service = port;

		free(pep->info->src_addr);
		pep->info->src_addr = ud_addr;
		pep->info->src_addrlen = sizeof(*ud_addr);
	} else {
		if (pep->info->src_addr)
			ofi_addr_set_port(pep->info->src_addr, port);
	}

	VRB_INFO(FI_LOG_EP_CTRL,
		 "udcm: PEP listening QPN %u LID %u port %u\n",
		 ctx->ib->local_name.qpn, ctx->ib->local_name.lid, port);

	ctx->service = VERBS_IB_UD_NS_ANY_SERVICE;
	if (fab && fab->name_server.is_initialized) {
		ret = ofi_ns_add_local_name(&fab->name_server,
					    &ctx->service,
					    &ctx->ib->local_name);
		if (ret)
			VRB_INFO(FI_LOG_EP_CTRL,
				 "udcm: NS add_local_name failed (%d)\n",
				 ret);
	}

	ibv_req_notify_cq(ctx->ib->ud_recv_cq, 0);
	if (pep->eq) {
		if (ofi_epoll_add(pep->eq->epollfd, ctx->ib->comp_channel->fd,
				  OFI_EPOLL_IN, NULL)) {
			ret = -errno;
			VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "ofi_epoll_add (PEP)");
			goto err;
		}
		ctx->ib->eq = pep->eq;
	}

	ctx->listening = true;
	return 0;
err:
	vrb_udcm_pep_close(pep);
	return ret;
}

static int vrb_udcm_pep_bind(struct vrb_pep *pep, struct vrb_eq *eq)
{
	if (!eq->cm_ops)
		eq->cm_ops = &vrb_udcm_ops;
	if (!eq->cm_ctx)
		return vrb_udcm_eq_open(eq);
	return 0;
}

static int vrb_udcm_ep_init(struct vrb_ep *ep, struct fi_info *info)
{
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_udcm_domain_ctx *dctx;
	struct vrb_connreq *connreq;
	struct vrb_udcm_ep_ctx *ctx;
	int ret;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return -FI_ENOMEM;
	ctx->ep = ep;
	dlist_init(&ctx->list_entry);
	ep->cm_ctx = ctx;

	if (domain->util_domain.threading == FI_THREAD_SAFE) {
		*ep->util_ep.ep_fid.msg = vrb_msg_ep_msg_ops_ts;
		ep->util_ep.ep_fid.rma = &vrb_msg_ep_rma_ops_ts;
	} else {
		*ep->util_ep.ep_fid.msg = vrb_msg_ep_msg_ops;
		ep->util_ep.ep_fid.rma = &vrb_msg_ep_rma_ops;
	}
	ep->util_ep.ep_fid.cm = &vrb_msg_ep_cm_ops;
	ep->util_ep.ep_fid.atomic = &vrb_msg_ep_atomic_ops;

	ret = vrb_udcm_domain_get(domain);
	if (ret) {
		free(ctx);
		ep->cm_ctx = NULL;
		return ret;
	}

	if (info->handle && info->handle->fclass == FI_CLASS_CONNREQ) {
		connreq = container_of(info->handle, struct vrb_connreq, handle);
		if (connreq->udcm_conn_id || connreq->udcm_peer_rc_qpn) {
			dctx = domain->cm_ctx;

			ctx->peer_name = connreq->udcm_peer_name;
			ctx->peer_rc_qpn = connreq->udcm_peer_rc_qpn;
			ctx->peer_conn_id = connreq->udcm_conn_id;
			ctx->conn_id = (uint32_t)ofi_atomic_inc32(
						&dctx->next_conn_id);
			vrb_udcm_trace_state_change(ctx, VRB_UDCM_REQ_RCVD,
						    "ep_init() from FI_CONNREQ");
			ctx->state = VRB_UDCM_REQ_RCVD;
			ep->state = VRB_REQ_RCVD;
			if (connreq->udcm_peer_addr_len) {
				memcpy(&ctx->peer_addr,
				       &connreq->udcm_peer_addr,
				       connreq->udcm_peer_addr_len);
				ctx->peer_addr_len =
					connreq->udcm_peer_addr_len;
			}

			ofi_spin_lock(&dctx->lock);
			dlist_insert_tail(&ctx->list_entry, &dctx->ep_list);
			ofi_spin_unlock(&dctx->lock);
		}
	}
	return 0;
}

static void vrb_udcm_ep_preclose(struct vrb_ep *ep)
{
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_udcm_ep_ctx *ctx = ep->cm_ctx;
	struct vrb_udcm_domain_ctx *dctx = domain->cm_ctx;
	struct ibv_qp_attr attr = {0};

	/* With RDMA-CM, the QP already entered ERROR from the
	* disconnect event and rdma_destroy_ep manages the QP
	* lifecycle, so ep_preclose is a no-op.
	* With UDCM the QP may still be in RTS when fi_close is
	* called, so ep_preclose destroys it to clean the CQ. */

	if (ctx && dctx) {
		ofi_spin_lock(&dctx->lock);
		if (!dlist_empty(&ctx->list_entry)) {
			dlist_remove(&ctx->list_entry);
			dlist_init(&ctx->list_entry);
		}
		ofi_spin_unlock(&dctx->lock);
	}

	if (ep->ibv_qp) {
		attr.qp_state = IBV_QPS_ERR;
		(void) ibv_modify_qp(ep->ibv_qp, &attr, IBV_QP_STATE);

		ibv_destroy_qp(ep->ibv_qp);
		ep->ibv_qp = NULL;
	}
}

static int vrb_udcm_ep_close(struct vrb_ep *ep)
{
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_udcm_ep_ctx *ctx = ep->cm_ctx;
	struct vrb_udcm_domain_ctx *dctx = domain->cm_ctx;

	if (ctx) {
		if (dctx && !dlist_empty(&ctx->list_entry)) {
			ofi_spin_lock(&dctx->lock);
			dlist_remove(&ctx->list_entry);
			ofi_spin_unlock(&dctx->lock);
		}
		free(ctx);
		ep->cm_ctx = NULL;
	}
	vrb_udcm_domain_put(domain);
	return 0;
}

static int vrb_udcm_ep_bind(struct vrb_ep *ep, struct vrb_eq *eq)
{
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_udcm_eq_ctx *eq_ctx;
	struct vrb_udcm_domain_ctx *dctx;
	struct vrb_udcm_pep_ctx *pctx;
	struct vrb_pep *pep;
	struct ibv_device *dev;
	const char *dev_name;
	int ret;

	if (!eq->cm_ctx) {
		ret = vrb_udcm_eq_open(eq);
		if (ret)
			return ret;
	}
	eq_ctx = eq->cm_ctx;
	if (!eq_ctx->domain)
		eq_ctx->domain = domain;

	dctx = domain->cm_ctx;
	if (!dctx)
		return 0;

	if (dctx->ib)
		return 0;

	dlist_foreach_container(&eq->pep_list, struct vrb_pep,
				pep, eq_entry) {
		pctx = pep->cm_ctx;
		if (pctx && pctx->ib) {
			vrb_udcm_ib_ctx_ref(pctx->ib);
			dctx->ib = pctx->ib;
			return 0;
		}
	}

	dev_name = ibv_get_device_name(domain->verbs->device);
	dev = vrb_udcm_find_ib_device(dev_name);
	if (!dev) {
		VRB_WARN(FI_LOG_EP_CTRL,
			 "udcm: IB device '%s' not found\n", dev_name);
		return -FI_ENODATA;
	}

	dctx->ib = vrb_udcm_ib_ctx_create(dev);
	if (!dctx->ib)
		return -FI_ENOMEM;

	ret = ofi_epoll_add(eq->epollfd, dctx->ib->comp_channel->fd,
			    OFI_EPOLL_IN, NULL);
	if (ret) {
		vrb_udcm_ib_ctx_put(dctx->ib);
		dctx->ib = NULL;
		return -errno;
	}
	dctx->ib->eq = eq;

	return 0;
}

static int vrb_udcm_ep_enable(struct vrb_ep *ep, struct ibv_qp_init_attr *attr)
{
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_udcm_domain_ctx *dctx = domain->cm_ctx;
	struct vrb_udcm_ep_ctx *ep_ctx = ep->cm_ctx;
	int ret;

	if (ep->ibv_qp)
		return 0;

	ep->ibv_qp = ibv_create_qp(domain->pd, attr);
	if (!ep->ibv_qp)
		return -errno;

	ret = vrb_udcm_rc_qp_to_init(ep->ibv_qp, dctx->ib->port_num,
				     dctx->ib->pkey_index);
	if (ret)
		goto err;

	if (ep->state == VRB_REQ_RCVD) {
		ret = vrb_udcm_rc_qp_to_rtr(ep->ibv_qp, dctx->ib->port_num,
					     &ep_ctx->peer_name,
					     ep_ctx->peer_rc_qpn);
		if (ret)
			goto err;
	}
	return 0;
err:
	ibv_destroy_qp(ep->ibv_qp);
	ep->ibv_qp = NULL;
	return ret;
}

static int vrb_udcm_ep_setname(struct vrb_ep *ep, void *addr, size_t addrlen)
{
	void *dup;

	if (!addr || !addrlen)
		return -FI_EINVAL;

	dup = mem_dup(addr, addrlen);
	if (!dup)
		return -FI_ENOMEM;

	free(ep->info_attr.src_addr);
	ep->info_attr.src_addr = dup;
	ep->info_attr.src_addrlen = addrlen;
	return 0;
}

static int vrb_udcm_ep_getname(struct vrb_ep *ep, void *addr, size_t *addrlen)
{
	size_t src_len;

	if (!ep->info_attr.src_addr)
		return -FI_EOPBADSTATE;

	src_len = ep->info_attr.src_addrlen;
	if (*addrlen == 0) {
		*addrlen = src_len;
		return -FI_ETOOSMALL;
	}
	memcpy(addr, ep->info_attr.src_addr, MIN(*addrlen, src_len));
	*addrlen = src_len;
	return 0;
}

static int vrb_udcm_setname(struct vrb_pep *pep, void *addr, size_t addrlen)
{
	struct vrb_udcm_pep_ctx *ctx = pep->cm_ctx;
	void *dup;

	if (!addrlen || !addr)
		return -FI_EINVAL;

	if (pep->info->addr_format == FI_ADDR_IB_UD) {
		if (addrlen < sizeof(struct ofi_addr_ib_ud))
			return -FI_EINVAL;
		ctx->service = (int)((const struct ofi_addr_ib_ud *)addr)->service;
	} else if (ctx->service) {
		ofi_addr_set_port(addr, ctx->service);
	}

	dup = mem_dup(addr, addrlen);
	if (!dup)
		return -FI_ENOMEM;

	free(pep->info->src_addr);
	pep->info->src_addr = dup;
	pep->info->src_addrlen = addrlen;
	return 0;
}

static int vrb_udcm_getname(struct vrb_pep *pep, void *addr, size_t *addrlen)
{
	size_t src_len;

	if (!pep->info || !pep->info->src_addr)
		return -FI_EOPBADSTATE;

	src_len = pep->info->src_addrlen;
	if (*addrlen == 0) {
		*addrlen = src_len;
		return -FI_ETOOSMALL;
	}
	memcpy(addr, pep->info->src_addr, MIN(*addrlen, src_len));
	*addrlen = src_len;
	return 0;
}

static int vrb_udcm_getpeer(struct vrb_ep *ep, void *addr, size_t *addrlen)
{
	struct vrb_udcm_ep_ctx *ep_ctx = ep->cm_ctx;
	size_t src_len = sizeof(struct ofi_addr_ib_ud);

	if (!ep_ctx)
		return -FI_EOPBADSTATE;

	if (*addrlen == 0) {
		*addrlen = src_len;
		return -FI_ETOOSMALL;
	}
	memcpy(addr, &ep_ctx->peer_name, MIN(*addrlen, src_len));
	*addrlen = src_len;
	return 0;
}

struct vrb_cm_ops vrb_udcm_ops = {
	.connect	= vrb_udcm_connect,
	.accept		= vrb_udcm_accept,
	.shutdown	= vrb_udcm_shutdown,
	.listen		= vrb_udcm_listen,
	.reject		= vrb_udcm_reject,
	.progress	= vrb_udcm_progress,
	.ep_init	= vrb_udcm_ep_init,
	.ep_close	= vrb_udcm_ep_close,
	.ep_preclose	= vrb_udcm_ep_preclose,
	.ep_bind	= vrb_udcm_ep_bind,
	.ep_enable	= vrb_udcm_ep_enable,
	.ep_setname	= vrb_udcm_ep_setname,
	.ep_getname	= vrb_udcm_ep_getname,
	.pep_init	= vrb_udcm_pep_init,
	.pep_close	= vrb_udcm_pep_close,
	.pep_bind	= vrb_udcm_pep_bind,
	.domain_init	= vrb_udcm_domain_init,
	.domain_close	= vrb_udcm_domain_close,
	.eq_open	= vrb_udcm_eq_open,
	.eq_close	= vrb_udcm_eq_close,
	.setname	= vrb_udcm_setname,
	.getname	= vrb_udcm_getname,
	.getpeer	= vrb_udcm_getpeer,
	.getinfo_addr	= vrb_udcm_getinfo_addr,
	.cm_backend	= VRB_CM_UDCM,
};
