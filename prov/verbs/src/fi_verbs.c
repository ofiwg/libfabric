/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <fi_mem.h>

#include "fi_verbs.h"
#include "ep_rdm/verbs_rdm.h"

static void fi_ibv_fini(void);

static const char *local_node = "localhost";

#define VERBS_DEFAULT_MIN_RNR_TIMER 12

size_t verbs_default_tx_size 		= 384;
size_t verbs_default_rx_size 		= 384;
size_t verbs_default_tx_iov_limit 	= 4;
size_t verbs_default_rx_iov_limit 	= 4;
size_t verbs_default_inline_size 	= 64;

size_t verbs_min_rnr_timer = VERBS_DEFAULT_MIN_RNR_TIMER;

struct fi_provider fi_ibv_prov = {
	.name = VERBS_PROV_NAME,
	.version = VERBS_PROV_VERS,
	.fi_version = FI_VERSION(1, 5),
	.getinfo = fi_ibv_getinfo,
	.fabric = fi_ibv_fabric,
	.cleanup = fi_ibv_fini
};

int fi_ibv_sockaddr_len(struct sockaddr *addr)
{
	if (!addr)
		return 0;

	switch (addr->sa_family) {
	case AF_INET:
		return sizeof(struct sockaddr_in);
	case AF_INET6:
		return sizeof(struct sockaddr_in6);
	case AF_IB:
		return sizeof(struct sockaddr_ib);
	default:
		return 0;
	}
}

int fi_ibv_rdm_cm_bind_ep(struct fi_ibv_rdm_cm *cm, struct fi_ibv_rdm_ep *ep)
{
	char my_ipoib_addr_str[INET6_ADDRSTRLEN];

	assert(cm->ec && cm->listener);

	if (ep->info->src_addr) {
		memcpy(&ep->my_addr, ep->info->src_addr, sizeof(ep->my_addr));

		inet_ntop(ep->my_addr.sin_family,
			  &ep->my_addr.sin_addr.s_addr,
			  my_ipoib_addr_str, INET_ADDRSTRLEN);
	} else {
		strcpy(my_ipoib_addr_str, "undefined");
	}

	VERBS_INFO(FI_LOG_EP_CTRL, "My IPoIB: %s\n", my_ipoib_addr_str);

	if (!cm->is_bound) {
		errno = 0;
		if (rdma_bind_addr(cm->listener, (struct sockaddr *)&ep->my_addr)) {
			VERBS_INFO(FI_LOG_EP_CTRL,
				"Failed to bind cm listener to my IPoIB addr %s: %s\n",
				my_ipoib_addr_str, strerror(errno));
			return -FI_EOTHER;
		}
		if (rdma_listen(cm->listener, 1024)) {
			VERBS_INFO(FI_LOG_EP_CTRL, "rdma_listen failed: %s\n",
				strerror(errno));
			return -FI_EOTHER;
		}
		cm->is_bound = 1;
	}

	if (!ep->my_addr.sin_port) {
		ep->my_addr.sin_port = rdma_get_src_port(cm->listener);
	}
	assert(ep->my_addr.sin_family == AF_INET);

	VERBS_INFO(FI_LOG_EP_CTRL, "My ep_addr: %s:%u\n",
		inet_ntoa(ep->my_addr.sin_addr), ntohs(ep->my_addr.sin_port));

	return FI_SUCCESS;
}

int fi_ibv_get_rdma_rai(const char *node, const char *service, uint64_t flags,
		   const struct fi_info *hints, struct rdma_addrinfo **rai)
{
	struct rdma_addrinfo rai_hints, *_rai;
	struct rdma_addrinfo **rai_current;
	int ret = fi_ibv_fi_to_rai(hints, flags, &rai_hints);

	if (ret)
		goto out;

	if (!node && !rai_hints.ai_dst_addr) {
		if ((!rai_hints.ai_src_addr && !service) ||
		    (!rai_hints.ai_src_addr && FI_IBV_EP_TYPE_IS_RDM(hints)))
		{
			node = local_node;
		}
		rai_hints.ai_flags |= RAI_PASSIVE;
	}

	ret = rdma_getaddrinfo((char *) node, (char *) service,
				&rai_hints, &_rai);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_getaddrinfo", errno);
		if (errno) {
			ret = -errno;
		}
		goto out;
	}

	/*
	 * If caller requested rai, remove ib_rai entries added by IBACM to
	 * prevent wrong ib_connect_hdr from being sent in connect request.
	 */
	if (rai && hints && (hints->addr_format != FI_SOCKADDR_IB)) {
		for (rai_current = &_rai; *rai_current;) {
			struct rdma_addrinfo *rai_next;
			if ((*rai_current)->ai_family == AF_IB) {
				rai_next = (*rai_current)->ai_next;
				(*rai_current)->ai_next = NULL;
				rdma_freeaddrinfo(*rai_current);
				*rai_current = rai_next;
				continue;
			}
			rai_current = &(*rai_current)->ai_next;
		}
	}

	if (rai)
		*rai = _rai;

out:
	if (rai_hints.ai_src_addr)
		free(rai_hints.ai_src_addr);
	if (rai_hints.ai_dst_addr)
		free(rai_hints.ai_dst_addr);
	return ret;
}

int fi_ibv_create_ep(const char *node, const char *service,
		     uint64_t flags, const struct fi_info *hints,
		     struct rdma_addrinfo **rai, struct rdma_cm_id **id)
{
	struct rdma_addrinfo *_rai = NULL;
	int ret;

	ret = fi_ibv_get_rdma_rai(node, service, flags, hints, &_rai);
	if (ret) {
		return ret;
	}

	ret = rdma_create_ep(id, _rai, NULL, NULL);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_create_ep", errno);
		ret = -errno;
		goto err1;
	}

	if (rai) {
		*rai = _rai;
	} else {
		rdma_freeaddrinfo(_rai);
	}

	return ret;
err1:
	rdma_freeaddrinfo(_rai);

	return ret;
}

void fi_ibv_destroy_ep(struct rdma_addrinfo *rai, struct rdma_cm_id **id)
{
	rdma_freeaddrinfo(rai);
	rdma_destroy_ep(*id);
}

#define VERBS_SIGNAL_SEND(ep) \
	(ofi_atomic_get32(&ep->unsignaled_send_cnt) >= VERBS_SEND_SIGNAL_THRESH(ep) && \
	 !ofi_atomic_get32(&ep->comp_pending))

static int fi_ibv_signal_send(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr)
{
	struct fi_ibv_msg_epe *epe;

	fastlock_acquire(&ep->scq->lock);
	if (VERBS_SIGNAL_SEND(ep)) {
		epe = util_buf_alloc(ep->scq->epe_pool);
		if (!epe) {
			fastlock_release(&ep->scq->lock);
			return -FI_ENOMEM;
		}
		memset(epe, 0, sizeof(*epe));
		wr->send_flags |= IBV_SEND_SIGNALED;
		wr->wr_id = ep->ep_id;
		epe->ep = ep;
		slist_insert_tail(&epe->entry, &ep->scq->ep_list);
		ofi_atomic_inc32(&ep->comp_pending);
	}
	fastlock_release(&ep->scq->lock);
	return 0;
}

static int fi_ibv_reap_comp(struct fi_ibv_msg_ep *ep)
{
	struct fi_ibv_wce *wce = NULL;
	int got_wc = 0;
	int ret = 0;

	fastlock_acquire(&ep->scq->lock);
	while (ofi_atomic_get32(&ep->comp_pending) > 0) {
		if (!wce) {
			wce = util_buf_alloc(ep->scq->wce_pool);
			if (!wce) {
				fastlock_release(&ep->scq->lock);
				return -FI_ENOMEM;
			}
			memset(wce, 0, sizeof(*wce));
		}
		ret = fi_ibv_poll_cq(ep->scq, &wce->wc);
		if (ret < 0) {
			VERBS_WARN(FI_LOG_EP_DATA,
				   "Failed to read completion for signaled send\n");
			util_buf_release(ep->scq->wce_pool, wce);
			fastlock_release(&ep->scq->lock);
			return ret;
		} else if (ret > 0) {
			slist_insert_tail(&wce->entry, &ep->scq->wcq);
			got_wc = 1;
			wce = NULL;
		}
	}
	if (wce)
		util_buf_release(ep->scq->wce_pool, wce);

	if (got_wc && ep->scq->channel)
		ret = fi_ibv_cq_signal(&ep->scq->cq_fid);

	fastlock_release(&ep->scq->lock);
	return ret;
}

ssize_t fi_ibv_send(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr, size_t len,
		    int count, void *context)
{
	struct ibv_send_wr *bad_wr;
	int ret;

	assert(ep->scq);
	wr->num_sge = count;
	wr->wr_id = (uintptr_t) context;

	if (wr->send_flags & IBV_SEND_SIGNALED) {
		assert((wr->wr_id & ep->scq->wr_id_mask) != ep->scq->send_signal_wr_id);
		ofi_atomic_set32(&ep->unsignaled_send_cnt, 0);
	} else {
		if (VERBS_SIGNAL_SEND(ep)) {
			ret = fi_ibv_signal_send(ep, wr);
			if (ret)
				return ret;
		} else {
			ofi_atomic_inc32(&ep->unsignaled_send_cnt);

			if (ofi_atomic_get32(&ep->unsignaled_send_cnt) >=
					VERBS_SEND_COMP_THRESH(ep)) {
				ret = fi_ibv_reap_comp(ep);
				if (ret)
					return ret;
			}
		}
	}

	ret = ibv_post_send(ep->id->qp, wr, &bad_wr);
	switch (ret) {
	case ENOMEM:
		return -FI_EAGAIN;
	case -1:
		/* Deal with non-compliant libibverbs drivers which set errno
		 * instead of directly returning the error value */
		return (errno == ENOMEM) ? -FI_EAGAIN : -errno;
	default:
		return -ret;
	}
}

ssize_t fi_ibv_send_buf(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
			const void *buf, size_t len, void *desc, void *context)
{
	struct ibv_sge sge;

	fi_ibv_set_sge(sge, buf, len, desc);
	wr->sg_list = &sge;

	return fi_ibv_send(ep, wr, len, 1, context);
}

ssize_t fi_ibv_send_buf_inline(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
			       const void *buf, size_t len)
{
	struct ibv_sge sge;

	fi_ibv_set_sge_inline(sge, buf, len);
	wr->sg_list = &sge;

	return fi_ibv_send(ep, wr, len, 1, NULL);
}

ssize_t fi_ibv_send_iov_flags(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
			      const struct iovec *iov, void **desc, int count,
			      void *context, uint64_t flags)
{
	size_t len = 0;

	if (!desc)
		fi_ibv_set_sge_iov_inline(wr->sg_list, iov, count, len);
	else
		fi_ibv_set_sge_iov(wr->sg_list, iov, count, desc, len);

	wr->send_flags = VERBS_INJECT_FLAGS(ep, len, flags) | VERBS_COMP_FLAGS(ep, flags);

	if (flags & FI_FENCE)
		wr->send_flags = IBV_SEND_FENCE;

	return fi_ibv_send(ep, wr, len, count, context);
}

static int fi_ibv_get_param_int(char *param_name, char *param_str,
				size_t *param_default)
{
	char *param_help;
	size_t len, ret_len;
	int param, ret = FI_SUCCESS;

	len = strlen(param_str) + 50;
	param_help = calloc(1, len);
	if (!param_help)
		return -FI_ENOMEM;

	ret_len = snprintf(param_help, len, "%s (default: %zu)", param_str,
			   *param_default);
	if (ret_len >= len) {
		VERBS_WARN(FI_LOG_EP_DATA,
			   "param_help string size insufficient!\n");
		assert(0);
		ret = -FI_ETOOSMALL;
		goto out;
	}

	fi_param_define(&fi_ibv_prov, param_name, FI_PARAM_INT, param_help);

	if (!fi_param_get_int(&fi_ibv_prov, param_name, &param))
		*param_default = param;

out:
	free(param_help);
	return ret;
}

#if ENABLE_DEBUG
static int fi_ibv_dbg_query_qp_attr(struct ibv_qp *qp)
{
	struct ibv_qp_init_attr attr = { 0 };
	struct ibv_qp_attr qp_attr = { 0 };
	int ret;

	ret = ibv_query_qp(qp, &qp_attr, IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
			   IBV_QP_RNR_RETRY | IBV_QP_MIN_RNR_TIMER, &attr);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to query QP\n");
		return ret;
	}
	FI_DBG_TRACE(&fi_ibv_prov, FI_LOG_EP_CTRL, "QP attributes: "
		     "min_rnr_timer"	": %" PRIu8 ", "
		     "timeout"		": %" PRIu8 ", "
		     "retry_cnt"	": %" PRIu8 ", "
		     "rnr_retry"	": %" PRIu8 "\n",
		     qp_attr.min_rnr_timer, qp_attr.timeout, qp_attr.retry_cnt,
		     qp_attr.rnr_retry);
	return 0;
}
#else
static int fi_ibv_dbg_query_qp_attr(struct ibv_qp *qp)
{
	return 0;
}
#endif

int fi_ibv_set_rnr_timer(struct ibv_qp *qp)
{
	struct ibv_qp_attr attr = { 0 };
	int ret;

	if (verbs_min_rnr_timer > 31) {
		VERBS_WARN(FI_LOG_EQ, "min_rnr_timer value out of valid range; "
			   "using default value of %d\n",
			   VERBS_DEFAULT_MIN_RNR_TIMER);
		attr.min_rnr_timer = VERBS_DEFAULT_MIN_RNR_TIMER;
	} else {
		attr.min_rnr_timer = verbs_min_rnr_timer;
	}

	ret = ibv_modify_qp(qp, &attr, IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		VERBS_WARN(FI_LOG_EQ, "Unable to modify QP attribute\n");
		return ret;
	}
	ret = fi_ibv_dbg_query_qp_attr(qp);
	if (ret)
		return ret;
	return 0;
}

static void fi_ibv_fini(void)
{
	fi_ibv_free_info();
}

VERBS_INI
{
	fi_param_define(&fi_ibv_prov, "iface", FI_PARAM_STRING,
			"the prefix or the full name of the network interface "
			"associated with the IB device (default: ib)");
	fi_param_define(&fi_ibv_prov, "rdm_buffer_num", FI_PARAM_INT,
			"the number of pre-registered buffers for buffered "
			"operations between the endpoints, must be a power of 2 "
			"(default: 8)");
	fi_param_define(&fi_ibv_prov, "rdm_buffer_size", FI_PARAM_INT,
			"the maximum size of a buffered operation (bytes) "
			"(default: platform specific)");
	fi_param_define(&fi_ibv_prov, "rdm_use_odp", FI_PARAM_BOOL,
			"enable on-demand paging experimental feature"
			"(default: platform specific)");
	fi_param_define(&fi_ibv_prov, "rdm_rndv_seg_size", FI_PARAM_INT,
			"the segment size for zero copy protocols (bytes)"
			"(default: platform specific");
	fi_param_define(&fi_ibv_prov, "rdm_cqread_bunch_size", FI_PARAM_INT,
			"the number of entries to be read from the verbs "
			"completion queue at a time (default: 8)");
	fi_param_define(&fi_ibv_prov, "rdm_thread_timeout", FI_PARAM_INT,
			"the wake up timeout of the helper thread (usec) "
			"(default: 100)");
	fi_param_define(&fi_ibv_prov, "rdm_eager_send_opcode", FI_PARAM_STRING,
			"the operation code that will be used for eager messaging. "
			"Only IBV_WR_SEND and IBV_WR_RDMA_WRITE_WITH_IMM are supported. "
			"The last one is not applicable for iWarp. "
			"(default: IBV_WR_SEND)");

	if (fi_ibv_get_param_int("tx_size", "Default maximum tx context size",
				 &verbs_default_tx_size))
		return NULL;

	if (fi_ibv_get_param_int("rx_size", "Default maximum rx context size",
				 &verbs_default_rx_size))
		return NULL;

	if (fi_ibv_get_param_int("tx_iov_limit", "Default maximum tx iov_limit",
				 &verbs_default_tx_iov_limit))
		return NULL;

	if (fi_ibv_get_param_int("rx_iov_limit", "Default maximum rx iov_limit",
				 &verbs_default_rx_iov_limit))
		return NULL;

	if (fi_ibv_get_param_int("inline_size", "Default maximum inline size. "
				 "Actual inject size returned in fi_info may be "
				 "greater", &verbs_default_inline_size))
		return NULL;

	if (fi_ibv_get_param_int("min_rnr_timer", "Set min_rnr_timer QP "
				 "attribute (0 - 31)", &verbs_min_rnr_timer))
		return NULL;

	return &fi_ibv_prov;
}
