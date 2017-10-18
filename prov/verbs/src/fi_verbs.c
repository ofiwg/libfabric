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

struct fi_ibv_gl_data fi_ibv_gl_data = {
	.def_tx_size		= 384,
	.def_rx_size		= 384,
	.def_tx_iov_limit	= 4,
	.def_rx_iov_limit	= 4,
	.min_rnr_timer		= VERBS_DEFAULT_MIN_RNR_TIMER,
	.fork_unsafe		= 0,
	.use_odp		= 1,
	.cqread_bunch_size	= 8,
	.iface			= NULL,

	.rdm			= {
		.buffer_num		= FI_IBV_RDM_TAGGED_DFLT_BUFFER_NUM,
		.buffer_size		= FI_IBV_RDM_DFLT_BUFFERED_SIZE,
		.rndv_seg_size		= FI_IBV_RDM_SEG_MAXSIZE,
		.thread_timeout		= FI_IBV_RDM_CM_THREAD_TIMEOUT,
		.eager_send_opcode	= "IBV_WR_SEND",
	},

	.dgram			= {
		.use_name_server	= 1,
		.name_server_port	= 5678,
	},
};

struct fi_provider fi_ibv_prov = {
	.name = VERBS_PROV_NAME,
	.version = VERBS_PROV_VERS,
	.fi_version = FI_VERSION(1, 5),
	.getinfo = fi_ibv_getinfo,
	.fabric = fi_ibv_fabric,
	.cleanup = fi_ibv_fini
};

struct util_prov fi_ibv_util_prov = {
	.prov = &fi_ibv_prov,
	.info = NULL,
	/* The support of the shared recieve contexts
	 * is dynamically calculated */
	.flags = 0,
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

// WR must be filled out by now except for context
ssize_t
fi_ibv_send(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr, void *context)
{
	struct fi_ibv_wre *wre = NULL;
	struct ibv_send_wr *bad_wr;
	int ret;

	if (ep->scq) {
		if (wr->send_flags & IBV_SEND_SIGNALED) {
			wre = util_buf_alloc(ep->wre_pool);
			if (OFI_UNLIKELY(!wre))
				return -FI_EAGAIN;
			memset(wre, 0, sizeof(*wre));
			wre->context = context;
			wr->wr_id = (uintptr_t)wre;
			wre->ep = ep;
			wre->context = context;
			wre->wr.type = IBV_SEND_WR;
			wre->wr.swr = *wr;

			assert((wr->wr_id & ep->scq->wr_id_mask) !=
			       ep->scq->send_signal_wr_id);
			ofi_atomic_set32(&ep->unsignaled_send_cnt, 0);

			dlist_insert_tail(&wre->entry, &ep->wre_list);
		} else {
			if (VERBS_SIGNAL_SEND(ep)) {
				ret = fi_ibv_signal_send(ep, wr);
				if (ret)
					return ret;
			} else {
				wr->wr_id = 0ULL;
				ofi_atomic_inc32(&ep->unsignaled_send_cnt);

				if (ofi_atomic_get32(&ep->unsignaled_send_cnt) >=
						VERBS_SEND_COMP_THRESH(ep)) {
					ret = fi_ibv_reap_comp(ep);
					if (ret)
						return ret;
				}
			}
		}
	}

	ret = ibv_post_send(ep->id->qp, wr, &bad_wr);
	if (ret) {
		if (wre) {
			dlist_remove(&wre->entry);
			util_buf_release(ep->wre_pool, wre);
		}
		switch (ret) {
		case ENOMEM:
			return -FI_EAGAIN;
		case -1:
			/* Deal with non-compliant libibverbs drivers
			 * which set errno instead of directly returning
			 * the error value */
			return (errno == ENOMEM) ? -FI_EAGAIN : -errno;
		default:
			return -ret;
		}
	}

	return ret;
}

ssize_t fi_ibv_send_buf(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
			const void *buf, size_t len, void *desc, void *context)
{
	struct ibv_sge sge = fi_ibv_init_sge(buf, len, desc);

	wr->sg_list = &sge;
	wr->num_sge = 1;

	return fi_ibv_send(ep, wr, context);
}

ssize_t fi_ibv_send_buf_inline(struct fi_ibv_msg_ep *ep, struct ibv_send_wr *wr,
			       const void *buf, size_t len)
{
	struct ibv_sge sge = fi_ibv_init_sge_inline(buf, len);

	wr->sg_list = &sge;
	wr->num_sge = 1;

	return fi_ibv_send(ep, wr, NULL);
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

	wr->num_sge = count;
	wr->send_flags = VERBS_INJECT_FLAGS(ep, len, flags) | VERBS_COMP_FLAGS(ep, flags);

	if (flags & FI_FENCE)
		wr->send_flags |= IBV_SEND_FENCE;

	return fi_ibv_send(ep, wr, context);
}

static int fi_ibv_param_define(const char *param_name, const char *param_str,
			       enum fi_param_type type, void *param_default)
{
	char *param_help, param_default_str[256] = { 0 };
	char *begin_def_section = " (default: ", *end_def_section = ")";
	int ret = FI_SUCCESS;
	size_t len, param_default_sz = 0;
	size_t param_str_sz = strlen(param_str);
	size_t begin_def_section_sz = strlen(begin_def_section);
	size_t end_def_section_sz = strlen(end_def_section);

	if (param_default != NULL) {
		switch (type) {
		case FI_PARAM_STRING:
			if (*(char **)param_default != NULL) {
				param_default_sz =
					MIN(strlen(*(char **)param_default),
					    254);
				strncpy(param_default_str, *(char **)param_default,
					param_default_sz);
				param_default_str[param_default_sz + 1] = '\0';
			}
			break;
		case FI_PARAM_INT:
		case FI_PARAM_BOOL:
			snprintf(param_default_str, 256, "%d", *((int *)param_default));
			param_default_sz = strlen(param_default_str);
			break;
		default:
			assert(0);
			ret = -FI_EINVAL;
			goto fn;
		}
	}

	len = param_str_sz + strlen(begin_def_section) +
		param_default_sz + end_def_section_sz + 1;
	param_help = calloc(1, len);
	if (!param_help) {
 		assert(0);
		ret = -FI_ENOMEM;
		goto fn;
	}

	strncat(param_help, param_str, param_str_sz);
	strncat(param_help, begin_def_section, begin_def_section_sz);
	strncat(param_help, param_default_str, param_default_sz);
	strncat(param_help, end_def_section, end_def_section_sz);

	param_help[len - 1] = '\0';

	fi_param_define(&fi_ibv_prov, param_name, type, param_help);

	free(param_help);
fn:
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
	FI_DBG(&fi_ibv_prov, FI_LOG_EP_CTRL, "QP attributes: "
	       "min_rnr_timer"	": %" PRIu8 ", "
	       "timeout"	": %" PRIu8 ", "
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

	if (fi_ibv_gl_data.min_rnr_timer > 31) {
		VERBS_WARN(FI_LOG_EQ, "min_rnr_timer value out of valid range; "
			   "using default value of %d\n",
			   VERBS_DEFAULT_MIN_RNR_TIMER);
		attr.min_rnr_timer = VERBS_DEFAULT_MIN_RNR_TIMER;
	} else {
		attr.min_rnr_timer = fi_ibv_gl_data.min_rnr_timer;
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

static int fi_ibv_get_param_int(const char *param_name,
				const char *param_str,
				int *param_default)
{
	int param, ret;

	ret = fi_ibv_param_define(param_name, param_str,
				  FI_PARAM_INT,
				  param_default);
	if (ret)
		return ret;

	if (!fi_param_get_int(&fi_ibv_prov, param_name, &param))
		*param_default = param;

	return 0;
}

static int fi_ibv_get_param_bool(const char *param_name,
				 const char *param_str,
				 int *param_default)
{
	int param, ret;

	ret = fi_ibv_param_define(param_name, param_str,
				  FI_PARAM_BOOL,
				  param_default);
	if (ret)
		return ret;

	if (!fi_param_get_bool(&fi_ibv_prov, param_name, &param)) {
		*param_default = param;
		if ((*param_default != 1) && (*param_default != 0))
			return -FI_EINVAL;
	}

	return 0;
}

static int fi_ibv_get_param_str(const char *param_name,
				const char *param_str,
				char **param_default)
{
	char *param;
	int ret;

	ret = fi_ibv_param_define(param_name, param_str,
				  FI_PARAM_STRING,
				  param_default);
	if (ret)
		return ret;

	if (!fi_param_get_str(&fi_ibv_prov, param_name, &param))
		*param_default = param;

	return 0;
}

static int fi_ibv_read_params(void)
{
	/* Common parameters */
	if (fi_ibv_get_param_int("tx_size", "Default maximum tx context size",
				 &fi_ibv_gl_data.def_tx_size) ||
	    (fi_ibv_gl_data.def_tx_size < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of tx_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("rx_size", "Default maximum rx context size",
				 &fi_ibv_gl_data.def_rx_size) ||
	    (fi_ibv_gl_data.def_rx_size < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of rx_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("tx_iov_limit", "Default maximum tx iov_limit",
				 &fi_ibv_gl_data.def_tx_iov_limit) ||
	    (fi_ibv_gl_data.def_tx_iov_limit < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of tx_iov_limit\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("rx_iov_limit", "Default maximum rx iov_limit",
				 &fi_ibv_gl_data.def_rx_iov_limit) ||
	    (fi_ibv_gl_data.def_rx_iov_limit < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of rx_iov_limit\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("inline_size", "Default maximum inline size. "
				 "Actual inject size returned in fi_info may be "
				 "greater", &fi_ibv_gl_data.def_inline_size) ||
	    (fi_ibv_gl_data.def_inline_size < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of inline_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("min_rnr_timer", "Set min_rnr_timer QP "
				 "attribute (0 - 31)",
				 &fi_ibv_gl_data.min_rnr_timer) ||
	    ((fi_ibv_gl_data.min_rnr_timer < 0) ||
	     (fi_ibv_gl_data.min_rnr_timer > 31))) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of min_rnr_timer\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_bool("fork_unsafe", "Enable safety of fork() system call "
				  "for verbs provider. If you're sure that fork() "
				  "support isn't needed - No need to use this option, "
				  "because extra memory will be consumed when enabling "
				  "fork suppport.",
				  &fi_ibv_gl_data.fork_unsafe)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of fork_unsafe\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_bool("use_odp", "Enable on-demand paging experimental feature",
				  &fi_ibv_gl_data.use_odp)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of use_odp\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("cqread_bunch_size", "The number of entries to "
				 "be read from the verbs completion queue at a time",
				 &fi_ibv_gl_data.cqread_bunch_size) ||
	    (fi_ibv_gl_data.cqread_bunch_size <= 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of cqread_bunch_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_str("iface", "The prefix or the full name of the "
				 "network interface associated with the IB device",
				 &fi_ibv_gl_data.iface)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of iface\n");
		return -FI_EINVAL;
	}

	/* RDM-specific parameters */
	if (fi_ibv_get_param_int("rdm_buffer_num", "The number of pre-registered "
				 "buffers for buffered operations between "
				 "the endpoints, must be a power of 2",
				 &fi_ibv_gl_data.rdm.buffer_num) ||
	    (fi_ibv_gl_data.rdm.buffer_num & (fi_ibv_gl_data.rdm.buffer_num - 1))) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of rdm_buffer_num\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("rdm_buffer_size", "The maximum size of a "
				 "buffered operation (bytes)",
				 &fi_ibv_gl_data.rdm.buffer_size) ||
	    (fi_ibv_gl_data.rdm.buffer_size < sizeof(struct fi_ibv_rdm_rndv_header))) {
		VERBS_WARN(FI_LOG_CORE,
			   "rdm_buffer_size should be greater than %"PRIu64"\n",
			   sizeof(struct fi_ibv_rdm_rndv_header));
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("rdm_rndv_seg_size", "The segment size for "
				 "zero copy protocols (bytes)",
				 &fi_ibv_gl_data.rdm.rndv_seg_size) ||
	    (fi_ibv_gl_data.rdm.rndv_seg_size <= 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of rdm_rndv_seg_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("rdm_thread_timeout", "The wake up timeout of "
				 "the helper thread (usec)",
				 &fi_ibv_gl_data.rdm.thread_timeout) ||
	    (fi_ibv_gl_data.rdm.thread_timeout < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of rdm_thread_timeout\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_str("rdm_eager_send_opcode", "The operation code that "
				 "will be used for eager messaging. Only IBV_WR_SEND "
				 "and IBV_WR_RDMA_WRITE_WITH_IMM are supported. "
				 "The last one is not applicable for iWarp.",
				 &fi_ibv_gl_data.rdm.eager_send_opcode)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of rdm_eager_send_opcode\n");
		return -FI_EINVAL;
	}

	/* DGRAM-specific parameters */
	if (getenv("OMPI_COMM_WORLD_RANK") || getenv("PMI_RANK"))
		fi_ibv_gl_data.dgram.use_name_server = 0;
	if (fi_ibv_get_param_bool("dgram_use_name_server", "The option that "
				  "enables/disables OFI Name Server thread that is used "
				  "to resolve IP-addresses to provider specific "
				  "addresses. If MPI is used, the NS is disabled "
				  "by default.", &fi_ibv_gl_data.dgram.use_name_server)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of dgram_use_name_server\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("dgram_name_server_port", "The port on which Name Server "
				 "thread listens incoming connections and requestes.",
				 &fi_ibv_gl_data.dgram.name_server_port) ||
	    (fi_ibv_gl_data.dgram.name_server_port < 0 ||
	     fi_ibv_gl_data.dgram.name_server_port > 65535)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of dgram_name_server_port\n");
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

static void fi_ibv_fini(void)
{
	fi_freeinfo((void *)fi_ibv_util_prov.info);
	fi_ibv_util_prov.info = NULL;
}

VERBS_INI
{
	if (fi_ibv_read_params()|| fi_ibv_init_info(&fi_ibv_util_prov.info))
		return NULL;
	return &fi_ibv_prov;
}
