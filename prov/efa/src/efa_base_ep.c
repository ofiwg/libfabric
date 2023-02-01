/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <sys/time.h>
#include "efa.h"
#include "efa_av.h"
#include "rdm/rdm_proto_v4.h"

int efa_base_ep_bind_av(struct efa_base_ep *base_ep, struct efa_av *av)
{
	/*
	 * Binding multiple endpoints to a single AV is currently not
	 * supported.
	 */
	if (av->base_ep) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Address vector already has endpoint bound to it.\n");
		return -FI_ENOSYS;
	}
	if (base_ep->domain != av->domain) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Address vector doesn't belong to same domain as EP.\n");
		return -FI_EINVAL;
	}
	if (base_ep->av) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Address vector already bound to EP.\n");
		return -FI_EINVAL;
	}

	base_ep->av = av;
	base_ep->av->base_ep = base_ep;

	return 0;
}

static int efa_base_ep_destruct_qp(struct efa_base_ep *base_ep)
{
	struct efa_domain *domain;
	struct efa_qp *qp = base_ep->qp;
	int err;

	if (qp) {
		domain = qp->base_ep->domain;
		domain->qp_table[qp->qp_num & domain->qp_table_sz_m1] = NULL;
		err = -ibv_destroy_qp(qp->ibv_qp);
		if (err)
			EFA_INFO(FI_LOG_CORE, "destroy qp[%u] failed!\n", qp->qp_num);

		free(qp);
	}

	return 0;
}

int efa_base_ep_destruct(struct efa_base_ep *base_ep)
{
	int err;

	free(base_ep->src_addr);
	fi_freeinfo(base_ep->info);

	if (base_ep->self_ah)
		ibv_destroy_ah(base_ep->self_ah);

	err = efa_base_ep_destruct_qp(base_ep);

	if (base_ep->util_ep_initialized) {
		err = ofi_endpoint_close(&base_ep->util_ep);
		if (err)
			EFA_WARN(FI_LOG_EP_CTRL, "Unable to close util EP\n");
	}

	return err;
}

static int efa_generate_rdm_connid(void)
{
	struct timeval tv;
	uint32_t val;
	int err;

	err = gettimeofday(&tv, NULL);
	if (err) {
		EFA_WARN(FI_LOG_EP_CTRL, "Cannot gettimeofday, err=%d.\n", err);
		return 0;
	}

	/* tv_usec is in range [0,1,000,000), shift it by 12 to [0,4,096,000,000 */
	val = (tv.tv_usec << 12) + tv.tv_sec;

	val = ofi_xorshift_random(val);

	/* 0x80000000 and up is privileged Q Key range. */
	val &= 0x7fffffff;

	return val;
}

static int efa_base_ep_modify_qp_state(struct efa_base_ep *base_ep,
				       struct efa_qp *qp,
				       enum ibv_qp_state qp_state,
				       int attr_mask)
{
	struct ibv_qp_attr attr = { 0 };

	attr.qp_state = qp_state;

	if (attr_mask & IBV_QP_PORT)
		attr.port_num = 1;

	if (attr_mask & IBV_QP_QKEY)
		attr.qkey = qp->qkey;

	if (attr_mask & IBV_QP_RNR_RETRY)
		attr.rnr_retry = base_ep->rnr_retry;

	return -ibv_modify_qp(qp->ibv_qp, &attr, attr_mask);
}

static int efa_base_ep_modify_qp_rst2rts(struct efa_base_ep *base_ep,
					 struct efa_qp *qp)
{
	int err;

	err = efa_base_ep_modify_qp_state(base_ep, qp, IBV_QPS_INIT,
					  IBV_QP_STATE | IBV_QP_PKEY_INDEX |
						  IBV_QP_PORT | IBV_QP_QKEY);
	if (err)
		return err;

	err = efa_base_ep_modify_qp_state(base_ep, qp, IBV_QPS_RTR,
					  IBV_QP_STATE);
	if (err)
		return err;

	if (base_ep->util_ep.type != FI_EP_DGRAM &&
	    efa_domain_support_rnr_retry_modify(base_ep->domain))
		return efa_base_ep_modify_qp_state(
			base_ep, qp, IBV_QPS_RTS,
			IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY);

	return efa_base_ep_modify_qp_state(base_ep, qp, IBV_QPS_RTS,
					   IBV_QP_STATE | IBV_QP_SQ_PSN);
}

static int efa_base_ep_create_qp_ex(struct efa_base_ep *base_ep,
			     struct ibv_qp_init_attr_ex *init_attr_ex)
{
	struct efa_domain *domain;
	struct efa_qp *qp;
	struct efadv_qp_init_attr efa_attr = { 0 };
	int err;

	domain = base_ep->domain;
	qp = calloc(1, sizeof(*qp));
	if (!qp)
		return -FI_ENOMEM;

	if (init_attr_ex->qp_type == IBV_QPT_UD) {
		qp->ibv_qp = ibv_create_qp_ex(init_attr_ex->pd->context,
					      init_attr_ex);
	} else {
		assert(init_attr_ex->qp_type == IBV_QPT_DRIVER);
		efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
		qp->ibv_qp = efadv_create_qp_ex(
			init_attr_ex->pd->context, init_attr_ex, &efa_attr,
			sizeof(struct efadv_qp_init_attr));
	}

	if (!qp->ibv_qp) {
		EFA_WARN(FI_LOG_EP_CTRL, "ibv_create_qp failed\n");
		err = -EINVAL;
		goto err_free_qp;
	}

	qp->ibv_qp_ex = ibv_qp_to_qp_ex(qp->ibv_qp);
	qp->qkey = (init_attr_ex->qp_type == IBV_QPT_UD) ?
			   EFA_DGRAM_CONNID :
			   efa_generate_rdm_connid();
	err = efa_base_ep_modify_qp_rst2rts(base_ep, qp);
	if (err)
		goto err_destroy_qp;

	qp->qp_num = qp->ibv_qp->qp_num;
	base_ep->qp = qp;
	qp->base_ep = base_ep;
	domain->qp_table[base_ep->qp->qp_num & domain->qp_table_sz_m1] =
		base_ep->qp;
	EFA_INFO(FI_LOG_EP_CTRL, "%s(): create QP %d qkey: %d\n", __func__,
		 qp->qp_num, qp->qkey);

	return 0;

err_destroy_qp:
	ibv_destroy_qp(qp->ibv_qp);
err_free_qp:
	free(qp);

	return err;
}

/* efa_base_ep_create_self_ah() create an address handler for
 * an EP's own address. The address handler is used by
 * an EP to read from itself. It is used to
 * copy data from host memory to GPU memory.
 */
static inline
int efa_base_ep_create_self_ah(struct efa_base_ep *base_ep, struct ibv_pd *ibv_pd)
{
	struct ibv_ah_attr ah_attr;
	struct efa_ep_addr *self_addr;

	self_addr = (struct efa_ep_addr *)base_ep->src_addr;

	memset(&ah_attr, 0, sizeof(ah_attr));
	ah_attr.port_num = 1;
	ah_attr.is_global = 1;
	memcpy(ah_attr.grh.dgid.raw, self_addr->raw, sizeof(self_addr->raw));
	base_ep->self_ah = ibv_create_ah(ibv_pd, &ah_attr);
	return base_ep->self_ah ? 0 : -FI_EINVAL;
}

int efa_base_ep_enable(struct efa_base_ep *base_ep,
		       struct ibv_qp_init_attr_ex *attr_ex)
{
	int err;

	err = efa_base_ep_create_qp_ex(base_ep, attr_ex);
	if (err)
		return err;

	err = efa_base_ep_create_self_ah(base_ep, base_ep->domain->ibv_pd);
	if (err) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Endpoint cannot create ah for its own address\n");
		efa_base_ep_destruct_qp(base_ep);
	}

	return err;
}

int efa_base_ep_construct(struct efa_base_ep *base_ep, struct fi_info *info)
{
	base_ep->info = fi_dupinfo(info);
	if (!base_ep->info) {
		EFA_WARN(FI_LOG_EP_CTRL, "fi_dupinfo() failed for base_ep->info!\n");
		return -FI_ENOMEM;
	}

	if (info->src_addr) {
		base_ep->src_addr = (void *)calloc(1, EFA_EP_ADDR_LEN);
		if (!base_ep->src_addr) {
			EFA_WARN(FI_LOG_EP_CTRL, "calloc() failed for base_ep->scr_addr!\n");
			fi_freeinfo(base_ep->info);
			return -FI_ENOMEM;
		}

		memcpy(base_ep->src_addr, info->src_addr, info->src_addrlen);
	}

	base_ep->rnr_retry = rxr_env.rnr_retry;

	base_ep->xmit_more_wr_tail = &base_ep->xmit_more_wr_head;
	base_ep->recv_more_wr_tail = &base_ep->recv_more_wr_head;

	return 0;
}

int efa_base_ep_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct efa_base_ep *base_ep;
	struct efa_ep_addr *ep_addr;

	base_ep = container_of(fid, struct efa_base_ep, util_ep.ep_fid.fid);

	char str[INET6_ADDRSTRLEN] = { 0 };

	ep_addr = (struct efa_ep_addr *)base_ep->src_addr;
	ep_addr->qpn = base_ep->qp->qp_num;
	ep_addr->pad = 0;
	ep_addr->qkey = base_ep->qp->qkey;

	inet_ntop(AF_INET6, ep_addr->raw, str, INET6_ADDRSTRLEN);

	EFA_INFO(FI_LOG_EP_CTRL, "EP addr: GID[%s] QP[%d] QKEY[%d] (length %zu)\n",
		 str, ep_addr->qpn, ep_addr->qkey, *addrlen);

	size_t len = MIN(*addrlen, EFA_EP_ADDR_LEN);

	memcpy(addr, ep_addr, len);
	*addrlen = EFA_EP_ADDR_LEN;

	return (len == EFA_EP_ADDR_LEN) ? 0 : -FI_ETOOSMALL;
}
