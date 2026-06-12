/*
 * Copyright (c) Intel Corporation, Inc.  All rights reserved.
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

#include "verbs_ofi.h"


int vrb_copy_addr(void *dst_addr, size_t *dst_addrlen, void *src_addr)
{
	size_t src_addrlen = ofi_sizeofaddr(src_addr);

	if (*dst_addrlen == 0) {
		*dst_addrlen = src_addrlen;
		return -FI_ETOOSMALL;
	}

	if (*dst_addrlen < src_addrlen) {
		memcpy(dst_addr, src_addr, *dst_addrlen);
	} else {
		memcpy(dst_addr, src_addr, src_addrlen);
	}
	*dst_addrlen = src_addrlen;
	return 0;
}

static int vrb_msg_ep_setname(fid_t ep_fid, void *addr, size_t addrlen)
{
	struct vrb_ep *ep = container_of(ep_fid, struct vrb_ep, util_ep.ep_fid);

	if (addrlen != ep->info_attr.src_addrlen) {
		VRB_INFO(FI_LOG_EP_CTRL, "addrlen expected: %zu, got: %zu.\n",
			 ep->info_attr.src_addrlen, addrlen);
		return -FI_EINVAL;
	}

	return ep->cm_ops->ep_setname(ep, addr, addrlen);
}

static int vrb_msg_ep_getname(fid_t ep_fid, void *addr, size_t *addrlen)
{
	struct vrb_ep *ep = container_of(ep_fid, struct vrb_ep, util_ep.ep_fid);
	return ep->cm_ops->ep_getname(ep, addr, addrlen);
}

static int vrb_msg_ep_getpeer(struct fid_ep *ep_fid, void *addr, size_t *addrlen)
{
	struct vrb_ep *ep = container_of(ep_fid, struct vrb_ep, util_ep.ep_fid);
	return ep->cm_ops->getpeer(ep, addr, addrlen);
}

void vrb_msg_ep_prepare_cm_data(const void *param, size_t param_size,
				struct vrb_cm_data_hdr *cm_hdr)
{
	cm_hdr->size = (uint8_t)param_size;
	memcpy(cm_hdr->data, param, cm_hdr->size);
}

static int vrb_msg_ep_connect(struct fid_ep *ep_fid, const void *addr,
			      const void *param, size_t paramlen)
{
	struct vrb_ep *ep = container_of(ep_fid, struct vrb_ep, util_ep.ep_fid);
	int ret = 0;

	if (ep->profile)
		vrb_prof_st_start(ep->profile, ofi_gettime_ns());

	vrb_prof_func_start(__func__);

	if (OFI_UNLIKELY(paramlen > VERBS_CM_DATA_SIZE))
		return -FI_EINVAL;

	if (!ep->ibv_qp) {
		ret = fi_control(&ep_fid->fid, FI_ENABLE, NULL);
		if (ret) {
			VRB_WARN_ERR(FI_LOG_EP_CTRL, "fi_control", ret);
			return ret;
		}
	}

	ret = ep->cm_ops->connect(ep, addr, param, paramlen);
	vrb_prof_func_end(__func__);
	return ret;
}

static int
vrb_msg_ep_accept(struct fid_ep *ep_fid, const void *param, size_t paramlen)
{
	int ret;
	struct vrb_ep *ep = container_of(ep_fid, struct vrb_ep, util_ep.ep_fid);

	if (OFI_UNLIKELY(paramlen > VERBS_CM_DATA_SIZE))
		return -FI_EINVAL;

	if (!ep->ibv_qp) {
		ret = fi_control(&ep_fid->fid, FI_ENABLE, NULL);
		if (ret) {
			VRB_WARN_ERR(FI_LOG_EP_CTRL, "fi_control", ret);
			return ret;
		}
	}

	ret = ep->cm_ops->accept(ep, param, paramlen);

	return ret;
}

static int vrb_msg_alloc_xrc_params(void **adjusted_param,
				       const void *param, size_t *paramlen)
{
	struct vrb_xrc_cm_data *cm_data;
	size_t cm_datalen = sizeof(*cm_data) + *paramlen;

	*adjusted_param = NULL;

	if (cm_datalen > VRB_CM_DATA_SIZE) {
		VRB_WARN(FI_LOG_EP_CTRL, "XRC CM data overflow %zu\n",
			   cm_datalen);
		return -FI_EINVAL;
	}

	cm_data = malloc(cm_datalen);
	if (!cm_data) {
		VRB_WARN(FI_LOG_EP_CTRL, "Unable to allocate XRC CM data\n");
		return -FI_ENOMEM;
	}

	if (*paramlen)
		memcpy((cm_data + 1), param, *paramlen);

	*paramlen = cm_datalen;
	*adjusted_param = cm_data;
	return FI_SUCCESS;
}

static int
vrb_msg_xrc_ep_reject(struct vrb_connreq *connreq,
			 const void *param, size_t paramlen)
{
	struct vrb_xrc_cm_data *cm_data;
	int ret;

	ret = vrb_msg_alloc_xrc_params((void **)&cm_data, param, &paramlen);
	if (ret)
		return ret;

	vrb_set_xrc_cm_data(cm_data, connreq->xrc.is_reciprocal,
			       connreq->xrc.conn_tag, connreq->xrc.port, 0, 0);
	vrb_prof_func_start("rdma_reject");
	ret = rdma_reject(connreq->id, cm_data,
			  (uint8_t) paramlen) ? -errno : 0;
	vrb_prof_func_end("rdma_reject");

	if (rdma_destroy_id(connreq->id))
		VRB_WARN_ERR(FI_LOG_EP_CTRL, "rdma_destroy_id", -errno);
	connreq->id = NULL;
	free(cm_data);
	return ret;
}

static int
vrb_msg_ep_reject(struct fid_pep *pep, fid_t handle,
		     const void *param, size_t paramlen)
{
	struct vrb_connreq *connreq =
		container_of(handle, struct vrb_connreq, handle);
	struct vrb_cm_data_hdr *cm_hdr;
	struct vrb_pep *_pep = container_of(pep, struct vrb_pep,
					       pep_fid);
	int ret;

	if (OFI_UNLIKELY(paramlen > VERBS_CM_DATA_SIZE))
		return -FI_EINVAL;

	cm_hdr = alloca(sizeof(*cm_hdr) + paramlen);
	vrb_msg_ep_prepare_cm_data(param, paramlen, cm_hdr);

	ofi_mutex_lock(&_pep->eq->event_lock);
	if (connreq->is_xrc) {
		ret = vrb_msg_xrc_ep_reject(connreq, cm_hdr,
				(uint8_t)(sizeof(*cm_hdr) + paramlen));
	} else {
		ret = _pep->cm_ops->reject(_pep, connreq, cm_hdr,
					   sizeof(*cm_hdr) + paramlen);
	}
	if (!ret && _pep->profile)
		vrb_prof_cntr_inc(_pep->profile, FI_VAR_CONN_REJECT);

	ofi_mutex_unlock(&_pep->eq->event_lock);
	if (ret)
		VRB_WARN_ERR(FI_LOG_EP_CTRL, "rdma_reject", ret);

	free(connreq);
	return ret;
}

static int vrb_msg_ep_shutdown(struct fid_ep *ep_fid, uint64_t flags)
{
	struct vrb_ep *ep = container_of(ep_fid, struct vrb_ep, util_ep.ep_fid);

	ep->cm_ops->shutdown(ep, flags);

	return 0;
}

struct fi_ops_cm vrb_msg_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = vrb_msg_ep_setname,
	.getname = vrb_msg_ep_getname,
	.getpeer = vrb_msg_ep_getpeer,
	.connect = vrb_msg_ep_connect,
	.listen = fi_no_listen,
	.accept = vrb_msg_ep_accept,
	.reject = fi_no_reject,
	.shutdown = vrb_msg_ep_shutdown,
	.join = fi_no_join,
};

static int
vrb_msg_xrc_cm_common_verify(struct vrb_xrc_ep *ep, size_t paramlen)
{
	int ret;

	if (!vrb_is_xrc_ep(&ep->base_ep)) {
		VRB_WARN(FI_LOG_EP_CTRL, "EP is not using XRC\n");
		return -FI_EINVAL;
	}

	if (!ep->srqn) {
		ret = fi_control(&ep->base_ep.util_ep.ep_fid.fid,
				 FI_ENABLE, NULL);
		if (ret)
			return ret;
	}

	if (OFI_UNLIKELY(paramlen > VERBS_CM_DATA_SIZE -
			 sizeof(struct vrb_xrc_cm_data)))
		return -FI_EINVAL;

	return FI_SUCCESS;
}

static int
vrb_msg_xrc_ep_connect(struct fid_ep *ep, const void *addr,
		   const void *param, size_t paramlen)
{
	void *adjusted_param;
	struct vrb_ep *_ep = container_of(ep, struct vrb_ep,
					     util_ep.ep_fid);
	struct vrb_xrc_ep *xrc_ep = container_of(_ep, struct vrb_xrc_ep,
						    base_ep);
	int ret;
	struct vrb_cm_data_hdr *cm_hdr;

	ret = vrb_msg_xrc_cm_common_verify(xrc_ep, paramlen);
	if (ret)
		return ret;

	cm_hdr = malloc(sizeof(*cm_hdr) + paramlen);
	if (!cm_hdr)
		return -FI_ENOMEM;

	vrb_msg_ep_prepare_cm_data(param, paramlen, cm_hdr);
	paramlen += sizeof(*cm_hdr);

	ret = vrb_msg_alloc_xrc_params(&adjusted_param, cm_hdr, &paramlen);
	if (ret) {
		free(cm_hdr);
		return ret;
	}

	xrc_ep->conn_setup = calloc(1, sizeof(*xrc_ep->conn_setup));
	if (!xrc_ep->conn_setup) {
		VRB_WARN(FI_LOG_EP_CTRL,
			   "Unable to allocate connection setup memory\n");
		free(adjusted_param);
		free(cm_hdr);
		return -FI_ENOMEM;
	}
	xrc_ep->conn_setup->conn_tag = VERBS_CONN_TAG_INVALID;

	ofi_mutex_lock(&xrc_ep->base_ep.eq->event_lock);
	ret = vrb_connect_xrc(xrc_ep, NULL, 0, adjusted_param, paramlen);
	ofi_mutex_unlock(&xrc_ep->base_ep.eq->event_lock);

	free(adjusted_param);
	free(cm_hdr);
	return ret;
}

static int
vrb_msg_xrc_ep_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	void *adjusted_param;
	struct vrb_ep *_ep =
		container_of(ep, struct vrb_ep, util_ep.ep_fid);
	struct vrb_xrc_ep *xrc_ep = container_of(_ep, struct vrb_xrc_ep,
						    base_ep);
	int ret;
	struct vrb_cm_data_hdr *cm_hdr;

	ret = vrb_msg_xrc_cm_common_verify(xrc_ep, paramlen);
	if (ret)
		return ret;

	cm_hdr = alloca(sizeof(*cm_hdr) + paramlen);
	vrb_msg_ep_prepare_cm_data(param, paramlen, cm_hdr);
	paramlen += sizeof(*cm_hdr);

	ret = vrb_msg_alloc_xrc_params(&adjusted_param, cm_hdr, &paramlen);
	if (ret)
		return ret;

	ofi_mutex_lock(&xrc_ep->base_ep.eq->event_lock);
	ret = vrb_accept_xrc(xrc_ep, 0, adjusted_param, paramlen);
	ofi_mutex_unlock(&xrc_ep->base_ep.eq->event_lock);

	free(adjusted_param);
	return ret;
}

struct fi_ops_cm vrb_msg_xrc_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = vrb_msg_ep_setname,
	.getname = vrb_msg_ep_getname,
	.getpeer = vrb_msg_ep_getpeer,
	.connect = vrb_msg_xrc_ep_connect,
	.listen = fi_no_listen,
	.accept = vrb_msg_xrc_ep_accept,
	.reject = fi_no_reject,
	.shutdown = vrb_msg_ep_shutdown,
	.join = fi_no_join,
};

static int vrb_pep_listen(struct fid_pep *pep_fid)
{
	struct vrb_pep *pep = container_of(pep_fid, struct vrb_pep, pep_fid);

	return pep->cm_ops->listen(pep);
}

static int vrb_pep_setname(fid_t pep_fid, void *addr, size_t addrlen)
{
	struct vrb_pep *pep =container_of(pep_fid, struct vrb_pep, pep_fid);

	return pep->cm_ops->setname(pep, addr, addrlen);
}

static int vrb_pep_getname(struct fid *fid, void *addr, size_t *addrlen)
{
	struct vrb_pep *pep = container_of(fid, struct vrb_pep, pep_fid);

	return pep->cm_ops->getname(pep, addr, addrlen);
}

static struct fi_ops_cm vrb_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = vrb_pep_setname,
	.getname = vrb_pep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = vrb_pep_listen,
	.accept = fi_no_accept,
	.reject = vrb_msg_ep_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

struct fi_ops_cm *vrb_pep_ops_cm(struct vrb_pep *pep)
{
	return &vrb_pep_cm_ops;
}
