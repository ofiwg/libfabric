/*
 * Copyright (c) 2016 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015-2016 Cray Inc. All rights reserved.
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

/*
 * Endpoint common code
 */
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gnix.h"
#include "gnix_cm_nic.h"
#include "gnix_ep.h"
#include "gnix_vc.h"
#include "gnix_util.h"
#include "gnix_msg.h"

/******************************************************************************
 * Forward declaration for ops structures.
 ******************************************************************************/

static struct fi_ops gnix_sep_fi_ops;
static struct fi_ops_ep gnix_sep_ops;
/*
static struct fi_ops gnix_tx_fi_ops;
static struct fi_ops_ep gnix_tx_ops;
*/
static struct fi_ops_cm gnix_sep_rxtx_cm_ops;
static struct fi_ops_msg gnix_sep_msg_ops;
static struct fi_ops_rma gnix_sep_rma_ops;
static struct fi_ops_tagged gnix_sep_tagged_ops;
static struct fi_ops_atomic gnix_sep_atomic_ops;

/*******************************************************************************
 * SEP(EP) OPS API function implementations.
 ******************************************************************************/
/* TODO:
	initialize capabilities for tx_priv?
	initialize attr?
*/

static void __trx_destruct(void *obj)
{
	int __attribute__((unused)) ret;
	struct gnix_fid_trx *trx = (struct gnix_fid_trx *) obj;
	struct gnix_fid_ep *ep_priv;
	struct gnix_fid_sep *sep_priv;
	struct fid_domain *domain;
	int refs_held;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	ep_priv = trx->ep;
	assert(ep_priv != NULL);
	sep_priv = trx->sep;
	assert(sep_priv != NULL);
	domain = sep_priv->domain;
	assert(domain != NULL);

	refs_held = _gnix_ref_put(ep_priv);
	if (refs_held == 0)
		_gnix_ref_put(sep_priv->cm_nic);
	_gnix_ref_put(sep_priv);

	free(trx);
}

static int gnix_sep_tx_ctx(struct fid_ep *sep, int index,
			   struct fi_tx_attr *attr,
			   struct fid_ep **tx_ep, void *context)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_sep *sep_priv;
	struct gnix_fid_ep *ep_priv = NULL;
	struct gnix_fid_trx *tx_priv = NULL;
	struct fid_ep *ep_ptr;
	struct gnix_ep_attr ep_attr = {0};

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	sep_priv = container_of(sep, struct gnix_fid_sep, ep_fid);

	if (!sep_priv) {
		GNIX_WARN(FI_LOG_EP_CTRL, "endpoint is not initialized\n");
		return -FI_EINVAL;
	}

	if ((sep_priv->ep_fid.fid.fclass != FI_CLASS_SEP) ||
		(index >= sep_priv->info->ep_attr->tx_ctx_cnt))
		return -FI_EINVAL;

	/*
	 * check to see if the tx context was already
	 * allocated
	 */

	fastlock_acquire(&sep_priv->sep_lock);

	if (sep_priv->tx_ep_table[index] != NULL) {
		ret = -FI_EBUSY;
		goto err;
	}

	tx_priv = calloc(1, sizeof(struct gnix_fid_trx));
	if (!tx_priv) {
		ret = -FI_ENOMEM;
		goto err;
	}

	tx_priv->ep_fid.fid.fclass = FI_CLASS_TX_CTX;
	tx_priv->ep_fid.fid.context = context;
	tx_priv->ep_fid.fid.ops = &gnix_sep_fi_ops;
	tx_priv->ep_fid.ops = &gnix_sep_ops;
	tx_priv->ep_fid.msg = &gnix_sep_msg_ops;
	tx_priv->ep_fid.rma = &gnix_sep_rma_ops;
	tx_priv->ep_fid.tagged = &gnix_sep_tagged_ops;
	tx_priv->ep_fid.atomic = &gnix_sep_atomic_ops;
	tx_priv->ep_fid.cm = &gnix_sep_rxtx_cm_ops;

	/* if an EP already allocated for this index, use it */
	if (sep_priv->ep_table[index] != NULL) {
		ep_priv = container_of(sep_priv->ep_table[index],
				       struct gnix_fid_ep, ep_fid);
		sep_priv->tx_ep_table[index] = sep_priv->ep_table[index];
		_gnix_ref_get(ep_priv);
	} else {

		/*
		 * allocate the underlying gnix_fid_ep struct
		 */

		ep_attr.use_cdm_id = true;
		ep_attr.cdm_id = sep_priv->cdm_id_base + index;
		ep_attr.cm_nic = sep_priv->cm_nic;
		ep_attr.cm_ops = &gnix_sep_rxtx_cm_ops;
		/* TODO: clean up this cm_nic */
		_gnix_ref_get(sep_priv->cm_nic);
		ret = _gnix_ep_alloc(sep_priv->domain,
				     sep_priv->info,
				     &ep_attr,
				     &ep_ptr, context);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "gnix_ep_alloc returned %s\n",
				  fi_strerror(-ret));
			goto err;
		}

		sep_priv->ep_table[index] = ep_ptr;
		sep_priv->tx_ep_table[index] = ep_ptr;
		ep_priv = container_of(ep_ptr, struct gnix_fid_ep, ep_fid);
	}

	_gnix_ref_init(&tx_priv->ref_cnt, 1, __trx_destruct);
	tx_priv->ep = ep_priv;
	tx_priv->sep = sep_priv;
	_gnix_ref_get(sep_priv);
	tx_priv->caps = ep_priv->caps;
	*tx_ep = &tx_priv->ep_fid;
err:
	fastlock_release(&sep_priv->sep_lock);

	return ret;
}

static int gnix_sep_rx_ctx(struct fid_ep *sep, int index,
			   struct fi_rx_attr *attr,
			   struct fid_ep **rx_ep, void *context)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_sep *sep_priv;
	struct gnix_fid_ep *ep_priv = NULL;
	struct gnix_fid_trx *rx_priv = NULL;
	struct fid_ep *ep_ptr;
	struct gnix_ep_attr ep_attr = {0};

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	sep_priv = container_of(sep, struct gnix_fid_sep, ep_fid);

	if (!sep_priv) {
		GNIX_WARN(FI_LOG_EP_CTRL, "endpoint is not initialized\n");
		return -FI_EINVAL;
	}

	if ((sep_priv->ep_fid.fid.fclass != FI_CLASS_SEP) ||
		(index >= sep_priv->info->ep_attr->rx_ctx_cnt))
		return -FI_EINVAL;

	/*
	 * check to see if the rx context was already
	 * allocated
	 */

	fastlock_acquire(&sep_priv->sep_lock);

	if (sep_priv->rx_ep_table[index] != NULL) {
		ret = -FI_EBUSY;
		goto err;
	}

	rx_priv = calloc(1, sizeof(struct gnix_fid_trx));
	if (!rx_priv) {
		ret = -FI_ENOMEM;
		goto err;
	}

	rx_priv->ep_fid.fid.fclass = FI_CLASS_RX_CTX;
	rx_priv->ep_fid.fid.context = context;
	rx_priv->ep_fid.fid.ops = &gnix_sep_fi_ops;
	rx_priv->ep_fid.ops = &gnix_sep_ops;
	rx_priv->ep_fid.msg = &gnix_sep_msg_ops;
	rx_priv->ep_fid.rma = &gnix_sep_rma_ops;
	rx_priv->ep_fid.tagged = &gnix_sep_tagged_ops;
	rx_priv->ep_fid.atomic = &gnix_sep_atomic_ops;
	rx_priv->ep_fid.cm = &gnix_sep_rxtx_cm_ops;

	/* if an EP already allocated for this index, use it */
	if (sep_priv->ep_table[index] != NULL) {
		ep_priv = container_of(sep_priv->ep_table[index],
				       struct gnix_fid_ep, ep_fid);
		sep_priv->rx_ep_table[index] = sep_priv->ep_table[index];
		_gnix_ref_get(ep_priv);
	} else {

		/*
		 * compute cdm_id and allocate an EP.
		 */

		ep_attr.use_cdm_id = true;
		ep_attr.cdm_id = sep_priv->cdm_id_base + index;
		ep_attr.cm_nic = sep_priv->cm_nic;
		ep_attr.cm_ops = &gnix_sep_rxtx_cm_ops;
		_gnix_ref_get(sep_priv->cm_nic);
		ret = _gnix_ep_alloc(sep_priv->domain,
				     sep_priv->info,
				     &ep_attr,
				     &ep_ptr, context);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "gnix_ep_alloc returned %s\n",
				  fi_strerror(-ret));
			goto err;
		}

		sep_priv->ep_table[index] = ep_ptr;
		sep_priv->rx_ep_table[index] = ep_ptr;
		ep_priv = container_of(ep_ptr, struct gnix_fid_ep, ep_fid);
	}

	_gnix_ref_init(&rx_priv->ref_cnt, 1, __trx_destruct);
	rx_priv->ep = ep_priv;
	rx_priv->sep = sep_priv;
	_gnix_ref_get(sep_priv);
	rx_priv->caps = ep_priv->caps;
	*rx_ep = &rx_priv->ep_fid;
err:
	fastlock_release(&sep_priv->sep_lock);

	return ret;
}

static int gnix_sep_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
	int i, ret;
	struct gnix_fid_ep  *ep;
	struct gnix_fid_av  *av;
	struct gnix_fid_sep *sep;
	struct gnix_fid_trx *trx_priv;
	struct gnix_fid_domain *domain_priv;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	switch (fid->fclass) {
	case FI_CLASS_SEP:
		break;
	case FI_CLASS_TX_CTX:
	case FI_CLASS_RX_CTX:
		trx_priv = container_of(fid, struct gnix_fid_trx, ep_fid);
		return gnix_ep_bind(&trx_priv->ep->ep_fid.fid, bfid, flags);
	default:
		return -FI_ENOSYS;
	}

	sep = container_of(fid, struct gnix_fid_sep, ep_fid);
	domain_priv = container_of(sep->domain, struct gnix_fid_domain,
				   domain_fid);

	ret = ofi_ep_bind_valid(&gnix_prov, bfid, flags);
	if (ret)
		return ret;

	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct gnix_fid_av, av_fid.fid);
		if (domain_priv != av->domain) {
			return -FI_EINVAL;
		}

		/* We currently only support FI_AV_MAP */
		if (av->type != FI_AV_MAP) {
			return -FI_EINVAL;
		}

		for (i = 0; i < sep->info->ep_attr->tx_ctx_cnt; i++) {
			ep = container_of(sep->tx_ep_table[i],
					  struct gnix_fid_ep, ep_fid);
			if (ep == NULL) {
				return -FI_EINVAL;
			}
			ep->av = av;
			_gnix_ep_init_vc(ep);
			_gnix_ref_get(ep->av);
		}

		for (i = 0; i < sep->info->ep_attr->rx_ctx_cnt; i++) {
			ep = container_of(sep->rx_ep_table[i],
					  struct gnix_fid_ep, ep_fid);
			if (ep == NULL) {
				return -FI_EINVAL;
			}
			ep->av = av;
			_gnix_ep_init_vc(ep);
			_gnix_ref_get(ep->av);
		}

		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

/*******************************************************************************
 * Base SEP API function implementations.
 ******************************************************************************/
static int gnix_sep_control(fid_t fid, int command, void *arg)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_ep *ep;
	struct gnix_fid_trx *trx_priv;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	switch (fid->fclass) {
	case FI_CLASS_SEP:
		/* nothing to do for scalable endpoints */
		return FI_SUCCESS;
	case FI_CLASS_TX_CTX:
	case FI_CLASS_RX_CTX:
		trx_priv = container_of(fid, struct gnix_fid_trx, ep_fid);
		ep = trx_priv->ep;
		break;
	default:
		return -FI_EINVAL;
	}

	if (!ep) {
		return -FI_EINVAL;
	}

	switch (command) {
	case FI_ENABLE:
		if (GNIX_EP_RDM_DGM(ep->type)) {
			if (ep->cm_nic == NULL) {
				ret = -FI_EOPBADSTATE;
				goto err;
			}
			ret = _gnix_vc_cm_init(ep->cm_nic);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_CTRL,
				     "_gnix_vc_cm_nic_init call returned %d\n",
					ret);
				goto err;
			}
			ret = _gnix_cm_nic_enable(ep->cm_nic);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_CTRL,
				     "_gnix_cm_nic_enable call returned %d\n",
					ret);
				goto err;
			}
			if (ep->send_cq)
				ep->tx_enabled = true;
			if (ep->recv_cq)
				ep->rx_enabled = true;
		}

		break;
	case FI_GETFIDFLAG:
	case FI_SETFIDFLAG:
	case FI_ALIAS:
	default:
		return -FI_ENOSYS;
	}
err:
	return ret;
}

static void __sep_destruct(void *obj)
{
	int i;
	struct fid_domain *domain;
	struct gnix_fid_ep *ep;
	struct gnix_fid_domain *domain_priv;
	struct gnix_fid_sep *sep = (struct gnix_fid_sep *) obj;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	domain = sep->domain;
	assert(domain != NULL);
	domain_priv = container_of(domain, struct gnix_fid_domain, domain_fid);

	_gnix_ref_put(domain_priv);

	if (sep->ep_table) {
		for (i = 0; i < sep->info->ep_attr->tx_ctx_cnt; i++) {
			ep = container_of(sep->ep_table[i],
					  struct gnix_fid_ep, ep_fid);
			if (ep == NULL) {
				continue;
			}

			if (ep->av) {
				_gnix_ref_put(ep->av);
			}
		}

		free(sep->ep_table);
	}

	if (sep->tx_ep_table)
		free(sep->tx_ep_table);
	if (sep->rx_ep_table)
		free(sep->rx_ep_table);
	fi_freeinfo(sep->info);
	free(sep);
}

static int gnix_sep_close(fid_t fid)
{
	int ret = FI_SUCCESS;
	int refs_held;
	struct gnix_fid_sep *sep;
	struct gnix_fid_trx *trx_priv;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	switch (fid->fclass) {
	case FI_CLASS_SEP:
		sep = container_of(fid, struct gnix_fid_sep, ep_fid.fid);
		refs_held = _gnix_ref_put(sep);
		if (refs_held) {
			GNIX_INFO(FI_LOG_CQ, "failed to fully close sep due to"
				  " lingering references. refs=%i sep=%p\n",
				  refs_held, sep);
		}
		break;
	case FI_CLASS_TX_CTX:
	case FI_CLASS_RX_CTX:
		trx_priv = container_of(fid, struct gnix_fid_trx, ep_fid);
		_gnix_ref_put(trx_priv);
		break;
	default:
		return -FI_EINVAL;
	}

	return ret;
}

int gnix_sep_open(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **sep, void *context)
{
	struct gnix_fid_sep *sep_priv = NULL;
	struct gnix_fid_domain *domain_priv = NULL;
	int ret = FI_SUCCESS;
	int n_ids = GNIX_SEP_MAX_CNT;
	uint32_t cdm_id, cdm_id_base;
	struct gnix_ep_name *name;
	uint32_t name_type = GNIX_EPN_TYPE_UNBOUND;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	if ((domain == NULL) || (info == NULL) || (sep == NULL) ||
	    (info->ep_attr == NULL))
		return -FI_EINVAL;

	if (!GNIX_EP_RDM_DGM(info->ep_attr->type))
		return -FI_ENOSYS;

	/*
	 * check limits for rx and tx ctx's
	 */

	if ((info->ep_attr->tx_ctx_cnt > n_ids) ||
	    (info->ep_attr->rx_ctx_cnt > n_ids))
		return -FI_EINVAL;

	n_ids = MAX(info->ep_attr->tx_ctx_cnt, info->ep_attr->rx_ctx_cnt);

	domain_priv = container_of(domain, struct gnix_fid_domain, domain_fid);

	sep_priv = calloc(1, sizeof(*sep_priv));
	if (!sep_priv)
		return -FI_ENOMEM;

	sep_priv->type = info->ep_attr->type;
	sep_priv->ep_fid.fid.fclass = FI_CLASS_SEP;
	sep_priv->ep_fid.fid.context = context;

	sep_priv->ep_fid.fid.ops = &gnix_sep_fi_ops;
	sep_priv->ep_fid.ops = &gnix_sep_ops;
	sep_priv->ep_fid.cm = &gnix_cm_ops;
	sep_priv->domain = domain;

	sep_priv->info = fi_dupinfo(info);
	if (!sep_priv->info) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			    "fi_dupinfo NULL\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	_gnix_ref_init(&sep_priv->ref_cnt, 1, __sep_destruct);

	sep_priv->caps = info->caps & GNIX_EP_RDM_PRIMARY_CAPS;

	sep_priv->op_flags = info->tx_attr->op_flags;
	sep_priv->op_flags |= info->rx_attr->op_flags;
	sep_priv->op_flags &= GNIX_EP_OP_FLAGS;

	sep_priv->ep_table = calloc(n_ids, sizeof(struct gnix_fid_ep *));
	if (sep_priv->ep_table == NULL) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			    "call returned NULL\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	sep_priv->tx_ep_table = calloc(n_ids, sizeof(struct gnix_fid_ep *));
	if (sep_priv->tx_ep_table == NULL) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			    "call returned NULL\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	sep_priv->rx_ep_table = calloc(n_ids, sizeof(struct gnix_fid_ep *));
	if (sep_priv->rx_ep_table == NULL) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			    "call returned NULL\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	/*
	 * allocate a block of cm nic ids for both tx/rx ctx - first
	 * checking to see if the application has specified a base
	 * via a node/service option to fi_getinfo
	 */

	if ((info->src_addr != NULL) &&
		info->src_addrlen == sizeof(struct gnix_ep_name)) {
		name = (struct gnix_ep_name *)info->src_addr;

		if (name->name_type & GNIX_EPN_TYPE_BOUND) {
			cdm_id_base = name->gnix_addr.cdm_id;
			name_type = name->name_type;
		}
	}

	name_type |= GNIX_EPN_TYPE_SEP;

	cdm_id = (name_type & GNIX_EPN_TYPE_UNBOUND) ? -1 : cdm_id_base;

	ret = _gnix_get_new_cdm_id_set(domain_priv, n_ids, &cdm_id);
	if (ret != FI_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "_gnix_get_new_cdm_id_set call returned %s\n",
			  fi_strerror(-ret));
		goto err;
	}

	sep_priv->cdm_id_base = cdm_id;

	/*
	 * allocate cm_nic for this SEP
	 */
	ret = _gnix_cm_nic_alloc(domain_priv,
				 info,
				 cdm_id,
				 &sep_priv->cm_nic);
	if (ret != FI_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			    "gnix_cm_nic_alloc call returned %s\n",
			     fi_strerror(-ret));
		goto err;
	}

	/*
	 * ep name of SEP is the same as the cm_nic
	 * since there's a one-to-one relationship
	 * between a given SEP and its cm_nic.
	 */
	sep_priv->my_name = sep_priv->cm_nic->my_name;
	sep_priv->my_name.cm_nic_cdm_id =
				sep_priv->cm_nic->my_name.gnix_addr.cdm_id;
	sep_priv->my_name.rx_ctx_cnt = info->ep_attr->rx_ctx_cnt;
	sep_priv->my_name.name_type = name_type;

	fastlock_init(&sep_priv->sep_lock);
	_gnix_ref_get(domain_priv);

	*sep = &sep_priv->ep_fid;
	return ret;

err:
	if (sep_priv->ep_table)
		free(sep_priv->ep_table);
	if (sep_priv->tx_ep_table)
		free(sep_priv->tx_ep_table);
	if (sep_priv->rx_ep_table)
		free(sep_priv->rx_ep_table);
	if (sep_priv)
		free(sep_priv);
	return ret;

}

/*******************************************************************************
ssize_t (*recv)(struct fid_ep *ep, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context);
ssize_t (*send)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, void *context);
 ******************************************************************************/

/*
 * TODO: need to define the other msg/rma/amo methods for tx/rx contexts
 */

DIRECT_FN STATIC ssize_t gnix_sep_recv(struct fid_ep *ep, void *buf,
				       size_t len, void *desc,
				       fi_addr_t src_addr, void *context)
{
	struct gnix_fid_trx *rx_ep = container_of(ep, struct gnix_fid_trx,
						  ep_fid);

	return _ep_recv(&rx_ep->ep->ep_fid, buf, len, desc, src_addr,
			context, 0, 0, 0);
}

DIRECT_FN STATIC ssize_t gnix_sep_recvv(struct fid_ep *ep,
					const struct iovec *iov,
					void **desc, size_t count,
					fi_addr_t src_addr,
					void *context)
{
	struct gnix_fid_trx *rx_ep = container_of(ep, struct gnix_fid_trx,
						  ep_fid);

	return _ep_recvv(&rx_ep->ep->ep_fid, iov, desc, count, src_addr,
			 context, 0, 0, 0);
}

DIRECT_FN STATIC ssize_t gnix_sep_recvmsg(struct fid_ep *ep,
					 const struct fi_msg *msg,
					 uint64_t flags)
{
	struct gnix_fid_trx *rx_ep = container_of(ep, struct gnix_fid_trx,
						  ep_fid);

	return _ep_recvmsg(&rx_ep->ep->ep_fid, msg, flags & GNIX_RECVMSG_FLAGS,
			   0, 0);
}

DIRECT_FN STATIC ssize_t gnix_sep_send(struct fid_ep *ep, const void *buf,
				       size_t len, void *desc,
				       fi_addr_t dest_addr, void *context)
{
	struct gnix_fid_trx *tx_ep = container_of(ep, struct gnix_fid_trx,
						  ep_fid);

	return _ep_send(&tx_ep->ep->ep_fid, buf, len, desc, dest_addr,
			context, 0, 0);
}

DIRECT_FN ssize_t gnix_sep_sendv(struct fid_ep *ep,
				 const struct iovec *iov,
				 void **desc, size_t count,
				 fi_addr_t dest_addr,
				 void *context)
{
	struct gnix_fid_trx *tx_ep = container_of(ep, struct gnix_fid_trx,
						  ep_fid);

	return _ep_sendv(&tx_ep->ep->ep_fid, iov, desc, count, dest_addr,
			 context, 0, 0);
}

DIRECT_FN ssize_t gnix_sep_sendmsg(struct fid_ep *ep,
				  const struct fi_msg *msg,
				  uint64_t flags)
{
	struct gnix_fid_trx *tx_ep = container_of(ep, struct gnix_fid_trx,
						  ep_fid);

	return _ep_sendmsg(&tx_ep->ep->ep_fid, msg,
			   flags & GNIX_SENDMSG_FLAGS, 0);
}

DIRECT_FN ssize_t gnix_sep_msg_inject(struct fid_ep *ep, const void *buf,
				      size_t len, fi_addr_t dest_addr)
{
	struct gnix_fid_trx *tx_ep = container_of(ep, struct gnix_fid_trx,
						  ep_fid);

	return _ep_inject(&tx_ep->ep->ep_fid, buf, len, 0, dest_addr, 0, 0);
}

DIRECT_FN ssize_t gnix_sep_senddata(struct fid_ep *ep, const void *buf,
				    size_t len, void *desc, uint64_t data,
				    fi_addr_t dest_addr, void *context)
{
	struct gnix_fid_trx *tx_ep = container_of(ep, struct gnix_fid_trx,
						  ep_fid);

	return _ep_senddata(&tx_ep->ep->ep_fid, buf, len, desc, data,
			    dest_addr, context, 0, 0);
}

DIRECT_FN ssize_t
gnix_sep_msg_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr)
{
	uint64_t flags;
	struct gnix_fid_trx *tx_ep;

	if (!ep) {
		return -FI_EINVAL;
	}

	tx_ep = container_of(ep, struct gnix_fid_trx, ep_fid);
	assert(GNIX_EP_RDM_DGM_MSG(tx_ep->ep->type));

	flags = tx_ep->ep->op_flags | FI_INJECT | FI_REMOTE_CQ_DATA |
		GNIX_SUPPRESS_COMPLETION;

	return _gnix_send(tx_ep->ep, (uint64_t)buf, len, NULL, dest_addr,
			  NULL, flags, data, 0);
}

/*******************************************************************************
 * FI_OPS_* data structures.
 ******************************************************************************/

static struct fi_ops gnix_sep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = gnix_sep_close,
	.bind = gnix_sep_bind,
	.control = gnix_sep_control,
	.ops_open = fi_no_ops_open
};

static struct fi_ops_ep gnix_sep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = gnix_sep_tx_ctx,
	.rx_ctx = gnix_sep_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static struct fi_ops_msg gnix_sep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = gnix_sep_recv,
	.recvv = gnix_sep_recvv,
	.recvmsg = gnix_sep_recvmsg,
	.send = gnix_sep_send,
	.sendv = gnix_sep_sendv,
	.sendmsg = gnix_sep_sendmsg,
	.inject = gnix_sep_msg_inject,
	.senddata = gnix_sep_senddata,
	.injectdata = gnix_sep_msg_injectdata,
};

static struct fi_ops_rma gnix_sep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = NULL,
	.readv = NULL,
	.readmsg = NULL,
	.write = NULL,
	.writev = NULL,
	.writemsg = NULL,
	.inject = NULL,
	.writedata = NULL,
	.injectdata = NULL,
};

static struct fi_ops_tagged gnix_sep_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = NULL,
	.recvv = NULL,
	.recvmsg = NULL,
	.send = NULL,
	.sendv = NULL,
	.sendmsg = NULL,
	.inject = NULL,
	.senddata = NULL,
	.injectdata = NULL,
};

static struct fi_ops_atomic gnix_sep_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = NULL,
	.writev = NULL,
	.writemsg = NULL,
	.inject = NULL,
	.readwrite = NULL,
	.readwritev = NULL,
	.readwritemsg = NULL,
	.compwrite = NULL,
	.compwritev = NULL,
	.compwritemsg = NULL,
	.writevalid = NULL,
	.readwritevalid = NULL,
	.compwritevalid = NULL,
};

/*
 * rx/tx contexts don't do any connection management,
 * nor does the underlying gnix_fid_ep struct
 */
static struct fi_ops_cm gnix_sep_rxtx_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = fi_no_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
};
