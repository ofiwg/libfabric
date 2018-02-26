/*
 * Copyright (c) 2014-2018, Cisco Systems, Inc. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

#include <arpa/inet.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>
#include "ofi.h"
#include "ofi_enosys.h"
#include "ofi_util.h"

#include "usnic_direct.h"
#include "usdf.h"
#include "usdf_rdm.h"
#include "usdf_timer.h"
#include "usdf_poll.h"
#include "usdf_cm.h"

static int
usdf_domain_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
        struct usdf_domain *udp;

	USDF_TRACE_SYS(DOMAIN, "\n");

	if (flags & FI_REG_MR) {
		USDF_WARN_SYS(DOMAIN,
			"FI_REG_MR for EQs is not supported by the usnic provider");
		return -FI_EOPNOTSUPP;
	}

        udp = dom_fidtou(fid);

        switch (bfid->fclass) {
        case FI_CLASS_EQ:
                if (udp->dom_eq != NULL) {
                        return -FI_EINVAL;
                }
                udp->dom_eq = eq_fidtou(bfid);
                ofi_atomic_inc32(&udp->dom_eq->eq_refcnt);
                break;
        default:
                return -FI_EINVAL;
        }

        return 0;
}

static void
usdf_dom_rdc_free_data(struct usdf_domain *udp)
{
	struct usdf_rdm_connection *rdc;
	int i;

	if (udp->dom_rdc_hashtab != NULL) {

		pthread_spin_lock(&udp->dom_progress_lock);
		for (i = 0; i < USDF_RDM_HASH_SIZE; ++i) {
			rdc = udp->dom_rdc_hashtab[i];
			while (rdc != NULL) {
				usdf_timer_reset(udp->dom_fabric,
						rdc->dc_timer, 0);
				rdc = rdc->dc_hash_next;
			}
		}
		pthread_spin_unlock(&udp->dom_progress_lock);

		/* XXX probably want a timeout here... */
		while (ofi_atomic_get32(&udp->dom_rdc_free_cnt) <
		       (int)udp->dom_rdc_total) {
			pthread_yield();
		}

		free(udp->dom_rdc_hashtab);
		udp->dom_rdc_hashtab = NULL;
	}

	while (!SLIST_EMPTY(&udp->dom_rdc_free)) {
		rdc = SLIST_FIRST(&udp->dom_rdc_free);
		SLIST_REMOVE_HEAD(&udp->dom_rdc_free, dc_addr_link);
		usdf_timer_free(udp->dom_fabric, rdc->dc_timer);
		free(rdc);
	}
}

static int
usdf_dom_rdc_alloc_data(struct usdf_domain *udp)
{
	struct usdf_rdm_connection *rdc;
	int ret;
	int i;

	udp->dom_rdc_hashtab = calloc(USDF_RDM_HASH_SIZE,
			sizeof(*udp->dom_rdc_hashtab));
	if (udp->dom_rdc_hashtab == NULL) {
		return -FI_ENOMEM;
	}
	SLIST_INIT(&udp->dom_rdc_free);
	ofi_atomic_initialize32(&udp->dom_rdc_free_cnt, 0);
	for (i = 0; i < USDF_RDM_FREE_BLOCK; ++i) {
		rdc = calloc(1, sizeof(*rdc));
		if (rdc == NULL) {
			return -FI_ENOMEM;
		}
		ret = usdf_timer_alloc(usdf_rdm_rdc_timeout, rdc,
				&rdc->dc_timer);
		if (ret != 0) {
			free(rdc);
			return ret;
		}
		rdc->dc_flags = USDF_DCS_UNCONNECTED | USDF_DCF_NEW_RX;
		rdc->dc_next_rx_seq = 0;
		rdc->dc_next_tx_seq = 0;
		rdc->dc_last_rx_ack = rdc->dc_next_tx_seq - 1;
		TAILQ_INIT(&rdc->dc_wqe_posted);
		TAILQ_INIT(&rdc->dc_wqe_sent);
		SLIST_INSERT_HEAD(&udp->dom_rdc_free, rdc, dc_addr_link);
		ofi_atomic_inc32(&udp->dom_rdc_free_cnt);
	}
	udp->dom_rdc_total = USDF_RDM_FREE_BLOCK;
	return 0;
}

static int
usdf_domain_close(fid_t fid)
{
	struct usdf_domain *udp;
	int ret;

	USDF_TRACE_SYS(DOMAIN, "\n");

	udp = container_of(fid, struct usdf_domain, dom_fid.fid);
	if (ofi_atomic_get32(&udp->dom_refcnt) > 0) {
		return -FI_EBUSY;
	}

	if (udp->dom_dev != NULL) {
		ret = usd_close(udp->dom_dev);
		if (ret != 0) {
			return ret;
		}
	}
	usdf_dom_rdc_free_data(udp);

	if (udp->dom_eq != NULL) {
		ofi_atomic_dec32(&udp->dom_eq->eq_refcnt);
	}
	ofi_atomic_dec32(&udp->dom_fabric->fab_refcnt);
	LIST_REMOVE(udp, dom_link);
	fi_freeinfo(udp->dom_info);
	free(udp);

	return 0;
}

static struct fi_ops usdf_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = usdf_domain_close,
	.bind = usdf_domain_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_mr usdf_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = usdf_reg_mr,
	.regv = usdf_regv_mr,
	.regattr = usdf_regattr,
};

static struct fi_ops_domain usdf_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = usdf_av_open,
	.cq_open = usdf_cq_open,
	.endpoint = usdf_endpoint_open,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = usdf_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = usdf_query_atomic,
};

int
usdf_domain_open(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	struct usdf_fabric *fp;
	struct usdf_domain *udp;
	struct sockaddr_in *sin;
	size_t addrlen;
	int ret;
#if ENABLE_DEBUG
	char requested[INET_ADDRSTRLEN], actual[INET_ADDRSTRLEN];
#endif

	USDF_TRACE_SYS(DOMAIN, "\n");
	sin = NULL;

	fp = fab_fidtou(fabric);

	if (info->domain_attr != NULL) {
		/* No versioning information available here. */
		if (!usdf_domain_checkname(0, fp->fab_dev_attrs,
					   info->domain_attr->name)) {
			USDF_WARN_SYS(DOMAIN, "domain name mismatch\n");
			return -FI_ENODATA;
		}

		if (ofi_check_mr_mode(
			&usdf_ops, fabric->api_version,
			FI_MR_BASIC | FI_MR_ALLOCATED | FI_MR_LOCAL, info)) {
			/* the caller ignored our fi_getinfo results */
			USDF_WARN_SYS(DOMAIN, "MR mode (%d) not supported\n",
				      info->domain_attr->mr_mode);
			return -FI_ENODATA;
		}
	}

	udp = calloc(1, sizeof *udp);
	if (udp == NULL) {
		USDF_DBG("unable to alloc mem for domain\n");
		ret = -FI_ENOMEM;
		goto fail;
	}

	USDF_DBG("uda_devname=%s\n", fp->fab_dev_attrs->uda_devname);

	/*
	 * Make sure address format is good and matches this fabric
	 */
	switch (info->addr_format) {
	case FI_SOCKADDR:
		addrlen = sizeof(struct sockaddr);
		sin = info->src_addr;
		break;
	case FI_SOCKADDR_IN:
		addrlen = sizeof(struct sockaddr_in);
		sin = info->src_addr;
		break;
	case FI_ADDR_STR:
		sin = usdf_format_to_sin(info, info->src_addr);
		if (NULL == sin) {
			ret = -FI_ENOMEM;
			goto fail;
		}
		goto skip_size_check;
	default:
		ret = -FI_EINVAL;
		goto fail;
	}

	if (info->src_addrlen != addrlen) {
		ret =  -FI_EINVAL;
		goto fail;
	}

skip_size_check:
	if (sin->sin_family != AF_INET ||
	    sin->sin_addr.s_addr != fp->fab_dev_attrs->uda_ipaddr_be) {
		USDF_DBG_SYS(DOMAIN, "requested src_addr (%s) != fabric addr (%s)\n",
			inet_ntop(AF_INET, &sin->sin_addr.s_addr,
				requested, sizeof(requested)),
			inet_ntop(AF_INET, &fp->fab_dev_attrs->uda_ipaddr_be,
				actual, sizeof(actual)));

		ret = -FI_EINVAL;
		usdf_free_sin_if_needed(info, sin);
		goto fail;
	}
	usdf_free_sin_if_needed(info, sin);

	ret = usd_open(fp->fab_dev_attrs->uda_devname, &udp->dom_dev);
	if (ret != 0) {
		goto fail;
	}

	udp->dom_fid.fid.fclass = FI_CLASS_DOMAIN;
	udp->dom_fid.fid.context = context;
	udp->dom_fid.fid.ops = &usdf_fid_ops;
	udp->dom_fid.ops = &usdf_domain_ops;
	udp->dom_fid.mr = &usdf_domain_mr_ops;

	ret = pthread_spin_init(&udp->dom_progress_lock,
			PTHREAD_PROCESS_PRIVATE);
	if (ret != 0) {
		ret = -ret;
		goto fail;
	}
	TAILQ_INIT(&udp->dom_tx_ready);
	TAILQ_INIT(&udp->dom_hcq_list);

	udp->dom_info = fi_dupinfo(info);
	if (udp->dom_info == NULL) {
		ret = -FI_ENOMEM;
		goto fail;
	}
	if (udp->dom_info->dest_addr != NULL) {
		free(udp->dom_info->dest_addr);
		udp->dom_info->dest_addr = NULL;
	}

	ret = usdf_dom_rdc_alloc_data(udp);
	if (ret != 0) {
		goto fail;
	}

	udp->dom_fabric = fp;
	LIST_INSERT_HEAD(&fp->fab_domain_list, udp, dom_link);
	ofi_atomic_initialize32(&udp->dom_refcnt, 0);
	ofi_atomic_inc32(&fp->fab_refcnt);

	*domain = &udp->dom_fid;
	return 0;

fail:
	if (udp != NULL) {
		if (udp->dom_info != NULL) {
			fi_freeinfo(udp->dom_info);
		}
		if (udp->dom_dev != NULL) {
			usd_close(udp->dom_dev);
		}
		usdf_dom_rdc_free_data(udp);
		free(udp);
	}
	return ret;
}

int usdf_domain_getname(uint32_t version, struct usd_device_attrs *dap,
			char **name)
{
	int ret = FI_SUCCESS;
	char *buf = NULL;

	if (FI_VERSION_GE(version, FI_VERSION(1, 4))) {
		buf = strdup(dap->uda_devname);
		if (!buf) {
			ret = -errno;
			USDF_DBG("strdup failed while creating domain name\n");
		}
	}

	*name = buf;
	return ret;
}

/* In pre-1.4 the domain name was NULL. This is unfortunate as it makes it
 * difficult to tell whether providing a name was intended. In this case, it can
 * be broken into 4 cases:
 *
 * 1. Version is greater than or equal to 1.4 and a non-NULL hint is provided.
 *    Just do a string compare.
 * 2. Version is greater than or equal to 1.4 and provided hint is NULL.  Treat
 *    this as _valid_ as it could be an application requesting a 1.4 domain name
 *    but not providing an explicit hint.
 * 3. Version is less than 1.4 and a name hint is provided.  This should always
 *    be _invalid_.
 * 4. Version is less than 1.4 and name hint is NULL. This will always be
 *    _valid_.
 */
bool usdf_domain_checkname(uint32_t version, struct usd_device_attrs *dap,
			   const char *hint)
{
	char *reference;
	bool valid;
	int ret;

	USDF_DBG("checking domain name: version=%d, domain name='%s'\n",
		 version, hint);

	if (version) {
		valid = false;

		ret = usdf_domain_getname(version, dap, &reference);
		if (ret < 0)
			return false;

		/* If the reference name exists, then this is version 1.4 or
		 * greater.
		 */
		if (reference) {
			if (hint) {
				/* Case 1 */
				valid = (strcmp(reference, hint) == 0);
			} else {
				/* Case 2 */
				valid = true;
			}
		} else {
			/* Case 3 & 4 */
			valid = (hint == NULL);
		}

		if (!valid)
			USDF_DBG("given hint %s does not match %s -- invalid\n",
				 hint, reference);

		free(reference);
		return valid;
	}

	/* If hint is non-NULL then assume the version is 1.4 if not provided.
	 */
	if (hint)
		return usdf_domain_checkname(FI_VERSION(1, 4), dap, hint);

	return usdf_domain_checkname(FI_VERSION(1, 3), dap, hint);
}

/* Query domain's atomic capability.
 * We dont support atomic operations, just return EOPNOTSUPP.
 */
int usdf_query_atomic(struct fid_domain *domain, enum fi_datatype datatype,
		      enum fi_op op, struct fi_atomic_attr *attr, uint64_t flags)
{
	return -FI_EOPNOTSUPP;
}

/* Catch the version changes for domain_attr. */
int usdf_catch_dom_attr(uint32_t version, const struct fi_info *hints,
			struct fi_domain_attr *dom_attr)
{
	/* version 1.5 introduced new bits. If the user asked for older
	 * version, we can't return these new bits.
	 */
	if (FI_VERSION_LT(version, FI_VERSION(1, 5))) {
		/* We checked mr_mode compatibility before calling
		 * this function. This means it is safe to return
		 * 1.4 default mr_mode.
		 */
		dom_attr->mr_mode = FI_MR_BASIC;

		/* FI_REMOTE_COMM is introduced in 1.5. So don't return it. */
		dom_attr->caps &= ~FI_REMOTE_COMM;

		/* If FI_REMOTE_COMM is given for version < 1.5, fail. */
		if (hints && hints->domain_attr) {
			if (hints->domain_attr->caps == FI_REMOTE_COMM)
				return -FI_EBADFLAGS;
		}
        } else {
            dom_attr->mr_mode &= ~(FI_MR_BASIC | FI_MR_SCALABLE);
	}

	return FI_SUCCESS;
}

/* Catch the version changes for tx_attr. */
int usdf_catch_tx_attr(uint32_t version, const struct fi_tx_attr *tx_attr)
{
	/* In version < 1.5, FI_LOCAL_MR is required. */
	if (FI_VERSION_LT(version, FI_VERSION(1, 5))) {
		if ((tx_attr->mode & FI_LOCAL_MR) == 0)
			return -FI_ENODATA;
	}

	return FI_SUCCESS;
}

/* Catch the version changes for rx_attr. */
int usdf_catch_rx_attr(uint32_t version, const struct fi_rx_attr *rx_attr)
{
	/* In version < 1.5, FI_LOCAL_MR is required. */
	if (FI_VERSION_LT(version, FI_VERSION(1, 5))) {
		if ((rx_attr->mode & FI_LOCAL_MR) == 0)
			return -FI_ENODATA;
	}

	return FI_SUCCESS;
}
