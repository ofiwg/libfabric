/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>
#include "prov.h"

#include "gnix.h"
#include "gnix_nic.h"
#include "gnix_cm_nic.h"
#include "gnix_util.h"
#include "gnix_nameserver.h"
#include "gnix_wait.h"

const char gnix_fab_name[] = "gni";
const char gnix_dom_name[] = "/sys/class/gni/kgni0";
const char gnix_prov_name[] = "gni";

uint32_t gnix_cdm_modes =
	(GNI_CDM_MODE_FAST_DATAGRAM_POLL | GNI_CDM_MODE_FMA_SHARED |
	GNI_CDM_MODE_FMA_SMALL_WINDOW | GNI_CDM_MODE_FORK_PARTCOPY |
	GNI_CDM_MODE_ERR_NO_KILL);

/* default number of directed datagrams per domain */
static int gnix_def_gni_n_dgrams = 128;
/* default number of wildcard datagrams per domain */
static int gnix_def_gni_n_wc_dgrams = 4;
static uint64_t gnix_def_gni_datagram_timeouts = -1;

const struct fi_fabric_attr gnix_fabric_attr = {
	.fabric = NULL,
	.name = NULL,
	.prov_name = NULL,
	.prov_version = FI_VERSION(GNI_MAJOR_VERSION, GNI_MINOR_VERSION),
};

static struct fi_ops_fabric gnix_fab_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = gnix_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = gnix_eq_open,
	.wait_open = gnix_wait_open
};

static void __fabric_destruct(void *obj)
{
	struct gnix_fid_fabric *fab = (struct gnix_fid_fabric *) obj;

	_gnix_alps_cleanup();

	free(fab);
}

static int gnix_fabric_close(fid_t fid)
{
	struct gnix_fid_fabric *fab;
	int references_held;

	fab = container_of(fid, struct gnix_fid_fabric, fab_fid);

	references_held = _gnix_ref_put(fab);
	if (references_held)
		GNIX_INFO(FI_LOG_FABRIC, "failed to fully close fabric due "
				"to lingering references. references=%i fabric=%p\n",
				references_held, fab);

	return FI_SUCCESS;
}

static struct fi_ops gnix_fab_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = gnix_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

/*
 * define methods needed for the GNI fabric provider
 */
static int gnix_fabric_open(struct fi_fabric_attr *attr,
			    struct fid_fabric **fabric,
			    void *context)
{
	struct gnix_fid_fabric *fab;

	if (strcmp(attr->name, gnix_fab_name)) {
		return -FI_ENODATA;
	}

	fab = calloc(1, sizeof(*fab));
	if (!fab) {
		return -FI_ENOMEM;
	}

	/*
	 * set defaults related to use of GNI datagrams
	 */
	fab->n_bnd_dgrams = gnix_def_gni_n_dgrams;
	fab->n_wc_dgrams = gnix_def_gni_n_wc_dgrams;
	fab->datagram_timeout = gnix_def_gni_datagram_timeouts;

	fab->fab_fid.fid.fclass = FI_CLASS_FABRIC;
	fab->fab_fid.fid.context = context;
	fab->fab_fid.fid.ops = &gnix_fab_fi_ops;
	fab->fab_fid.ops = &gnix_fab_ops;
	_gnix_ref_init(&fab->ref_cnt, 1, __fabric_destruct);
	dlist_init(&fab->domain_list);

	*fabric = &fab->fab_fid;

	return FI_SUCCESS;
}

static int gnix_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, struct fi_info *hints,
			struct fi_info **info)
{
	int ret = 0;
	uint64_t mode = GNIX_FAB_MODES;
	enum fi_progress control_progress = FI_PROGRESS_AUTO;
	enum fi_progress data_progress = FI_PROGRESS_AUTO;
	struct fi_info *gnix_info = NULL;
	struct gnix_ep_name *dest_addr = NULL;
	struct gnix_ep_name *src_addr = NULL;
	struct gnix_ep_name *addr = NULL;

	/*
	 * the code below for resolving a node/service to what
	 * will be a gnix_ep_name address is not fully implemented,
	 * but put a place holder in place
	 */
	if (node) {
		addr = malloc(sizeof(*addr));
		if (!addr) {
			ret = -FI_ENOMEM;
			goto err;
		}

		/* resolve node/service to gnix_ep_name */
		ret = gnix_resolve_name(node, service, flags, addr);
		if (ret) {
			goto err;
		}

		if (flags & FI_SOURCE) {
			/* resolved address is the local address */
			src_addr = addr;
			if (hints && hints->dest_addr)
				dest_addr = hints->dest_addr;
		} else {
			/* resolved address is a peer */
			dest_addr = addr;
			if (hints && hints->src_addr)
				src_addr = hints->src_addr;
		}
	}

	if (src_addr)
		GNIX_INFO(FI_LOG_FABRIC, "src_pe: 0x%x src_port: 0x%lx\n",
			  src_addr->gnix_addr.device_addr,
			  src_addr->gnix_addr.cdm_id);
	if (dest_addr)
		GNIX_INFO(FI_LOG_FABRIC, "dest_pe: 0x%x dest_port: 0x%lx\n",
			  dest_addr->gnix_addr.device_addr,
			  dest_addr->gnix_addr.cdm_id);

	if (hints) {
		/*
		 * check for endpoint type, only support FI_EP_RDM for now
		 */
		if (hints->ep_attr) {
			switch (hints->ep_attr->type) {
			case FI_EP_UNSPEC:
			case FI_EP_RDM:
				break;
			default:
				ret = -FI_ENODATA;
				goto err;
			}
		}

		/*
		 * check the mode field
		 */
		if (hints->mode) {
			if ((hints->mode & GNIX_FAB_MODES) != GNIX_FAB_MODES) {
				ret = -FI_ENODATA;
				goto err;
			}
			mode = hints->mode & ~GNIX_FAB_MODES_CLEAR;
		}

		if ((hints->caps & GNIX_EP_RDM_CAPS) != hints->caps) {
			goto err;
		}

		if (hints->ep_attr) {
			switch (hints->ep_attr->protocol) {
			case FI_PROTO_UNSPEC:
			case FI_PROTO_GNI:
				break;
			default:
				ret = -FI_ENODATA;
				goto err;
			}

			if (hints->ep_attr->tx_ctx_cnt > 1) {
				ret = -FI_ENODATA;
				goto err;
			}

			if (hints->ep_attr->rx_ctx_cnt > 1) {
				ret = -FI_ENODATA;
				goto err;
			}
		}

		if (hints->tx_attr) {
			if ((hints->tx_attr->op_flags & GNIX_EP_OP_FLAGS) !=
				hints->tx_attr->op_flags) {
				ret = -FI_ENODATA;
				goto err;
			}
			if (hints->tx_attr->inject_size > GNIX_INJECT_SIZE) {
				ret = -FI_ENODATA;
				goto err;
			}
		}

		if (hints->rx_attr &&
		    (hints->rx_attr->op_flags & GNIX_EP_OP_FLAGS) !=
			hints->rx_attr->op_flags) {
			ret = -FI_ENODATA;
			goto err;
		}

		if (hints->fabric_attr && hints->fabric_attr->name &&
		    strncmp(hints->fabric_attr->name, gnix_fab_name,
			    strlen(gnix_fab_name))) {
			ret = -FI_ENODATA;
			goto err;
		}

		if (hints->domain_attr) {
			if (hints->domain_attr->name &&
			    strncmp(hints->domain_attr->name, gnix_dom_name,
				    strlen(gnix_dom_name))) {
				ret = -FI_ENODATA;
				goto err;
			}

			if (hints->domain_attr->control_progress !=
				FI_PROGRESS_UNSPEC)
				control_progress =
					hints->domain_attr->control_progress;

			if (hints->domain_attr->data_progress !=
				FI_PROGRESS_UNSPEC)
				data_progress =
					hints->domain_attr->control_progress;

			switch (hints->domain_attr->mr_mode) {
			case FI_MR_UNSPEC:
			case FI_MR_BASIC:
				break;
			case FI_MR_SCALABLE:
				ret = -FI_ENODATA;
				goto err;
			}
		}

		if (hints->ep_attr) {
			if (hints->ep_attr->max_msg_size > GNIX_MAX_MSG_SIZE) {
				ret = -FI_ENODATA;
				goto err;
			}
			/*
			 * TODO: tag matching
			 * max_tag_value =
			 * fi_tag_bits(hints->ep_attr->mem_tag_format);
			 */
		}
	}

	/*
	 * fill in the gnix_info struct
	 */
	gnix_info = fi_allocinfo();
	if (gnix_info == NULL) {
		ret = -FI_ENOMEM;
		goto err;
	}

	gnix_info->ep_attr->protocol = FI_PROTO_GNI;
	gnix_info->ep_attr->max_msg_size = GNIX_MAX_MSG_SIZE;
	/* TODO: need to work on this */
	gnix_info->ep_attr->mem_tag_format = 0x0;
	gnix_info->ep_attr->tx_ctx_cnt = 1;
	gnix_info->ep_attr->rx_ctx_cnt = 1;

	gnix_info->domain_attr->threading = FI_THREAD_SAFE;
	gnix_info->domain_attr->control_progress = control_progress;
	gnix_info->domain_attr->data_progress = data_progress;
	gnix_info->domain_attr->av_type = FI_AV_UNSPEC;
	gnix_info->domain_attr->tx_ctx_cnt = gnix_max_nics_per_ptag;
	/* only one aries per node */
	gnix_info->domain_attr->name = strdup(gnix_dom_name);

	gnix_info->next = NULL;
	gnix_info->ep_attr->type = FI_EP_RDM;
	gnix_info->caps = hints ? hints->caps & GNIX_EP_RDM_CAPS :
		GNIX_EP_RDM_CAPS;
	gnix_info->mode = mode;
	gnix_info->addr_format = FI_ADDR_GNI;
	gnix_info->src_addrlen = sizeof(struct gnix_ep_name);
	gnix_info->dest_addrlen = sizeof(struct gnix_ep_name);
	gnix_info->src_addr = src_addr;
	gnix_info->dest_addr = dest_addr;
	gnix_info->fabric_attr->name = strdup(gnix_fab_name);
	/* prov_name gets filled in by fi_getinfo from the gnix_prov struct */
	/* let's consider gni copyrighted :) */

	gnix_info->tx_attr->caps = gnix_info->caps;
	gnix_info->tx_attr->mode = gnix_info->mode;

	if (hints && hints->tx_attr && hints->tx_attr->op_flags) {
		gnix_info->tx_attr->op_flags =
				hints->tx_attr->op_flags & GNIX_EP_OP_FLAGS;
	} else {
		gnix_info->tx_attr->op_flags = 0;
	}

	gnix_info->tx_attr->msg_order = FI_ORDER_SAS;
	gnix_info->tx_attr->comp_order = FI_ORDER_NONE;
	gnix_info->tx_attr->size = GNIX_TX_SIZE_DEFAULT;
	gnix_info->tx_attr->iov_limit = 1;
	gnix_info->tx_attr->inject_size = GNIX_INJECT_SIZE;
	gnix_info->tx_attr->rma_iov_limit = 1;

	gnix_info->rx_attr->caps = gnix_info->caps;
	gnix_info->rx_attr->mode = gnix_info->mode;

	if (hints && hints->rx_attr && hints->rx_attr->op_flags) {
		gnix_info->rx_attr->op_flags =
				hints->rx_attr->op_flags & GNIX_EP_OP_FLAGS;
	} else {
		gnix_info->rx_attr->op_flags = 0;
	}

	gnix_info->rx_attr->msg_order = FI_ORDER_SAS;
	gnix_info->rx_attr->comp_order = FI_ORDER_NONE;
	gnix_info->rx_attr->size = GNIX_RX_SIZE_DEFAULT;
	gnix_info->rx_attr->iov_limit = 1;

	*info = gnix_info;
	return 0;
err:
	if (gnix_info) {
		if (gnix_info->tx_attr) free(gnix_info->tx_attr);
		if (gnix_info->rx_attr) free(gnix_info->rx_attr);
		if (gnix_info->ep_attr) free(gnix_info->ep_attr);
		if (gnix_info->domain_attr) free(gnix_info->domain_attr);
		if (gnix_info->fabric_attr) free(gnix_info->fabric_attr);
		free(gnix_info);
	}

	/*
	 *  for the getinfo method, we need to return -FI_ENODATA  otherwise
	 *  the fi_getinfo call will make an early exit without querying
	 *  other providers which may be avaialble.
	 */
	return -FI_ENODATA;
}

static void gnix_fini(void)
{
}

struct fi_provider gnix_prov = {
	.name = gnix_prov_name,
	.version = FI_VERSION(GNI_MAJOR_VERSION, GNI_MINOR_VERSION),
	.fi_version = FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
	.getinfo = gnix_getinfo,
	.fabric = gnix_fabric_open,
	.cleanup = gnix_fini
};

GNI_INI
{
	struct fi_provider *provider = NULL;
	gni_return_t status;
	gni_version_info_t lib_version;
	int num_devices;
	int rc;

	/*
	 * if no GNI devices available, don't register as provider
	 */
	status = GNI_GetNumLocalDevices(&num_devices);
	if ((status != GNI_RC_SUCCESS) || (num_devices == 0)) {
		return NULL;
	}

	/* sanity check that the 1 aries/node holds */
	assert(num_devices == 1);

	/*
	 * don't register if available ugni is older than one libfabric was
	 * built against
	 */
	status = GNI_GetVersionInformation(&lib_version);
	if ((GNI_GET_MAJOR(lib_version.ugni_version) > GNI_MAJOR_REV) ||
	    ((GNI_GET_MAJOR(lib_version.ugni_version) == GNI_MAJOR_REV) &&
	     GNI_GET_MINOR(lib_version.ugni_version) >= GNI_MINOR_REV)) {
		provider = &gnix_prov;
	}

	rc = _gnix_nics_per_rank(&gnix_max_nics_per_ptag);
	if (rc == FI_SUCCESS) {
		GNIX_INFO(FI_LOG_FABRIC, "gnix_max_nics_per_ptag: %u\n",
			  gnix_max_nics_per_ptag);
	} else {
		GNIX_INFO(FI_LOG_FABRIC, "_gnix_nics_per_rank failed: %d\n", rc);
	}
	if (getenv("GNIX_MAX_NICS") != NULL)
		gnix_max_nics_per_ptag = atoi(getenv("GNIX_MAX_NICS"));

	return (provider);
}
