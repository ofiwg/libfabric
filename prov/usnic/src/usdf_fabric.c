/*
 * Copyright (c) 2014, Cisco Systems, Inc. All rights reserved.
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <asm/types.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>
#include "fi.h"

#include "usnic_direct.h"
#include "usdf.h"

static int
usdf_freeinfo(struct fi_info *info)
{
	fi_freeinfo_internal(info);
	return 0;
}

static int
usdf_validate_hint_caps(struct fi_info *hints, struct usd_device_attrs *dap)
{
	struct fi_fabric_attr *fattrp;
	struct sockaddr_in *sin;

	switch (hints->ep_type) {
	case FI_EP_UNSPEC:
	case FI_EP_DGRAM:
		break;
	default:
		return -FI_ENODATA;
	}

	/* check that we are capable of what's requested */
	if ((hints->caps & ~USDF_CAPS) != 0) {
		return -FI_ENODATA;
	}

	/* app must support these modes */
	if ((hints->mode & USDF_REQ_MODE) != USDF_REQ_MODE) {
		return -FI_ENODATA;
	}

	switch (hints->addr_format) {
	case FI_ADDR_UNSPEC:
	case FI_SOCKADDR_IN:
	case FI_SOCKADDR:
		break;
	default:
		return -FI_ENODATA;
	}
	sin = hints->src_addr;
	if (sin != NULL) {
		if (hints->src_addrlen < sizeof(*sin)) {
			return -FI_ENODATA;
		}
		if (sin->sin_addr.s_addr != INADDR_ANY &&
			sin->sin_addr.s_addr != dap->uda_ipaddr_be) {
			return -FI_ENODATA;
		}
	}

	if (hints->ep_attr != NULL) {
		switch (hints->ep_attr->protocol) {
		case FI_PROTO_UNSPEC:
		case FI_PROTO_UDP:
			break;
		default:
			return -FI_ENODATA;
		}
	}

	fattrp = hints->fabric_attr;
	if (fattrp != NULL) {
		if (fattrp->prov_name != NULL &&
                    strcmp(fattrp->prov_name, USDF_FI_NAME) != 0) {
			return -FI_ENODATA;
		}
		if (fattrp->name != NULL &&
                    strcmp(fattrp->name, dap->uda_devname) != 0) {
			return -FI_ENODATA;
		}
	}

	return 0;
}

static int
usdf_fill_addr_info(struct fi_info *fi, struct fi_info *hints,
		struct usd_device_attrs *dap)
{
	struct sockaddr_in *sin;
	int ret;

	/* If hints speficied, we already validated requested addr_format */
	if (hints != NULL && hints->addr_format != FI_ADDR_UNSPEC) {
		fi->addr_format = hints->addr_format;
	} else {
		fi->addr_format = FI_SOCKADDR_IN;
	}

	switch (fi->addr_format) {
	case FI_SOCKADDR:
	case FI_SOCKADDR_IN:
		sin = calloc(1, sizeof(*sin));
		fi->src_addr = sin;
		if (sin == NULL) {
			ret = -FI_ENOMEM;
			goto fail;
		}
		fi->src_addrlen = sizeof(*sin);
		sin->sin_family = AF_INET;
		sin->sin_addr.s_addr = dap->uda_ipaddr_be;
		break;
	default:
		break;
	}

	return 0;

fail:
	return ret;		// fi_freeinfo() in caller frees all
}



static int
usdf_fill_info_dgram(struct fi_info *fi, struct fi_info *hints,
		struct usd_device_attrs *dap)
{
	struct fi_fabric_attr *fattrp;
	struct fi_domain_attr *dattrp;
	struct fi_ep_attr *eattrp;
	int ret;

	fi->caps = USDF_CAPS;

	if (hints != NULL) {
		fi->mode = hints->mode & USDF_SUPP_MODE;
	} else {
		fi->mode = USDF_SUPP_MODE;
	}
	fi->ep_type = FI_EP_DGRAM;

	ret = usdf_fill_addr_info(fi, hints, dap);
	if (ret != 0) {
		goto fail;
	}

	/* fabric attrs */
	fattrp = fi->fabric_attr;
	fattrp->name = strdup(dap->uda_devname);
	if (fattrp->name == NULL) {
		ret = -FI_ENOMEM;
		goto fail;
	}

	/* endpoint attrs */
	eattrp = fi->ep_attr;
	if (fi->mode & FI_MSG_PREFIX) {
		eattrp->msg_prefix_size = USDF_HDR_BUF_ENTRY;
	}
	eattrp->max_msg_size = dap->uda_mtu -
		sizeof(struct usd_udp_hdr);
	eattrp->protocol = FI_PROTO_UDP;
	eattrp->tx_ctx_cnt = 1;
	eattrp->rx_ctx_cnt = 1;

	/* domain attrs */
	dattrp = fi->domain_attr;
	dattrp->threading = FI_THREAD_UNSPEC;
	dattrp->control_progress = FI_PROGRESS_MANUAL;
	dattrp->data_progress = FI_PROGRESS_AUTO;

	return 0;

fail:
	return ret;		// fi_freeinfo() in caller frees all
}

static int
usdf_getinfo(uint32_t version, const char *node, const char *service,
	       uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	struct usd_device_entry devs[USD_MAX_DEVICES];
	struct fi_info *fi_first;
	struct fi_info *fi_last;
	struct fi_info *fi;
	struct usd_device *dev;
	struct usd_device_attrs dattr;
	struct addrinfo *ai;
	struct sockaddr_in *sin;
	int metric;
	int num_devs;
	int i;
	int ret;

	fi_first = NULL;
	fi_last = NULL;
	ai = NULL;
	sin = NULL;
	dev = NULL;

	if (node != NULL) {
		ret = getaddrinfo(node, service, NULL, &ai);
		if (ret != 0) {
			return -errno;
		}
		sin = (struct sockaddr_in *)ai->ai_addr;
	}

	num_devs = USD_MAX_DEVICES;
	ret = usd_get_device_list(devs, &num_devs);
	if (ret != 0) {
		goto fail;
	}

	for (i = 0; i < num_devs; ++i) {
		ret = usd_open(devs[i].ude_devname, &dev);
		if (ret != 0) {
			continue;
		}

		ret = usd_get_device_attrs(dev, &dattr);
		if (ret != 0) {
			goto next_dev;
		}

		/* See if dest is reachable from this device */
		if (node != NULL) {
			ret = usd_get_dest_distance(dev,
					sin->sin_addr.s_addr, &metric);
			if (ret != 0) {
				goto fail;
			}
			if (metric == -1) {
				goto next_dev;
			}
		}

		/* Does this device match requested attributes? */
		if (hints != NULL) {
			ret = usdf_validate_hint_caps(hints, &dattr);
			if (ret != 0) {
				goto next_dev;
			}
		}

		fi = fi_allocinfo_internal();
		if (fi == NULL) {
			ret = -FI_ENOMEM;
			goto fail;
		}

		/* Fill info stuct to return */
		ret = usdf_fill_info_dgram(fi, hints, &dattr);
		if (ret != 0) {
			goto fail;
		}

		/* add to tail of list */
		if (fi_first == NULL) {
			fi_first = fi;
		} else {
			fi_last->next = fi;
		}
		fi_last = fi;

next_dev:
		usd_close(dev);
		dev = NULL;
	}
	if (ai != NULL) {
		freeaddrinfo(ai);
	}

	if (fi_first != NULL) {
		*info = fi_first;
		return 0;
	} else {
		ret = -FI_ENODATA;
		goto fail;
	}

fail:
	if (dev != NULL) {
		usd_close(dev);
	}
	if (fi_first != NULL) {
		fi_freeinfo(fi_first);
	}
	if (ai != NULL) {
		freeaddrinfo(ai);
	}
	return ret;
}

static int
usdf_fabric_close(fid_t fid)
{
	struct usdf_fabric *fp;

	fp = fab_fidtou(fid);
	if (atomic_get(&fp->fab_refcnt) > 0) {
		return -FI_EBUSY;
	}

	free(fp);
	return 0;
}

static struct fi_ops usdf_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = usdf_fabric_close,
};

static struct fi_ops_fabric usdf_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = usdf_domain_open,
	.eq_open = usdf_eq_open,
};

int
usdf_fabric_open(struct fi_fabric_attr *fattrp, struct fid_fabric **fabric,
	       void *context)
{
	struct usdf_fabric *fp;
	int ret;

	fp = calloc(1, sizeof(*fp));
	if (fp == NULL) {
		return -FI_ENOMEM;
	}

	fp->fab_name = strdup(fattrp->name);
	if (fp->fab_name == NULL) {
		ret = -FI_ENOMEM;
		goto fail;
	}

	fp->fab_fid.fid.fclass = FI_CLASS_FABRIC;
	fp->fab_fid.fid.context = context;
	fp->fab_fid.fid.ops = &usdf_fi_ops;
	fp->fab_fid.ops = &usdf_ops_fabric;

	atomic_init(&fp->fab_refcnt, 0);
	*fabric = &fp->fab_fid;
	return 0;

fail:
	if (fp != NULL) {
		free(fp);
	}
	return ret;
}

static struct fi_provider usdf_ops = {
	.name = USDF_FI_NAME,
	.version = FI_VERSION(0, 7),
	.getinfo = usdf_getinfo,
	.freeinfo = usdf_freeinfo,
	.fabric = usdf_fabric_open,
};

static void __attribute__((constructor))
usdf_ini(void)
{
	(void) fi_register(&usdf_ops);
}

static void __attribute__((destructor)) 
usdf_fini(void)
{
}
