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
# include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <poll.h>
#include <sys/eventfd.h>

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
usdf_close_domain(fid_t fid)
{
	struct usdf_domain *udp;
	void *rv;
	int ret;

	udp = container_of(fid, struct usdf_domain, dom_fid.fid);
	if (atomic_get(&udp->dom_refcnt) > 0) {
		return -FI_EBUSY;
	}

	/* Tell progression thread to exit */
	udp->dom_exit = 1;
	usdf_add_progression_item(udp);
	pthread_join(udp->dom_thread, &rv);

	if (udp->dom_dev != NULL) {
		ret = usd_close(udp->dom_dev);
		if (ret != 0) {
			return ret;
		}
	}

	atomic_dec(&udp->dom_fabric->fab_refcnt);
	free(udp);

	return 0;
}

static struct fi_ops usdf_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = usdf_close_domain,
};

static struct fi_ops_mr usdf_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = usdf_reg_mr,
};

static struct fi_ops_domain usdf_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.cq_open = usdf_cq_open,
	.av_open = usdf_av_open,
	.endpoint = usdf_endpoint_open,
};

int
usdf_domain_open(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	struct usdf_fabric *fab;
	struct usdf_domain *udp;
	int ret;

	udp = calloc(1, sizeof *udp);
	if (udp == NULL) {
		ret = -FI_ENOMEM;
		goto fail;
	}
	udp->dom_eventfd = -1;

	fab = fab_fidtou(fabric);
	ret = usd_open(fab->fab_name, &udp->dom_dev);
	if (ret != 0) {
		goto fail;
	}

	udp->dom_fid.fid.fclass = FI_CLASS_DOMAIN;
	udp->dom_fid.fid.context = context;
	udp->dom_fid.fid.ops = &usdf_fid_ops;
	udp->dom_fid.ops = &usdf_domain_ops;
	udp->dom_fid.mr = &usdf_domain_mr_ops;

	udp->dom_eventfd = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
	if (udp->dom_eventfd == -1) {
		ret = -errno;
		goto fail;
	}
	ret = pthread_create(&udp->dom_thread, NULL,
			usdf_progression_thread, udp);
	if (ret != 0) {
		ret = -ret;
		goto fail;
	}
	atomic_init(&udp->dom_pending_items, 0);

	udp->dom_fabric = fab;
	pthread_spin_init(&udp->dom_usd_lock, PTHREAD_PROCESS_PRIVATE);
	atomic_init(&udp->dom_refcnt, 0);
	atomic_inc(&fab->fab_refcnt);

	*domain = &udp->dom_fid;
	return 0;

fail:
	if (udp != NULL) {
		if (udp->dom_eventfd != -1) {
			close(udp->dom_eventfd);
		}
		free(udp);
	}
	return ret;
}
