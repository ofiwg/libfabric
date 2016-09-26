/*
 * Copyright (c) 2015-2016 Cray Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>
#include <poll.h>
#include <time.h>
#include <string.h>


#include "gnix_ep.h"

#include <criterion/criterion.h>
#include "gnix_rdma_headers.h"

static struct fi_info *hints;
static struct fi_info *fi;
static struct fid_fabric *fab;
static struct fid_domain *dom;

static void setup(void)
{
	int ret;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->fabric_attr->name = strdup("gni");

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(!ret, "fi_getinfo");

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	cr_assert(!ret, "fi_fabric");

	ret = fi_domain(fab, fi, &dom, NULL);
	cr_assert(!ret, "fi_domain");

}

static void teardown(void)
{
	int ret;

	ret = fi_close(&dom->fid);
	cr_assert(!ret, "fi_close domain");

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "fi_close fabric");

	fi_freeinfo(fi);
	fi_freeinfo(hints);
}

TestSuite(endpoint, .init = setup, .fini = teardown);

Test(endpoint_info, info)
{
	int ret;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");
	hints->fabric_attr->name = strdup("gni");

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(!ret, "fi_getinfo");
	cr_assert_eq(fi->ep_attr->type, FI_EP_RDM);
	cr_assert_eq(fi->next->ep_attr->type, FI_EP_DGRAM);

	fi_freeinfo(fi);

	hints->ep_attr->type = FI_EP_RDM;
	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(!ret, "fi_getinfo");
	cr_assert_eq(fi->ep_attr->type, FI_EP_RDM);

	fi_freeinfo(fi);

	hints->ep_attr->type = FI_EP_DGRAM;
	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(!ret, "fi_getinfo");
	cr_assert_eq(fi->ep_attr->type, FI_EP_DGRAM);

	fi_freeinfo(fi);
	fi_freeinfo(hints);
}

Test(endpoint, open_close)
{
	int i, ret;
	const int num_eps = 61;
	struct fid_ep *eps[num_eps];

	memset(eps, 0, num_eps*sizeof(struct fid_ep *));

	for (i = 0; i < num_eps; i++) {
		ret = fi_endpoint(dom, fi, &eps[i], NULL);
		cr_assert(!ret, "fi_endpoint");
		struct gnix_fid_ep *ep = container_of(eps[i],
						      struct gnix_fid_ep,
						      ep_fid);
		cr_assert(ep, "endpoint not allcoated");

		/* Check fields (fill in as implemented) */
		cr_assert(ep->nic, "NIC not allocated");
		cr_assert(!_gnix_fl_empty(&ep->fr_freelist),
			  "gnix_fab_req freelist empty");
	}

	for (i = num_eps-1; i >= 0; i--) {
		ret = fi_close(&eps[i]->fid);
		cr_assert(!ret, "fi_close endpoint");
	}

}

Test(endpoint, getsetopt)
{
	int ret;
	struct fid_ep *ep = NULL;
	uint64_t val;
	size_t len;

	ret = fi_endpoint(dom, fi, &ep, NULL);
	cr_assert(!ret, "fi_endpoint");

	/* Test bad params. */
	ret = fi_getopt(&ep->fid, !FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			(void *)&val, &len);
	cr_assert(ret == -FI_ENOPROTOOPT, "fi_getopt");

	ret = fi_getopt(&ep->fid, FI_OPT_ENDPOINT, !FI_OPT_MIN_MULTI_RECV,
			(void *)&val, &len);
	cr_assert(ret == -FI_ENOPROTOOPT, "fi_getopt");

	ret = fi_getopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			NULL, &len);
	cr_assert(ret == -FI_EINVAL, "fi_getopt");

	ret = fi_getopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			(void *)&val, NULL);
	cr_assert(ret == -FI_EINVAL, "fi_getopt");

	ret = fi_setopt(&ep->fid, !FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			(void *)&val, sizeof(size_t));
	cr_assert(ret == -FI_ENOPROTOOPT, "fi_setopt");

	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT, !FI_OPT_MIN_MULTI_RECV,
			(void *)&val, sizeof(size_t));
	cr_assert(ret == -FI_ENOPROTOOPT, "fi_setopt");

	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			NULL, sizeof(size_t));
	cr_assert(ret == -FI_EINVAL, "fi_setopt");

	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			(void *)&val, sizeof(size_t) - 1);
	cr_assert(ret == -FI_EINVAL, "fi_setopt");

	/* Test update. */
	ret = fi_getopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			(void *)&val, &len);
	cr_assert(!ret, "fi_getopt");
	cr_assert(val == GNIX_OPT_MIN_MULTI_RECV_DEFAULT, "fi_getopt");
	cr_assert(len == sizeof(size_t), "fi_getopt");

	val = 128;
	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			(void *)&val, sizeof(size_t));
	cr_assert(!ret, "fi_setopt");

	ret = fi_getopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			(void *)&val, &len);
	cr_assert(!ret, "fi_getopt");
	cr_assert(val == 128, "fi_getopt");
	cr_assert(len == sizeof(size_t), "fi_getopt");

	ret = fi_close(&ep->fid);
	cr_assert(!ret, "fi_close endpoint");
}

Test(endpoint, sizeleft)
{
	int ret;
	size_t sz;
	struct fid_ep *ep = NULL;

	ret = fi_endpoint(dom, fi, &ep, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_endpoint");

	/* Test in disabled state. */
	sz = fi_rx_size_left(ep);
	cr_assert(sz == -FI_EOPBADSTATE, "fi_rx_size_left");

	sz = fi_tx_size_left(ep);
	cr_assert(sz == -FI_EOPBADSTATE, "fi_tx_size_left");

	ret = fi_enable(ep);
	cr_assert(ret == FI_SUCCESS, "fi_enable");

	/* Test default values. */
	sz = fi_rx_size_left(ep);
	cr_assert(sz == GNIX_RX_SIZE_DEFAULT, "fi_rx_size_left");

	sz = fi_tx_size_left(ep);
	cr_assert(sz == GNIX_TX_SIZE_DEFAULT, "fi_tx_size_left");

	ret = fi_close(&ep->fid);
	cr_assert(!ret, "fi_close endpoint");
}

Test(endpoint, getsetopt_gni_ep)
{
	int ret;
	int val;
	struct fid_ep *ep = NULL;
	struct fi_gni_ops_ep *ep_ops;
	struct gnix_fid_ep *ep_priv = NULL;

	ret = fi_endpoint(dom, fi, &ep, NULL);
	cr_assert(!ret, "fi_endpoint");

	ep_priv = (struct gnix_fid_ep *) ep;

	ret = fi_open_ops(&ep->fid, "ep ops 1", 0, (void **) &ep_ops, NULL);
	cr_assert(!ret, "fi_open_ops endpoint");

	ret = ep_ops->get_val(&ep->fid, GNI_HASH_TAG_IMPL, &val);
	cr_assert(!ret, "ep_ops get_val");
	cr_assert_eq(val, 0);
	cr_assert_eq(ep_priv->use_tag_hlist, 0);

	val = 1; // set the hash implementation
	ret = ep_ops->set_val(&ep->fid, GNI_HASH_TAG_IMPL, &val);
	cr_assert(!ret, "ep_ops set_val");
	cr_assert_eq(ep_priv->use_tag_hlist, 1);
	cr_assert_eq(ep_priv->unexp_recv_queue.attr.type, GNIX_TAG_HLIST);
	cr_assert_eq(ep_priv->posted_recv_queue.attr.type, GNIX_TAG_HLIST);
	cr_assert_eq(ep_priv->tagged_unexp_recv_queue.attr.type, GNIX_TAG_HLIST);
	cr_assert_eq(ep_priv->tagged_posted_recv_queue.attr.type, GNIX_TAG_HLIST);


	val = 0; // reset the value
	ret = ep_ops->get_val(&ep->fid, GNI_HASH_TAG_IMPL, &val);
	cr_assert(!ret, "ep_ops get_val");
	cr_assert_eq(val, 1);
	cr_assert_eq(ep_priv->use_tag_hlist, 1);

	ret = fi_close(&ep->fid);
	cr_assert(!ret, "fi_close endpoint");
}
