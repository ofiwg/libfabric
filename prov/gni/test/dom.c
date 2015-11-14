/*
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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>
#include <poll.h>
#include <time.h>
#include <string.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>

#include "gnix.h"

#include <criterion/criterion.h>

static struct fid_fabric *fabric;
static struct fi_info *fi;

static void setup(void)
{
	int ret;
	struct fi_info *hints;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->fabric_attr->name = strdup("gni");

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(ret == FI_SUCCESS, "fi_getinfo");

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_fabric");

	fi_freeinfo(hints);
}

static void teardown(void)
{
	int ret;

	ret = fi_close(&fabric->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close fabric");

	fi_freeinfo(fi);
}

TestSuite(domain, .init = setup, .fini = teardown);

Test(domain, many_domains)
{
	int i, ret;
	const int num_doms = 7919;
	struct fid_domain *doms[num_doms];
	struct gnix_fid_domain *gdom;
	struct gnix_fid_fabric *gfab;

	memset(doms, 0, num_doms*sizeof(struct fid_domain *));

	gfab = container_of(fabric, struct gnix_fid_fabric, fab_fid);
	for (i = 0; i < num_doms; i++) {
		ret = fi_domain(fabric, fi, &doms[i], NULL);
		cr_assert(ret == FI_SUCCESS, "fi_domain");
		gdom = container_of(doms[i], struct gnix_fid_domain,
				    domain_fid);
		cr_assert(gdom, "domain not allcoated");
		cr_assert(gdom->fabric == gfab, "Incorrect fabric");
		cr_assert(atomic_get(&gdom->ref_cnt.references) == 1,
				"Incorrect ref_cnt");

	}

	for (i = num_doms-1; i >= 0; i--) {
		ret = fi_close(&doms[i]->fid);
		cr_assert(ret == FI_SUCCESS, "fi_close domain");
	}

}

Test(domain, open_ops)
{
	int i, ret;
	const int num_doms = 11;
	struct fid_domain *doms[num_doms];
	struct fi_gni_ops_domain *gni_domain_ops;
	enum dom_ops_val op;
	uint32_t val;

	memset(doms, 0, num_doms*sizeof(struct fid_domain *));

	for (i = 0; i < num_doms; i++) {
		ret = fi_domain(fabric, fi, &doms[i], NULL);
		cr_assert(ret == FI_SUCCESS, "fi_domain");
		ret = fi_open_ops(&doms[i]->fid, FI_GNI_DOMAIN_OPS_1,
				  0, (void **) &gni_domain_ops, NULL);
		cr_assert(ret == FI_SUCCESS, "fi_open_ops");
		for (op = 0; op < GNI_NUM_DOM_OPS; op++) {
			val = i*op+op;
			ret = gni_domain_ops->set_val(&doms[i]->fid, op, &val);
			cr_assert(ret == FI_SUCCESS, "set_val");
		}
	}

	for (i = num_doms-1; i >= 0; i--) {
		for (op = 0; op < GNI_NUM_DOM_OPS; op++) {
			ret = gni_domain_ops->get_val(&doms[i]->fid, op, &val);
			cr_assert(ret == FI_SUCCESS, "get_val");
			cr_assert(val == i*op+op, "Incorrect op value");
		}
		ret = fi_close(&doms[i]->fid);
		cr_assert(ret == FI_SUCCESS, "fi_close domain");
	}

}

Test(domain, invalid_open_ops)
{
	int ret;
	struct fid_domain *dom;
	struct fi_gni_ops_domain *gni_domain_ops;
	uint32_t val = 0;

	ret = fi_domain(fabric, fi, &dom, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_domain");
	ret = fi_open_ops(&dom->fid, FI_GNI_DOMAIN_OPS_1,
			  0, (void **) &gni_domain_ops, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_open_ops");

	ret = gni_domain_ops->get_val(&dom->fid, GNI_NUM_DOM_OPS, &val);
	cr_assert(ret == -FI_EINVAL, "get_val");

	ret = gni_domain_ops->set_val(&dom->fid, GNI_NUM_DOM_OPS, &val);
	cr_assert(ret == -FI_EINVAL, "set_val");

	ret = fi_close(&dom->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close domain");
}
