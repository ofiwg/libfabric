/*
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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>
#include <poll.h>
#include <time.h>
#include <string.h>


#include "gnix.h"

#include <criterion/criterion.h>
#include "gnix_rdma_headers.h"

static struct fid_fabric *fabric;
static struct fi_info *fi;

static void setup(void)
{
	int ret;
	struct fi_info *hints;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->fabric_attr->prov_name = strdup("gni");

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
	char *other_reg_type = "none";
	char *string_val;
	bool xpmem_toggle = false, xpmem_check;

	memset(doms, 0, num_doms*sizeof(struct fid_domain *));

	for (i = 0; i < num_doms; i++) {
		ret = fi_domain(fabric, fi, &doms[i], NULL);
		cr_assert(ret == FI_SUCCESS, "fi_domain");
		ret = fi_open_ops(&doms[i]->fid, FI_GNI_DOMAIN_OPS_1,
				  0, (void **) &gni_domain_ops, NULL);
		cr_assert(ret == FI_SUCCESS, "fi_open_ops");
		for (op = 0; op < GNI_NUM_DOM_OPS; op++) {
			val = i*op+op;
			switch (op) {
			case GNI_MR_CACHE:
				ret = gni_domain_ops->set_val(&doms[i]->fid, op,
						&other_reg_type);
				break;
			case GNI_XPMEM_ENABLE:
				ret = gni_domain_ops->set_val(&doms[i]->fid, op,
						&xpmem_toggle);
				break;
			default:
				ret = gni_domain_ops->set_val(&doms[i]->fid, op, &val);
				break;
			}
			cr_assert(ret == FI_SUCCESS, "set_val");

			switch (op) {
			case GNI_MR_CACHE:
				ret = gni_domain_ops->get_val(&doms[i]->fid, op, &string_val);
				break;
			case GNI_XPMEM_ENABLE:
				ret = gni_domain_ops->get_val(&doms[i]->fid, op,
							      &xpmem_check);
				break;
			default:
				ret = gni_domain_ops->get_val(&doms[i]->fid, op, &val);
				break;
			}
			cr_assert(ret == FI_SUCCESS, "get_val");

			switch (op) {
			case GNI_MR_CACHE:
				cr_assert_eq(strncmp(other_reg_type, string_val,
						strlen(other_reg_type)),  0, "Incorrect op value");
				break;
			case GNI_XPMEM_ENABLE:
				cr_assert(xpmem_toggle == xpmem_check,
					  "Incorrect op value");
			default:
				cr_assert(val == i*op+op, "Incorrect op value");
				break;
			}
		}
		ret = fi_close(&doms[i]->fid);
		cr_assert(ret == FI_SUCCESS, "fi_close domain");
	}
}

Test(domain, cache_flush_op)
{
	int i, ret;
	const int num_doms = 11;
	struct fid_domain *doms[num_doms];
	struct fi_gni_ops_domain *gni_domain_ops;
	struct fid_mr *mr;
	char *buf = calloc(1024, sizeof(char));

	cr_assert(buf);

	memset(doms, 0, num_doms*sizeof(struct fid_domain *));

	for (i = 0; i < num_doms; i++) {
		ret = fi_domain(fabric, fi, &doms[i], NULL);
		cr_assert(ret == FI_SUCCESS, "fi_domain");
		ret = fi_open_ops(&doms[i]->fid, FI_GNI_DOMAIN_OPS_1,
				  0, (void **) &gni_domain_ops, NULL);
		cr_assert(ret == FI_SUCCESS, "fi_open_ops");

		ret = fi_mr_reg(doms[i], buf, 1024, FI_READ, 0, 0, 0, &mr, NULL);
		cr_assert(ret == FI_SUCCESS, "fi_reg_mr");

		ret = fi_close(&mr->fid);
		cr_assert(ret == FI_SUCCESS, "fi_close mr");

		ret = gni_domain_ops->flush_cache(&doms[i]->fid);
		cr_assert(ret == FI_SUCCESS, "flush cache");

		ret = fi_close(&doms[i]->fid);
		cr_assert(ret == FI_SUCCESS, "fi_close domain");
	}

	free(buf);
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
