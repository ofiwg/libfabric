/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015 Cray Inc.  All rights reserved.
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
#include <pthread.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "gnix_vc.h"
#include "gnix_cm_nic.h"
#include "gnix_hashtable.h"
#include "gnix_rma.h"

#include <criterion/criterion.h>

#define NUM_EPS 11 /* 5 usually works, but sometimes hangs */

static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep[NUM_EPS];
static struct fid_av *av;
static struct fi_info *hints;
static struct fi_info *fi;
static void *ep_name[NUM_EPS];
static fi_addr_t gni_addr[NUM_EPS];
static struct fid_cq *msg_cq[NUM_EPS];
static struct fi_cq_attr cq_attr;

static int target[NUM_EPS];
static int source[NUM_EPS];
struct fid_mr *rem_mr, *loc_mr;
static uint64_t mr_key;

static struct fi_gni_ops_domain *gni_domain_ops;

static void setup(void)
{
	int i;
	int ret = 0;
	struct fi_av_attr attr;
	size_t addrlen = 0;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->domain_attr->cq_data_size = 4;
	hints->mode = ~0;

	hints->fabric_attr->name = strdup("gni");

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(!ret, "fi_getinfo");

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	cr_assert(!ret, "fi_fabric");

	ret = fi_domain(fab, fi, &dom, NULL);
	cr_assert(!ret, "fi_domain");

	ret = fi_open_ops(&dom->fid, FI_GNI_DOMAIN_OPS_1, 0,
			  (void **) &gni_domain_ops, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_open_ops");

	attr.type = FI_AV_TABLE;
	attr.count = NUM_EPS;

	ret = fi_av_open(dom, &attr, &av, NULL);
	cr_assert(!ret, "fi_av_open");

	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.size = 1024;
	cq_attr.wait_obj = 0;

	for (i = 0; i < NUM_EPS; i++) {
		ret = fi_endpoint(dom, fi, &ep[i], NULL);
		cr_assert(!ret, "fi_endpoint");
		cr_assert(ep[i]);
		ret = fi_cq_open(dom, &cq_attr, &msg_cq[i], 0);
		cr_assert(!ret, "fi_cq_open");
		ret = fi_ep_bind(ep[i], &msg_cq[i]->fid, FI_SEND | FI_RECV);
		cr_assert(!ret, "fi_ep_bind");
	}

	ret = fi_getname(&ep[0]->fid, NULL, &addrlen);
	cr_assert_eq(ret, -FI_ETOOSMALL);
	cr_assert(addrlen > 0);

	for (i = 0; i < NUM_EPS; i++) {
		ep_name[i] = malloc(addrlen);
		cr_assert(ep_name[i] != NULL);
		ret = fi_getname(&ep[i]->fid, ep_name[i], &addrlen);
		cr_assert(ret == FI_SUCCESS);
		ret = fi_av_insert(av, ep_name[i], 1, &gni_addr[i], 0, NULL);
		cr_assert(ret == 1);
	}

	for (i = 0; i < NUM_EPS; i++) {
		ret = fi_ep_bind(ep[i], &av->fid, 0);
		cr_assert(!ret, "fi_ep_bind");
		ret = fi_enable(ep[i]);
		cr_assert(!ret, "fi_ep_enable");
	}

	ret = fi_mr_reg(dom, target, NUM_EPS*sizeof(int),
			FI_RECV, 0, 0, 0, &rem_mr, &target);
	cr_assert_eq(ret, 0);

	ret = fi_mr_reg(dom, source, NUM_EPS*sizeof(int),
			FI_SEND, 0, 0, 0, &loc_mr, &source);
	cr_assert_eq(ret, 0);

	mr_key = fi_mr_key(rem_mr);
}

static void teardown(void)
{
	int i;
	int ret = 0;

	fi_close(&loc_mr->fid);
	fi_close(&rem_mr->fid);

	for (i = 0; i < NUM_EPS; i++) {
		ret = fi_close(&ep[i]->fid);
		cr_assert(!ret, "failure in closing ep.");
		ret = fi_close(&msg_cq[i]->fid);
		cr_assert(!ret, "failure in msg cq.");
		free(ep_name[i]);
	}

	ret = fi_close(&av->fid);
	cr_assert(!ret, "failure in closing av.");

	ret = fi_close(&dom->fid);
	cr_assert(!ret, "failure in closing domain.");

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "failure in closing fabric.");

	fi_freeinfo(fi);
	fi_freeinfo(hints);
}

TestSuite(rdm_rx_overrun, .init = setup, .fini = teardown, .disabled = true);


Test(rdm_rx_overrun, all_to_one)
{
	int i;
	int source_done = 0, dest_done = 0;
	struct fi_cq_entry s_cqe, d_cqe;
	ssize_t sz;
	int ctx[NUM_EPS];

	/*
	 * This test doesn't work even when the rx_cq_size is left at
	 * the default.  Changed to reduce the size when the bug is found.

	int ret;
	uint32_t rx_cq_size = 1;

	ret = gni_domain_ops->set_val(&dom->fid, GNI_RX_CQ_SIZE, &rx_cq_size);
	cr_assert(ret == FI_SUCCESS, "set_val");
	*/

	for (i = 0; i < NUM_EPS; i++) {
		source[i] = i;
		target[i] = -1;
		ctx[i] = -1;
	}

	for (i = 1; i < NUM_EPS; i++) {
		sz = fi_send(ep[i], &source[i], sizeof(int), loc_mr,
			     gni_addr[0], ctx+i);
		cr_assert_eq(sz, 0);
	}

	do {
		for (i = 1; i < NUM_EPS; i++) {
			if (fi_cq_read(msg_cq[i], &s_cqe, 1) == 1) {
				cr_assert_eq((uint64_t) s_cqe.op_context,
					     (uint64_t) (ctx+i));
				source_done += 1;
			}
		}
	} while (source_done != NUM_EPS-1);

	for (i = 1; i < NUM_EPS; i++) {
		sz = fi_recv(ep[0], &target[i], sizeof(int), rem_mr,
			     gni_addr[i], ctx+i);
		cr_assert_eq(sz, 0);
	}

	do {
		for (i = 1; i < NUM_EPS; i++) {
			if (fi_cq_read(msg_cq[0], &d_cqe, 1) == 1) {
				cr_assert_eq((uint64_t)d_cqe.op_context,
					     (uint64_t)(ctx+i));
				dest_done += 1;
			}
		}
	} while (dest_done != NUM_EPS-1);


	/* error checking */
	for (i = 1; i < NUM_EPS; i++) {
		cr_assert(target[i] < NUM_EPS);
		ctx[target[i]] = target[i];
	}

	for (i = 1; i < NUM_EPS; i++) {
		cr_assert(ctx[i] == i);
	}

}

