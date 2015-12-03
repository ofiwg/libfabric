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
#include "gnix_atomic.h"

#include <criterion/criterion.h>

#if 1
#define dbg_printf(...)
#else
#define dbg_printf(...)		\
do {				\
	printf(__VA_ARGS__);	\
	fflush(stdout);		\
} while (0)
#endif

static struct fid_fabric *fab;
static struct fid_domain *dom;
struct fi_gni_ops_domain *gni_domain_ops;
static struct fid_ep *ep[2];
static struct fid_av *av;
static struct fi_info *hints;
static struct fi_info *fi;
void *ep_name[2];
size_t gni_addr[2];
static struct fid_cq *send_cq;
static struct fid_cq *recv_cq;
static struct fi_cq_attr cq_attr;

#define BUF_SZ (64*1024)
char *target;
char *source;
char *uc_source;
struct fid_mr *rem_mr, *loc_mr;
uint64_t mr_key;

static struct fid_cntr *write_cntr, *read_cntr;
static struct fi_cntr_attr cntr_attr = {
	.events = FI_CNTR_EVENTS_COMP,
	.flags = 0
};
static uint64_t writes, reads, write_errs, read_errs;

void rdm_atomic_setup(void)
{
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

	ret = fi_open_ops(&dom->fid, FI_GNI_DOMAIN_OPS_1,
			  0, (void **) &gni_domain_ops, NULL);

	attr.type = FI_AV_MAP;
	attr.count = 16;

	ret = fi_av_open(dom, &attr, &av, NULL);
	cr_assert(!ret, "fi_av_open");

	ret = fi_endpoint(dom, fi, &ep[0], NULL);
	cr_assert(!ret, "fi_endpoint");

	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cq_attr.size = 1024;
	cq_attr.wait_obj = 0;

	ret = fi_cq_open(dom, &cq_attr, &send_cq, 0);
	cr_assert(!ret, "fi_cq_open");

	/*
	 * imitate shmem, etc. use FI_WRITE for bind
	 * flag
	 */
	ret = fi_ep_bind(ep[0], &send_cq->fid, FI_TRANSMIT);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_getname(&ep[0]->fid, NULL, &addrlen);
	cr_assert(addrlen > 0);

	ep_name[0] = malloc(addrlen);
	cr_assert(ep_name[0] != NULL);

	ep_name[1] = malloc(addrlen);
	cr_assert(ep_name[1] != NULL);

	ret = fi_getname(&ep[0]->fid, ep_name[0], &addrlen);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_endpoint(dom, fi, &ep[1], NULL);
	cr_assert(!ret, "fi_endpoint");

	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	ret = fi_cq_open(dom, &cq_attr, &recv_cq, 0);
	cr_assert(!ret, "fi_cq_open");

	ret = fi_ep_bind(ep[1], &recv_cq->fid, FI_RECV);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_getname(&ep[1]->fid, ep_name[1], &addrlen);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_av_insert(av, ep_name[0], 1, &gni_addr[0], 0,
				NULL);
	cr_assert(ret == 1);

	ret = fi_av_insert(av, ep_name[1], 1, &gni_addr[1], 0,
				NULL);
	cr_assert(ret == 1);

	ret = fi_ep_bind(ep[0], &av->fid, 0);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[1], &av->fid, 0);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_enable(ep[0]);
	cr_assert(!ret, "fi_ep_enable");

	ret = fi_enable(ep[1]);
	cr_assert(!ret, "fi_ep_enable");

	target = malloc(BUF_SZ);
	assert(target);

	source = malloc(BUF_SZ);
	assert(source);

	ret = fi_mr_reg(dom, target, BUF_SZ,
			FI_REMOTE_WRITE, 0, 0, 0, &rem_mr, &target);
	cr_assert_eq(ret, 0);

	ret = fi_mr_reg(dom, source, BUF_SZ,
			FI_REMOTE_WRITE, 0, 0, 0, &loc_mr, &source);
	cr_assert_eq(ret, 0);

	uc_source = malloc(BUF_SZ);
	assert(uc_source);

	mr_key = fi_mr_key(rem_mr);

	ret = fi_cntr_open(dom, &cntr_attr, &write_cntr, 0);
	cr_assert(!ret, "fi_cntr_open");

	ret = fi_ep_bind(ep[0], &write_cntr->fid, FI_WRITE);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_cntr_open(dom, &cntr_attr, &read_cntr, 0);
	cr_assert(!ret, "fi_cntr_open");

	ret = fi_ep_bind(ep[0], &read_cntr->fid, FI_READ);
	cr_assert(!ret, "fi_ep_bind");

	writes = reads = write_errs = read_errs = 0;
}

void rdm_atomic_teardown(void)
{
	int ret = 0;

	ret = fi_close(&read_cntr->fid);
	cr_assert(!ret, "failure in closing read counter.");

	ret = fi_close(&write_cntr->fid);
	cr_assert(!ret, "failure in closing write counter.");

	free(uc_source);

	ret = fi_close(&loc_mr->fid);
	cr_assert(!ret, "failure in closing av.");

	ret = fi_close(&rem_mr->fid);
	cr_assert(!ret, "failure in closing av.");

	free(target);
	free(source);

	ret = fi_close(&ep[0]->fid);
	cr_assert(!ret, "failure in closing ep.");

	ret = fi_close(&ep[1]->fid);
	cr_assert(!ret, "failure in closing ep.");

	ret = fi_close(&recv_cq->fid);
	cr_assert(!ret, "failure in recv cq.");

	ret = fi_close(&send_cq->fid);
	cr_assert(!ret, "failure in send cq.");

	ret = fi_close(&av->fid);
	cr_assert(!ret, "failure in closing av.");

	ret = fi_close(&dom->fid);
	cr_assert(!ret, "failure in closing domain.");

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "failure in closing fabric.");

	fi_freeinfo(fi);
	fi_freeinfo(hints);
	free(ep_name[0]);
	free(ep_name[1]);
}

void rdm_atomic_check_tcqe(struct fi_cq_tagged_entry *tcqe, void *ctx,
			   uint64_t flags, uint64_t data)
{
	cr_assert(tcqe->op_context == ctx, "CQE Context mismatch");
	cr_assert(tcqe->flags == flags, "CQE flags mismatch");

	if (flags & FI_REMOTE_CQ_DATA) {
		cr_assert(tcqe->data == data, "CQE data invalid");
	} else {
		cr_assert(tcqe->data == 0, "CQE data invalid");
	}

	cr_assert(tcqe->len == 0, "CQE length mismatch");
	cr_assert(tcqe->buf == 0, "CQE address mismatch");
	cr_assert(tcqe->tag == 0, "CQE tag invalid");
}

void rdm_atomic_check_cntrs(uint64_t w, uint64_t r, uint64_t w_e, uint64_t r_e)
{
	writes += w;
	reads += r;
	write_errs += w_e;
	read_errs += r_e;

	cr_assert(fi_cntr_read(write_cntr) == writes, "Bad write count");
	cr_assert(fi_cntr_read(read_cntr) == reads, "Bad read count");
	cr_assert(fi_cntr_readerr(write_cntr) == write_errs,
		  "Bad write err count");
	cr_assert(fi_cntr_readerr(read_cntr) == read_errs,
		  "Bad read err count");
}

void rdm_atomic_xfer_for_each_size(void (*xfer)(int len), int slen, int elen)
{
	int i;

	for (i = slen; i <= elen; i *= 2) {
		xfer(i);
	}
}

void rdm_atomic_err_inject_enable(void)
{
	int ret, err_count_val = 1;

	ret = gni_domain_ops->set_val(&dom->fid, GNI_ERR_INJECT_COUNT,
				      &err_count_val);
	cr_assert(!ret, "setval(GNI_ERR_INJECT_COUNT)");
}

/*******************************************************************************
 * Test RMA functions
 ******************************************************************************/

TestSuite(rdm_atomic, .init = rdm_atomic_setup, .fini = rdm_atomic_teardown,
	  .disabled = false);

#if 1
#define SOURCE_DATA	0xBBBB0000CCCCULL
#define TARGET_DATA	0xAAAA0000DDDDULL
#define SOURCE_DATA_FP	0.83203125
#define TARGET_DATA_FP	0.83984375
#else
/* swapped */
#define TARGET_DATA	0xB0000000CULL
#define SOURCE_DATA	0xA0000000DULL
#define TARGET_DATA_FP	0.83203125
#define SOURCE_DATA_FP	0.83984375
#endif
#define FETCH_SOURCE_DATA	0xACEDACEDULL
#define DATA_MASK		0xa5a5a5a5a5a5a5a5
#define U32_MASK	0xFFFFFFFFULL

#define ALL_GNI_DATATYPES_SUPPORTED	{ 0,0,0,0,1,1,1,1,1,1 }
#define GNI_DATATYPES_NO_FP_SUPPORTED	{ 0,0,0,0,1,1,1,1,0,0 }
#define NO_DATATYPES_SUPPORTED		{ }

/******************************************************************************
 *
 * Basic atomics
 *
 *****************************************************************************/

int supported_atomic_ops[FI_ATOMIC_OP_LAST][FI_DATATYPE_LAST] = {
	[FI_MIN] = { 0,0,0,0,1,0,1,0,1,1 },
	[FI_MAX] = { 0,0,0,0,1,0,1,0,1,1 },
	[FI_SUM] = { 0,0,0,0,1,1,1,1,1,0 }, /* GNI DP sum is broken */
	[FI_PROD] = NO_DATATYPES_SUPPORTED,
	[FI_LOR] = NO_DATATYPES_SUPPORTED,
	[FI_LAND] = NO_DATATYPES_SUPPORTED,
	[FI_BOR] = GNI_DATATYPES_NO_FP_SUPPORTED,
	[FI_BAND] = GNI_DATATYPES_NO_FP_SUPPORTED,
	[FI_LXOR] = NO_DATATYPES_SUPPORTED,
	[FI_BXOR] = GNI_DATATYPES_NO_FP_SUPPORTED,
	[FI_ATOMIC_READ] = NO_DATATYPES_SUPPORTED,
	[FI_ATOMIC_WRITE] = ALL_GNI_DATATYPES_SUPPORTED,
	[FI_CSWAP] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_NE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_LE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_LT] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_GE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_GT] = NO_DATATYPES_SUPPORTED,
	[FI_MSWAP] = NO_DATATYPES_SUPPORTED,
};

void do_invalid_atomic(enum fi_datatype dt, enum fi_op op)
{
	ssize_t sz;
	size_t count;

	if (!supported_atomic_ops[op][dt]) {
		sz = fi_atomic(ep[0], source, 1,
				 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
				 dt, op, target);

		cr_assert(sz == -FI_ENOENT);

		sz = fi_atomicvalid(ep[0], dt, op, &count);
		cr_assert(sz == -FI_ENOENT, "fi_atomicvalid() succeeded\n");
	} else {
		sz = fi_atomicvalid(ep[0], dt, op, &count);
		cr_assert(!sz, "fi_atomicvalid() failed\n");
		cr_assert(count == 1, "fi_atomicvalid(): bad count \n");
	}
}

Test(rdm_atomic, invalid_atomic)
{
	int i, j;

	for(i = 0; i < FI_ATOMIC_OP_LAST; i++) {
		for(j = 0; j < FI_DATATYPE_LAST; j++) {
			do_invalid_atomic(j, i);
		}
	}
}

void do_min(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;

	/* i64 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");

	/* float */
	*((float *)source) = SOURCE_DATA_FP;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((float *)target) == min_fp;
	cr_assert(ret, "Data mismatch");

	/* double */
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((double *)target) == min_dp;
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, min)
{
	rdm_atomic_xfer_for_each_size(do_min, 1, 1);
}

void do_max(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;

	/* i64 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_MAX, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int64_t)SOURCE_DATA > (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_MAX, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int32_t)SOURCE_DATA > (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");

	/* float */
	*((float *)source) = SOURCE_DATA_FP;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_MAX, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min_fp = (float)SOURCE_DATA_FP > (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((float *)target) == min_fp;
	cr_assert(ret, "Data mismatch");

	/* double */
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_MAX, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min_dp = (double)SOURCE_DATA_FP > (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((double *)target) == min_dp;
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, max)
{
	rdm_atomic_xfer_for_each_size(do_max, 1, 1);
}

void do_sum(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;

	/* u64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == (SOURCE_DATA + TARGET_DATA);
	cr_assert(ret, "Data mismatch");

	/* U32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) + TARGET_DATA);
	cr_assert(ret, "Data mismatch");

	/* i64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == (SOURCE_DATA + TARGET_DATA);
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) + TARGET_DATA);
	cr_assert(ret, "Data mismatch");

	/* float */
	*((float *)source) = SOURCE_DATA_FP;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) ==
			(float)(SOURCE_DATA_FP + TARGET_DATA_FP);
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, sum)
{
	rdm_atomic_xfer_for_each_size(do_sum, 1, 1);
}

void do_bor(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t res;

	/* u64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_BOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA | TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* U32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_BOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA | TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* i64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_BOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA | TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_BOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA | TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, bor)
{
	rdm_atomic_xfer_for_each_size(do_bor, 1, 1);
}

void do_band(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t res;

	/* u64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_BAND, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA & TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* U32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_BAND, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA & TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* i64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_BAND, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA & TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_BAND, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA & TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, band)
{
	rdm_atomic_xfer_for_each_size(do_band, 1, 1);
}

void do_bxor(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t res;

	/* u64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_BXOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA ^ TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* U32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_BXOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA ^ TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* i64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_BXOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA ^ TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_BXOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA ^ TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, bxor)
{
	rdm_atomic_xfer_for_each_size(do_bxor, 1, 1);
}

void do_atomic_write(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;

	/* u64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");

	/* U32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");

	/* i64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");

	/* float */
	*((float *)source) = SOURCE_DATA_FP;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) == (float)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");

	/* double */
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], source, 1,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((double *)target) == (double)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, write)
{
	rdm_atomic_xfer_for_each_size(do_atomic_write, 1, 1);
}

void do_min_buf(void *s, void *t)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;

	/* i64 */
	*((int64_t *)s) = SOURCE_DATA;
	*((int64_t *)t) = TARGET_DATA;
	sz = fi_atomic(ep[0], s, 1,
			 loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_INT64, FI_MIN, t);
	if ((uint64_t)t & 0x7) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_WRITE, 0);
		rdm_atomic_check_cntrs(1, 0, 0, 0);

		dbg_printf("got write context event!\n");

		min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
		ret = *((int64_t *)t) == min;
		cr_assert(ret, "Data mismatch");
	}

	/* i32 */
	*((int64_t *)s) = SOURCE_DATA;
	*((int64_t *)t) = TARGET_DATA;
	sz = fi_atomic(ep[0], s, 1,
			 loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_INT32, FI_MIN, t);
	if ((uint64_t)t & 0x3) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_WRITE, 0);
		rdm_atomic_check_cntrs(1, 0, 0, 0);

		dbg_printf("got write context event!\n");

		min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
				SOURCE_DATA : TARGET_DATA;
		min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
		ret = *((int64_t *)t) == min;
		cr_assert(ret, "Data mismatch");
	}

	/* float */
	*((float *)s) = SOURCE_DATA_FP;
	*((float *)t) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], s, 1,
			 loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_FLOAT, FI_MIN, t);
	if ((uint64_t)t & 0x3) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_WRITE, 0);
		rdm_atomic_check_cntrs(1, 0, 0, 0);

		dbg_printf("got write context event!\n");

		min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
				SOURCE_DATA_FP : TARGET_DATA_FP;
		ret = *((float *)t) == min_fp;
		cr_assert(ret, "Data mismatch");
	}

	/* double */
	*((double *)s) = SOURCE_DATA_FP;
	*((double *)t) = TARGET_DATA_FP;
	sz = fi_atomic(ep[0], s, 1,
			 loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_DOUBLE, FI_MIN, t);
	if ((uint64_t)t & 0x7) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_WRITE, 0);
		rdm_atomic_check_cntrs(1, 0, 0, 0);

		dbg_printf("got write context event!\n");

		min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
				SOURCE_DATA_FP : TARGET_DATA_FP;
		ret = *((double *)t) == min_dp;
		cr_assert(ret, "Data mismatch");
	}
}

Test(rdm_atomic, atomic_alignment)
{
	int s_off, t_off;

	for (s_off = 0; s_off < 7; s_off++) {
		for (t_off = 0; t_off < 7; t_off++) {
			do_min_buf(source + s_off, target + t_off);
		}
	}
}


Test(rdm_atomic, atomicv)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;
	struct fi_ioc iov;

	iov.addr = source;
	iov.count = 1;

	/* i64 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_atomicv(ep[0], &iov, (void **)&loc_mr, 1, gni_addr[1],
			(uint64_t)target, mr_key, FI_INT64, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_atomicv(ep[0], &iov, (void **)&loc_mr, 1, gni_addr[1],
			(uint64_t)target, mr_key, FI_INT32, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");

	/* float */
	*((float *)source) = SOURCE_DATA_FP;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_atomicv(ep[0], &iov, (void **)&loc_mr, 1, gni_addr[1],
			(uint64_t)target, mr_key, FI_FLOAT, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((float *)target) == min_fp;
	cr_assert(ret, "Data mismatch");

	/* double */
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_atomicv(ep[0], &iov, (void **)&loc_mr, 1, gni_addr[1],
			(uint64_t)target, mr_key, FI_DOUBLE, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((double *)target) == min_dp;
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, atomicmsg)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov;
	struct fi_rma_ioc rma_iov;

	msg_iov.addr = source;
	msg_iov.count = 1;
	msg.msg_iov = &msg_iov;
	msg.desc = (void **)&loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	rma_iov.addr = (uint64_t)target;
	rma_iov.count = 1;
	rma_iov.key = mr_key;
	msg.rma_iov = &rma_iov;
	msg.context = target;
	msg.op = FI_MIN;

	/* i64 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	msg.datatype = FI_INT64;
	sz = fi_atomicmsg(ep[0], &msg, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");

	/* i32 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	msg.datatype = FI_INT32;
	sz = fi_atomicmsg(ep[0], &msg, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");

	/* float */
	*((float *)source) = SOURCE_DATA_FP;
	*((float *)target) = TARGET_DATA_FP;
	msg.datatype = FI_FLOAT;
	sz = fi_atomicmsg(ep[0], &msg, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((float *)target) == min_fp;
	cr_assert(ret, "Data mismatch");

	/* double */
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	msg.datatype = FI_DOUBLE;
	sz = fi_atomicmsg(ep[0], &msg, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);
	rdm_atomic_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((double *)target) == min_dp;
	cr_assert(ret, "Data mismatch");
}

Test(rdm_atomic, atomicinject)
{
	int ret, loops;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;

	/* i64 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_inject_atomic(ep[0], source, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_MIN);
	cr_assert_eq(sz, 0);

	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	loops = 0;
	while (*((int64_t *)target) != min) {
		ret = fi_cq_read(send_cq, &cqe, 1); /* for progress */
		cr_assert(ret == -EAGAIN,
			  "Received unexpected event\n");

		pthread_yield();
		cr_assert(++loops < 10000, "Data mismatch");
	}

	/* i32 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_inject_atomic(ep[0], source, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_MIN);
	cr_assert_eq(sz, 0);

	min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	loops = 0;
	while (*((int64_t *)target) != min) {
		ret = fi_cq_read(send_cq, &cqe, 1); /* for progress */
		cr_assert(ret == -EAGAIN,
			  "Received unexpected event\n");

		pthread_yield();
		cr_assert(++loops < 10000, "Data mismatch");
	}

	/* float */
	*((float *)source) = SOURCE_DATA_FP;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_inject_atomic(ep[0], source, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_MIN);
	cr_assert_eq(sz, 0);

	min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	loops = 0;
	while (*((float *)target) != min_fp) {
		ret = fi_cq_read(send_cq, &cqe, 1); /* for progress */
		cr_assert(ret == -EAGAIN,
			  "Received unexpected event\n");

		pthread_yield();
		cr_assert(++loops < 10000, "Data mismatch");
	}

	/* double */
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_inject_atomic(ep[0], source, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_MIN);
	cr_assert_eq(sz, 0);

	min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	loops = 0;
	while (*((double *)target) != min_dp) {
		ret = fi_cq_read(send_cq, &cqe, 1); /* for progress */
		cr_assert(ret == -EAGAIN,
			  "Received unexpected event\n");

		pthread_yield();
		cr_assert(++loops < 10000, "Data mismatch");
	}
}

/******************************************************************************
 *
 * Fetching atomics
 *
 *****************************************************************************/

int supported_fetch_atomic_ops[FI_ATOMIC_OP_LAST][FI_DATATYPE_LAST] = {
	[FI_MIN] = { 0,0,0,0,1,0,1,0,1,1 },
	[FI_MAX] = { 0,0,0,0,1,0,1,0,1,1 },
	[FI_SUM] = { 0,0,0,0,1,1,1,1,1,0 }, /* GNI DP sum is broken */
	[FI_PROD] = NO_DATATYPES_SUPPORTED,
	[FI_LOR] = NO_DATATYPES_SUPPORTED,
	[FI_LAND] = NO_DATATYPES_SUPPORTED,
	[FI_BOR] = GNI_DATATYPES_NO_FP_SUPPORTED,
	[FI_BAND] = GNI_DATATYPES_NO_FP_SUPPORTED,
	[FI_LXOR] = NO_DATATYPES_SUPPORTED,
	[FI_BXOR] = GNI_DATATYPES_NO_FP_SUPPORTED,
	[FI_ATOMIC_READ] = ALL_GNI_DATATYPES_SUPPORTED,
	[FI_ATOMIC_WRITE] = ALL_GNI_DATATYPES_SUPPORTED,
	[FI_CSWAP] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_NE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_LE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_LT] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_GE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_GT] = NO_DATATYPES_SUPPORTED,
	[FI_MSWAP] = NO_DATATYPES_SUPPORTED,
};

void do_invalid_fetch_atomic(enum fi_datatype dt, enum fi_op op)
{
	ssize_t sz;
	size_t count;
	uint64_t operand;

	if (!supported_fetch_atomic_ops[op][dt]) {
		sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
				 source, loc_mr,
				 gni_addr[1], (uint64_t)target, mr_key,
				 dt, op, target);
		cr_assert(sz == -FI_ENOENT);

		sz = fi_fetch_atomicvalid(ep[0], dt, op, &count);
		cr_assert(sz == -FI_ENOENT, "fi_atomicvalid() succeeded\n");
	} else {
		sz = fi_fetch_atomicvalid(ep[0], dt, op, &count);
		cr_assert(!sz, "fi_atomicvalid() failed\n");
		cr_assert(count == 1, "fi_atomicvalid(): bad count \n");
	}
}

Test(rdm_atomic, invalid_fetch_atomic)
{
	int i, j;

	for(i = 0; i < FI_ATOMIC_OP_LAST; i++) {
		for(j = 0; j < FI_DATATYPE_LAST; j++) {
			do_invalid_fetch_atomic(j, i);
		}
	}
}

void do_fetch_min(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;
	uint64_t operand = SOURCE_DATA;
	float operand_fp;
	double operand_dp;

	/* i64 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	ret = *((int64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	min = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((int64_t *)source) == min;
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_fp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((float *)target) == min_fp;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_dp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((double *)target) == min_dp;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, fetch_min)
{
	rdm_atomic_xfer_for_each_size(do_fetch_min, 1, 1);
}

void do_fetch_max(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;
	uint64_t operand = SOURCE_DATA;
	float operand_fp;
	double operand_dp;

	/* i64 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_MAX, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int64_t)SOURCE_DATA > (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	ret = *((int64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_MAX, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int32_t)SOURCE_DATA > (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	min = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((int64_t *)source) == min;
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_fp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_MAX, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min_fp = (float)SOURCE_DATA_FP > (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((float *)target) == min_fp;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_dp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_MAX, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min_dp = (double)SOURCE_DATA_FP > (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((double *)target) == min_dp;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, fetch_max)
{
	rdm_atomic_xfer_for_each_size(do_fetch_max, 1, 1);
}

void do_fetch_sum(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand = SOURCE_DATA;
	float operand_fp;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == (SOURCE_DATA + TARGET_DATA);
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) + TARGET_DATA);
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == (SOURCE_DATA + TARGET_DATA);
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) + TARGET_DATA);
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
			           (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_fp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) ==
			(float)(SOURCE_DATA_FP + TARGET_DATA_FP);
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, fetch_sum)
{
	rdm_atomic_xfer_for_each_size(do_fetch_sum, 1, 1);
}

void do_fetch_bor(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t res;
	uint64_t operand = SOURCE_DATA;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_BOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA | TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_BOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA | TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	res = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)source) == res;
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_BOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA | TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_BOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA | TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	res = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)source) == res;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, fetch_bor)
{
	rdm_atomic_xfer_for_each_size(do_fetch_bor, 1, 1);
}

void do_fetch_band(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t res;
	uint64_t operand = SOURCE_DATA;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_BAND, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA & TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_BAND, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA & TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	res = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)source) == res;
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_BAND, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA & TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_BAND, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA & TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	res = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)source) == res;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, fetch_band)
{
	rdm_atomic_xfer_for_each_size(do_fetch_band, 1, 1);
}

void do_fetch_bxor(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t res;
	uint64_t operand = SOURCE_DATA;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_BXOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA ^ TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_BXOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA ^ TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	res = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)source) == res;
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_BXOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA ^ TARGET_DATA;
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_BXOR, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = SOURCE_DATA ^ TARGET_DATA;
	res = (res & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	res = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((uint64_t *)source) == res;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, fetch_bxor)
{
	rdm_atomic_xfer_for_each_size(do_fetch_bxor, 1, 1);
}

void do_fetch_atomic_write(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand = SOURCE_DATA;
	float operand_fp;
	double operand_dp;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = 0;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_fp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) == (float)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*(double *)&operand_dp = SOURCE_DATA_FP;
	*((double *)source) = FETCH_SOURCE_DATA;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_dp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((double *)target) == (double)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, fetch_atomic_write)
{
	rdm_atomic_xfer_for_each_size(do_fetch_atomic_write, 1, 1);
}

void do_fetch_atomic_read(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand;
	float operand_fp;
	double operand_dp;

	/* u64 */
	operand = SOURCE_DATA;
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_ATOMIC_READ, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == TARGET_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == (uint64_t)TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");
	ret = *((uint64_t *)&operand) == TARGET_DATA;
	cr_assert(ret, "Read data mismatch");

	/* U32 */
	operand = SOURCE_DATA;
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_ATOMIC_READ, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == TARGET_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");
	ret = *((uint64_t *)&operand) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Read data mismatch");

	/* i64 */
	operand = SOURCE_DATA;
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_ATOMIC_READ, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == TARGET_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");
	ret = *((uint64_t *)&operand) == TARGET_DATA;
	cr_assert(ret, "Read data mismatch");

	/* i32 */
	operand = SOURCE_DATA;
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_ATOMIC_READ, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == TARGET_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");
	ret = *((uint64_t *)&operand) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Read data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_fp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_ATOMIC_READ, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
	ret = *((float *)&operand_fp) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Read data mismatch");

	/* double */
	*(double *)&operand_dp = SOURCE_DATA_FP;
	*((double *)source) = FETCH_SOURCE_DATA;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_dp, 1, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_ATOMIC_READ, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((double *)target) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
	ret = *((double *)&operand_dp) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Read data mismatch");
}

Test(rdm_atomic, fetch_atomic_read)
{
	rdm_atomic_xfer_for_each_size(do_fetch_atomic_read, 1, 1);
}

void do_fetch_min_buf(void *s, void *t)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;
	uint64_t operand = SOURCE_DATA;
	float operand_fp;
	double operand_dp;

	/* i64 */
	*((int64_t *)s) = FETCH_SOURCE_DATA;
	*((int64_t *)t) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_INT64, FI_MIN, t);
	if ((uint64_t)s & 0x7 || (uint64_t)t & 0x7) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
				SOURCE_DATA : TARGET_DATA;
		ret = *((int64_t *)t) == min;
		cr_assert(ret, "Data mismatch");
		ret = *((int64_t *)s) == TARGET_DATA;
		cr_assert(ret, "Fetch data mismatch");
	}

	/* i32 */
	*((int64_t *)s) = FETCH_SOURCE_DATA;
	*((int64_t *)t) = TARGET_DATA;
	sz = fi_fetch_atomic(ep[0], &operand, 1, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_INT32, FI_MIN, t);
	if ((uint64_t)s & 0x3 || (uint64_t)t & 0x3) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
				SOURCE_DATA : TARGET_DATA;
		min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
		ret = *((int64_t *)t) == min;
		cr_assert(ret, "Data mismatch");
		min = (TARGET_DATA & U32_MASK) |
				(FETCH_SOURCE_DATA & (U32_MASK << 32));
		ret = *((int64_t *)s) == min;
		cr_assert(ret, "Fetch data mismatch");
	}

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)s) = FETCH_SOURCE_DATA;
	*((float *)t) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_fp, 1, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_FLOAT, FI_MIN, t);
	if ((uint64_t)s & 0x3 || (uint64_t)t & 0x3) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
				SOURCE_DATA_FP : TARGET_DATA_FP;
		ret = *((float *)t) == min_fp;
		cr_assert(ret, "Data mismatch");
		ret = *((float *)s) == (float)TARGET_DATA_FP;
		cr_assert(ret, "Fetch data mismatch");
	}

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)s) = SOURCE_DATA_FP;
	*((double *)t) = TARGET_DATA_FP;
	sz = fi_fetch_atomic(ep[0], &operand_dp, 1, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_DOUBLE, FI_MIN, t);
	if ((uint64_t)s & 0x7 || (uint64_t)t & 0x7) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
				SOURCE_DATA_FP : TARGET_DATA_FP;
		ret = *((double *)t) == min_dp;
		cr_assert(ret, "Data mismatch");
		ret = *((double *)s) == (double)TARGET_DATA_FP;
		cr_assert(ret, "Fetch data mismatch");
	}
}

Test(rdm_atomic, atomic_fetch_alignment)
{
	int s_off, t_off;

	for (s_off = 0; s_off < 7; s_off++) {
		for (t_off = 0; t_off < 7; t_off++) {
			do_fetch_min_buf(source + s_off, target + t_off);
		}
	}
}

Test(rdm_atomic, fatomicv)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;
	uint64_t operand = SOURCE_DATA;
	float operand_fp;
	double operand_dp;
	struct fi_ioc iov, r_iov;

	iov.count = 1;
	r_iov.count = 1;

	/* i64 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	iov.addr = &operand;
	r_iov.addr = source;
	sz = fi_fetch_atomicv(ep[0], &iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	ret = *((int64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	iov.addr = &operand;
	r_iov.addr = source;
	sz = fi_fetch_atomicv(ep[0], &iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	min = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((int64_t *)source) == min;
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	iov.addr = &operand_fp;
	r_iov.addr = source;
	sz = fi_fetch_atomicv(ep[0], &iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((float *)target) == min_fp;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	iov.addr = &operand_dp;
	r_iov.addr = source;
	sz = fi_fetch_atomicv(ep[0], &iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((double *)target) == min_dp;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, fatomicmsg)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;
	float min_fp;
	double min_dp;
	uint64_t operand = SOURCE_DATA;
	float operand_fp;
	double operand_dp;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov, res_iov;
	struct fi_rma_ioc rma_iov;

	msg_iov.count = 1;
	msg.msg_iov = &msg_iov;
	msg.desc = (void **)&loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	rma_iov.addr = (uint64_t)target;
	rma_iov.count = 1;
	rma_iov.key = mr_key;
	msg.rma_iov = &rma_iov;
	msg.context = target;
	msg.op = FI_MIN;

	res_iov.addr = source;
	res_iov.count = 1;

	/* i64 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	msg_iov.addr = &operand;
	msg.datatype = FI_INT64;
	sz = fi_fetch_atomicmsg(ep[0], &msg, &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	ret = *((int64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	msg_iov.addr = &operand;
	msg.datatype = FI_INT32;
	sz = fi_fetch_atomicmsg(ep[0], &msg, &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
			SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	min = (TARGET_DATA & U32_MASK) | (FETCH_SOURCE_DATA & (U32_MASK << 32));
	ret = *((int64_t *)source) == min;
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	msg_iov.addr = &operand_fp;
	msg.datatype = FI_FLOAT;
	sz = fi_fetch_atomicmsg(ep[0], &msg, &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min_fp = (float)SOURCE_DATA_FP < (float)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((float *)target) == min_fp;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)source) = SOURCE_DATA_FP;
	*((double *)target) = TARGET_DATA_FP;
	msg_iov.addr = &operand_dp;
	msg.datatype = FI_DOUBLE;
	sz = fi_fetch_atomicmsg(ep[0], &msg, &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	min_dp = (double)SOURCE_DATA_FP < (double)TARGET_DATA_FP ?
			SOURCE_DATA_FP : TARGET_DATA_FP;
	ret = *((double *)target) == min_dp;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

/******************************************************************************
 *
 * Compare atomics
 *
 *****************************************************************************/

int supported_compare_atomic_ops[FI_ATOMIC_OP_LAST][FI_DATATYPE_LAST] = {
	[FI_MIN] = NO_DATATYPES_SUPPORTED,
	[FI_MAX] = NO_DATATYPES_SUPPORTED,
	[FI_SUM] = NO_DATATYPES_SUPPORTED,
	[FI_PROD] = NO_DATATYPES_SUPPORTED,
	[FI_LOR] = NO_DATATYPES_SUPPORTED,
	[FI_LAND] = NO_DATATYPES_SUPPORTED,
	[FI_BOR] = NO_DATATYPES_SUPPORTED,
	[FI_BAND] = NO_DATATYPES_SUPPORTED,
	[FI_LXOR] = NO_DATATYPES_SUPPORTED,
	[FI_BXOR] = NO_DATATYPES_SUPPORTED,
	[FI_ATOMIC_READ] = NO_DATATYPES_SUPPORTED,
	[FI_ATOMIC_WRITE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP] = ALL_GNI_DATATYPES_SUPPORTED,
	[FI_CSWAP_NE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_LE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_LT] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_GE] = NO_DATATYPES_SUPPORTED,
	[FI_CSWAP_GT] = NO_DATATYPES_SUPPORTED,
	[FI_MSWAP] = ALL_GNI_DATATYPES_SUPPORTED,
};

void do_invalid_compare_atomic(enum fi_datatype dt, enum fi_op op)
{
	ssize_t sz;
	size_t count;
	uint64_t operand, op2;

	if (!supported_compare_atomic_ops[op][dt]) {
		sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
				 source, loc_mr,
				 gni_addr[1], (uint64_t)target, mr_key,
				 dt, op, target);
		cr_assert(sz == -FI_ENOENT);

		sz = fi_compare_atomicvalid(ep[0], dt, op, &count);
		cr_assert(sz == -FI_ENOENT, "fi_atomicvalid() succeeded\n");
	} else {
		sz = fi_compare_atomicvalid(ep[0], dt, op, &count);
		cr_assert(!sz, "fi_atomicvalid() failed\n");
		cr_assert(count == 1, "fi_atomicvalid(): bad count \n");
	}
}

Test(rdm_atomic, invalid_compare_atomic)
{
	int i, j;

	for(i = 0; i < FI_ATOMIC_OP_LAST; i++) {
		for(j = 0; j < FI_DATATYPE_LAST; j++) {
			do_invalid_compare_atomic(j, i);
		}
	}
}

void do_cswap(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand = SOURCE_DATA, op2 = TARGET_DATA;
	float operand_fp, op2_fp;
	double operand_dp, op2_dp;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)&op2_fp) = TARGET_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_compare_atomic(ep[0], &operand_fp, 1, NULL, &op2_fp, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) == (float)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)&op2_dp) = TARGET_DATA_FP;
	*((double *)source) = FETCH_SOURCE_DATA;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_compare_atomic(ep[0], &operand_dp, 1, NULL, &op2_dp, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((double *)target) == (double)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, cswap)
{
	rdm_atomic_xfer_for_each_size(do_cswap, 1, 1);
}

void do_mswap(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t res;
	uint64_t operand = SOURCE_DATA, op2 = DATA_MASK;
	float operand_fp, op2_fp;
	double operand_dp, op2_dp;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_MSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = (SOURCE_DATA & DATA_MASK) | (TARGET_DATA & ~DATA_MASK);
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_MSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = (SOURCE_DATA & DATA_MASK) | (TARGET_DATA & ~DATA_MASK);
	ret = *((uint64_t *)target) ==
			(uint64_t)((res & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_MSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = (SOURCE_DATA & DATA_MASK) | (TARGET_DATA & ~DATA_MASK);
	ret = *((uint64_t *)target) == res;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_MSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	res = (SOURCE_DATA & DATA_MASK) | (TARGET_DATA & ~DATA_MASK);
	ret = *((uint64_t *)target) ==
			(uint64_t)((res & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)&op2_fp) = TARGET_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	sz = fi_compare_atomic(ep[0], &operand_fp, 1, NULL, &op2_fp, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_MSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) == (float)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)&op2_dp) = TARGET_DATA_FP;
	*((double *)source) = FETCH_SOURCE_DATA;
	*((double *)target) = TARGET_DATA_FP;
	sz = fi_compare_atomic(ep[0], &operand_dp, 1, NULL, &op2_dp, NULL,
			 source, loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_MSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((double *)target) == (double)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, mswap)
{
	rdm_atomic_xfer_for_each_size(do_mswap, 1, 1);
}

void do_cswap_buf(void *s, void *t)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand = SOURCE_DATA, op2 = TARGET_DATA;
	float operand_fp, op2_fp;
	double operand_dp, op2_dp;

	/* u64 */
	*((uint64_t *)s) = FETCH_SOURCE_DATA;
	*((uint64_t *)t) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_UINT64, FI_CSWAP, t);
	if ((uint64_t)s & 0x7 || (uint64_t)t & 0x7) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		ret = *((uint64_t *)t) == SOURCE_DATA;
		cr_assert(ret, "Data mismatch");
		ret = *((uint64_t *)s) == TARGET_DATA;
		cr_assert(ret, "Fetch data mismatch");
	}

	/* U32 */
	*((uint64_t *)s) = FETCH_SOURCE_DATA;
	*((uint64_t *)t) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_UINT32, FI_CSWAP, t);
	if ((uint64_t)s & 0x3 || (uint64_t)t & 0x3) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		ret = *((uint64_t *)t) ==
				(uint64_t)((SOURCE_DATA & U32_MASK) |
					   (TARGET_DATA & (U32_MASK << 32)));
		cr_assert(ret, "Data mismatch");
		ret = *((uint64_t *)s) ==
				(uint64_t)((TARGET_DATA & U32_MASK) |
					   (FETCH_SOURCE_DATA &
					    (U32_MASK << 32)));
		cr_assert(ret, "Fetch data mismatch");
	}

	/* i64 */
	*((uint64_t *)s) = FETCH_SOURCE_DATA;
	*((uint64_t *)t) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_INT64, FI_CSWAP, t);
	if ((uint64_t)s & 0x7 || (uint64_t)t & 0x7) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		ret = *((uint64_t *)t) == SOURCE_DATA;
		cr_assert(ret, "Data mismatch");
		ret = *((uint64_t *)s) == TARGET_DATA;
		cr_assert(ret, "Fetch data mismatch");
	}

	/* i32 */
	*((uint64_t *)s) = FETCH_SOURCE_DATA;
	*((uint64_t *)t) = TARGET_DATA;
	sz = fi_compare_atomic(ep[0], &operand, 1, NULL, &op2, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_INT32, FI_CSWAP, t);
	if ((uint64_t)s & 0x3 || (uint64_t)t & 0x3) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		ret = *((uint64_t *)t) ==
				(uint64_t)((SOURCE_DATA & U32_MASK) |
					   (TARGET_DATA & (U32_MASK << 32)));
		cr_assert(ret, "Data mismatch");
		ret = *((uint64_t *)s) ==
				(uint64_t)((TARGET_DATA & U32_MASK) |
					   (FETCH_SOURCE_DATA &
					    (U32_MASK << 32)));
		cr_assert(ret, "Fetch data mismatch");
	}

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)&op2_fp) = TARGET_DATA_FP;
	*((float *)s) = FETCH_SOURCE_DATA;
	*((float *)t) = TARGET_DATA_FP;
	sz = fi_compare_atomic(ep[0], &operand_fp, 1, NULL, &op2_fp, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_FLOAT, FI_CSWAP, t);
	if ((uint64_t)s & 0x3 || (uint64_t)t & 0x3) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		ret = *((float *)t) == (float)SOURCE_DATA_FP;
		cr_assert(ret, "Data mismatch");
		ret = *((float *)s) == (float)TARGET_DATA_FP;
		cr_assert(ret, "Fetch data mismatch");
	}

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)&op2_dp) = TARGET_DATA_FP;
	*((double *)s) = FETCH_SOURCE_DATA;
	*((double *)t) = TARGET_DATA_FP;
	sz = fi_compare_atomic(ep[0], &operand_dp, 1, NULL, &op2_dp, NULL,
			 s, loc_mr, gni_addr[1], (uint64_t)t, mr_key,
			 FI_DOUBLE, FI_CSWAP, t);
	if ((uint64_t)s & 0x7 || (uint64_t)t & 0x7) {
		cr_assert_eq(sz, -FI_EINVAL);
	} else {
		cr_assert_eq(sz, 0);

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
			pthread_yield();
		}

		cr_assert_eq(ret, 1);
		rdm_atomic_check_tcqe(&cqe, t, FI_ATOMIC | FI_READ, 0);
		rdm_atomic_check_cntrs(0, 1, 0, 0);

		dbg_printf("got write context event!\n");

		ret = *((double *)t) == (double)SOURCE_DATA_FP;
		cr_assert(ret, "Data mismatch");
		ret = *((double *)s) == (double)TARGET_DATA_FP;
		cr_assert(ret, "Fetch data mismatch");
	}
}

Test(rdm_atomic, atomic_compare_alignment)
{
	int s_off, t_off;

	for (s_off = 0; s_off < 7; s_off++) {
		for (t_off = 0; t_off < 7; t_off++) {
			do_cswap_buf(source + s_off, target + t_off);
		}
	}
}

Test(rdm_atomic, catomicv)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand = SOURCE_DATA, op2 = TARGET_DATA;
	float operand_fp, op2_fp;
	double operand_dp, op2_dp;
	struct fi_ioc iov, r_iov, c_iov;

	iov.count = 1;
	r_iov.count = 1;
	c_iov.count = 1;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	iov.addr = &operand;
	r_iov.addr = source;
	c_iov.addr = &op2;
	sz = fi_compare_atomicv(ep[0],
			 &iov, NULL, 1,
			 &c_iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT64, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	iov.addr = &operand;
	r_iov.addr = source;
	c_iov.addr = &op2;
	sz = fi_compare_atomicv(ep[0],
			 &iov, NULL, 1,
			 &c_iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_UINT32, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	iov.addr = &operand;
	r_iov.addr = source;
	c_iov.addr = &op2;
	sz = fi_compare_atomicv(ep[0],
			 &iov, NULL, 1,
			 &c_iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT64, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	iov.addr = &operand;
	r_iov.addr = source;
	c_iov.addr = &op2;
	sz = fi_compare_atomicv(ep[0],
			 &iov, NULL, 1,
			 &c_iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_INT32, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)&op2_fp) = TARGET_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	iov.addr = &operand_fp;
	r_iov.addr = source;
	c_iov.addr = &op2_fp;
	sz = fi_compare_atomicv(ep[0],
			 &iov, NULL, 1,
			 &c_iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_FLOAT, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) == (float)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)&op2_dp) = TARGET_DATA_FP;
	*((double *)source) = FETCH_SOURCE_DATA;
	*((double *)target) = TARGET_DATA_FP;
	iov.addr = &operand_dp;
	r_iov.addr = source;
	c_iov.addr = &op2_dp;
	sz = fi_compare_atomicv(ep[0],
			 &iov, NULL, 1,
			 &c_iov, NULL, 1,
			 &r_iov, (void **)&loc_mr, 1,
			 gni_addr[1], (uint64_t)target, mr_key,
			 FI_DOUBLE, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((double *)target) == (double)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(rdm_atomic, catomicmsg)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand = SOURCE_DATA, op2 = TARGET_DATA;
	float operand_fp, op2_fp;
	double operand_dp, op2_dp;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov, res_iov, cmp_iov;
	struct fi_rma_ioc rma_iov;

	msg_iov.count = 1;
	msg.msg_iov = &msg_iov;
	msg.desc = (void **)&loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	rma_iov.addr = (uint64_t)target;
	rma_iov.count = 1;
	rma_iov.key = mr_key;
	msg.rma_iov = &rma_iov;
	msg.context = target;
	msg.op = FI_CSWAP;

	res_iov.count = 1;
	cmp_iov.count = 1;

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	msg_iov.addr = &operand;
	msg.datatype = FI_UINT64;
	res_iov.addr = source;
	cmp_iov.addr = &op2;
	sz = fi_compare_atomicmsg(ep[0], &msg, &cmp_iov, NULL, 1,
				  &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* U32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	msg_iov.addr = &operand;
	msg.datatype = FI_UINT32;
	res_iov.addr = source;
	cmp_iov.addr = &op2;
	sz = fi_compare_atomicmsg(ep[0], &msg, &cmp_iov, NULL, 1,
				  &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	msg_iov.addr = &operand;
	msg.datatype = FI_INT64;
	res_iov.addr = source;
	cmp_iov.addr = &op2;
	sz = fi_compare_atomicmsg(ep[0], &msg, &cmp_iov, NULL, 1,
				  &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");

	/* i32 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	msg_iov.addr = &operand;
	msg.datatype = FI_INT32;
	res_iov.addr = source;
	cmp_iov.addr = &op2;
	sz = fi_compare_atomicmsg(ep[0], &msg, &cmp_iov, NULL, 1,
				  &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((uint64_t *)target) ==
			(uint64_t)((SOURCE_DATA & U32_MASK) |
				   (TARGET_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) ==
			(uint64_t)((TARGET_DATA & U32_MASK) |
				   (FETCH_SOURCE_DATA & (U32_MASK << 32)));
	cr_assert(ret, "Fetch data mismatch");

	/* float */
	*((float *)&operand_fp) = SOURCE_DATA_FP;
	*((float *)&op2_fp) = TARGET_DATA_FP;
	*((float *)source) = FETCH_SOURCE_DATA;
	*((float *)target) = TARGET_DATA_FP;
	msg_iov.addr = &operand_fp;
	msg.datatype = FI_FLOAT;
	res_iov.addr = source;
	cmp_iov.addr = &op2_fp;
	sz = fi_compare_atomicmsg(ep[0], &msg, &cmp_iov, NULL, 1,
				  &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((float *)target) == (float)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((float *)source) == (float)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)&op2_dp) = TARGET_DATA_FP;
	*((double *)source) = FETCH_SOURCE_DATA;
	*((double *)target) = TARGET_DATA_FP;
	msg_iov.addr = &operand_dp;
	msg.datatype = FI_DOUBLE;
	res_iov.addr = source;
	cmp_iov.addr = &op2_dp;
	sz = fi_compare_atomicmsg(ep[0], &msg, &cmp_iov, NULL, 1,
				  &res_iov, (void **)&loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_atomic_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	rdm_atomic_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	ret = *((double *)target) == (double)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

