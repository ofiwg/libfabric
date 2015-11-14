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

void rdm_rma_setup(void)
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

void rdm_rma_teardown(void)
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

void init_data(char *buf, int len, char seed)
{
	int i;

	for (i = 0; i < len; i++) {
		buf[i] = seed++;
	}
}

int check_data(char *buf1, char *buf2, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (buf1[i] != buf2[i]) {
			printf("data mismatch, elem: %d, b1: 0x%hhx, b2: 0x%hhx, len: %d\n",
			       i, buf1[i], buf2[i], len);
			return 0;
		}
	}

	return 1;
}

void rdm_rma_check_tcqe(struct fi_cq_tagged_entry *tcqe, void *ctx,
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

void rdm_rma_check_cntrs(uint64_t w, uint64_t r, uint64_t w_e, uint64_t r_e)
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

void xfer_for_each_size(void (*xfer)(int len), int slen, int elen)
{
	int i;

	for (i = slen; i <= elen; i *= 2) {
		xfer(i);
	}
}

void err_inject_enable(void)
{
	int ret, err_count_val = 1;

	ret = gni_domain_ops->set_val(&dom->fid, GNI_ERR_INJECT_COUNT,
				      &err_count_val);
	cr_assert(!ret, "setval(GNI_ERR_INJECT_COUNT)");
}

/*******************************************************************************
 * Test RMA functions
 ******************************************************************************/

TestSuite(rdm_rma, .init = rdm_rma_setup, .fini = rdm_rma_teardown,
	  .disabled = false);

void do_write(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;

	init_data(source, len, 0xab);
	init_data(target, len, 0);
	sz = fi_write(ep[0], source, len,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);
	rdm_rma_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");
}

Test(rdm_rma, write)
{
	xfer_for_each_size(do_write, 8, BUF_SZ);
}

Test(rdm_rma, write_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_write, 8, BUF_SZ);
}

void do_writev(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	struct iovec iov;

	iov.iov_base = source;
	iov.iov_len = len;

	init_data(source, len, 0x25);
	init_data(target, len, 0);
	sz = fi_writev(ep[0], &iov, (void **)&loc_mr, 1,
		       gni_addr[1], (uint64_t)target, mr_key,
		       target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);
	rdm_rma_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");
}

Test(rdm_rma, writev)
{
	xfer_for_each_size(do_writev, 8, BUF_SZ);
}

Test(rdm_rma, writev_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_writev, 8, BUF_SZ);
}

void do_writemsg(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	struct iovec iov;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	iov.iov_base = source;
	iov.iov_len = len;

	rma_iov.addr = (uint64_t)target;
	rma_iov.len = len;
	rma_iov.key = mr_key;

	msg.msg_iov = &iov;
	msg.desc = (void **)&loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = target;
	msg.data = (uint64_t)target;

	init_data(source, len, 0xef);
	init_data(target, len, 0);
	sz = fi_writemsg(ep[0], &msg, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);
	rdm_rma_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");
}

Test(rdm_rma, writemsg)
{
	xfer_for_each_size(do_writemsg, 8, BUF_SZ);
}

Test(rdm_rma, writemsg_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_writemsg, 8, BUF_SZ);
}

/*
 * write_fence should be validated by inspecting debug.
 *
 * The following sequence of events should be seen:
 *
 * TX request processed: A
 * TX request queue stalled on FI_FENCE request: B
 * Added event: A
 * TX request processed: B
 *
 */

void do_write_fence(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	struct iovec iov;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	iov.iov_base = source;
	iov.iov_len = len;

	rma_iov.addr = (uint64_t)target;
	rma_iov.len = sizeof(target);
	rma_iov.key = mr_key;

	msg.msg_iov = &iov;
	msg.desc = (void **)&loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = target;
	msg.data = (uint64_t)target;

	init_data(source, len, 0xef);
	init_data(target, len, 0);

	/* write A */
	sz = fi_writemsg(ep[0], &msg, 0);
	cr_assert_eq(sz, 0);

	/* write B */
	sz = fi_writemsg(ep[0], &msg, FI_FENCE);
	cr_assert_eq(sz, 0);

	/* event A */
	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);

	/* event B */
	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);
	rdm_rma_check_cntrs(2, 0, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");
}

Test(rdm_rma, write_fence)
{
	xfer_for_each_size(do_write_fence, 8, BUF_SZ);
}

Test(rdm_rma, write_fence_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_write_fence, 8, BUF_SZ);
}

#define INJECT_SIZE 64
void do_inject_write(int len)
{
	ssize_t sz;
	int ret, i, loops = 0;
	struct fi_cq_tagged_entry cqe;

	init_data(source, len, 0x23);
	init_data(target, len, 0);
	sz = fi_inject_write(ep[0], source, len,
			     gni_addr[1], (uint64_t)target, mr_key);
	cr_assert_eq(sz, 0);

	for (i = 0; i < len; i++) {
		loops = 0;
		while (source[i] != target[i]) {
			ret = fi_cq_read(send_cq, &cqe, 1); /* for progress */
			cr_assert(ret == -EAGAIN,
				  "Received unexpected event\n");

			pthread_yield();
			cr_assert(++loops < 10000, "Data mismatch");
		}
	}
}

Test(rdm_rma, inject_write)
{
	xfer_for_each_size(do_inject_write, 8, INJECT_SIZE);
}

Test(rdm_rma, inject_write_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_inject_write, 8, INJECT_SIZE);
}

void do_writedata(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe, dcqe;

#define WRITE_DATA 0x5123da1a145
	init_data(source, len, 0x23);
	init_data(target, len, 0);
	sz = fi_writedata(ep[0], source, len, loc_mr, WRITE_DATA,
			 gni_addr[1], (uint64_t)target, mr_key,
			 target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);
	rdm_rma_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");

	while ((ret = fi_cq_read(recv_cq, &dcqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}
	cr_assert(ret != FI_SUCCESS, "Missing remote data");

	rdm_rma_check_tcqe(&dcqe, NULL,
			   (FI_RMA | FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA),
			   WRITE_DATA);
}

Test(rdm_rma, writedata)
{
	xfer_for_each_size(do_writedata, 8, BUF_SZ);
}

Test(rdm_rma, writedata_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_writedata, 8, BUF_SZ);
}

#define INJECTWRITE_DATA 0xdededadadeadbeaf
void do_inject_writedata(int len)
{
	ssize_t sz;
	int ret, i, loops = 0;
	struct fi_cq_tagged_entry cqe, dcqe;

	init_data(source, len, 0x23);
	init_data(target, len, 0);
	sz = fi_inject_writedata(ep[0], source, len, INJECTWRITE_DATA,
				 gni_addr[1], (uint64_t)target, mr_key);
	cr_assert_eq(sz, 0);

	for (i = 0; i < len; i++) {
		loops = 0;
		while (source[i] != target[i]) {
			ret = fi_cq_read(send_cq, &cqe, 1); /* for progress */
			cr_assert(ret == -EAGAIN,
				  "Received unexpected event\n");

			pthread_yield();
			cr_assert(++loops < 10000, "Data mismatch");
		}
	}

	while ((ret = fi_cq_read(recv_cq, &dcqe, 1)) == -FI_EAGAIN) {
		ret = fi_cq_read(send_cq, &cqe, 1); /* for progress */
		pthread_yield();
	}
	cr_assert(ret != FI_SUCCESS, "Missing remote data");

	rdm_rma_check_tcqe(&dcqe, NULL,
			   (FI_RMA | FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA),
			   INJECTWRITE_DATA);
}

Test(rdm_rma, inject_writedata)
{
	xfer_for_each_size(do_inject_writedata, 8, INJECT_SIZE);
}

Test(rdm_rma, inject_writedata_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_inject_writedata, 8, INJECT_SIZE);
}

void do_read(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;

#define READ_CTX 0x4e3dda1aULL
	init_data(source, len, 0);
	init_data(target, len, 0xad);
	sz = fi_read(ep[0], source, len,
			loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			(void *)READ_CTX);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, (void *)READ_CTX, FI_RMA | FI_READ, 0);
	rdm_rma_check_cntrs(0, 1, 0, 0);

	dbg_printf("got read context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");
}

Test(rdm_rma, read)
{
	xfer_for_each_size(do_read, 8, BUF_SZ);
}

Test(rdm_rma, read_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_read, 8, BUF_SZ);
}

void do_readv(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	struct iovec iov;

	iov.iov_base = source;
	iov.iov_len = len;

	init_data(target, len, 0x25);
	init_data(source, len, 0);
	sz = fi_readv(ep[0], &iov, (void **)&loc_mr, 1,
		       gni_addr[1], (uint64_t)target, mr_key,
		       target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_READ, 0);
	rdm_rma_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");
}

Test(rdm_rma, readv)
{
	xfer_for_each_size(do_readv, 8, BUF_SZ);
}

Test(rdm_rma, readv_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_readv, 8, BUF_SZ);
}

void do_readmsg(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	struct iovec iov;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	iov.iov_base = source;
	iov.iov_len = len;

	rma_iov.addr = (uint64_t)target;
	rma_iov.len = len;
	rma_iov.key = mr_key;

	msg.msg_iov = &iov;
	msg.desc = (void **)&loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = target;
	msg.data = (uint64_t)target;

	init_data(target, len, 0xef);
	init_data(source, len, 0);
	sz = fi_readmsg(ep[0], &msg, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_READ, 0);
	rdm_rma_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");
}

Test(rdm_rma, readmsg)
{
	xfer_for_each_size(do_readmsg, 8, BUF_SZ);
}

Test(rdm_rma, readmsg_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_readmsg, 8, BUF_SZ);
}

#define READ_DATA 0xdededadadeaddeef
void do_readmsgdata(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe, dcqe;
	struct iovec iov;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	iov.iov_base = source;
	iov.iov_len = len;

	rma_iov.addr = (uint64_t)target;
	rma_iov.len = len;
	rma_iov.key = mr_key;

	msg.msg_iov = &iov;
	msg.desc = (void **)&loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = target;
	msg.data = (uint64_t)READ_DATA;

	init_data(target, len, 0xef);
	init_data(source, len, 0);
	sz = fi_readmsg(ep[0], &msg, FI_REMOTE_CQ_DATA);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_READ, 0);
	rdm_rma_check_cntrs(0, 1, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");

	while ((ret = fi_cq_read(recv_cq, &dcqe, 1)) == -FI_EAGAIN) {
		ret = fi_cq_read(send_cq, &cqe, 1); /* for progress */
		pthread_yield();
	}
	cr_assert(ret != FI_SUCCESS, "Missing remote data");

	rdm_rma_check_tcqe(&dcqe, NULL,
			   (FI_RMA | FI_REMOTE_READ | FI_REMOTE_CQ_DATA),
			   READ_DATA);
}

Test(rdm_rma, readmsgdata)
{
	xfer_for_each_size(do_readmsgdata, 8, BUF_SZ);
}

Test(rdm_rma, readmsgdata_retrans)
{
	err_inject_enable();
	xfer_for_each_size(do_readmsgdata, 8, BUF_SZ);
}

Test(rdm_rma, inject)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	struct iovec iov;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	iov.iov_base = source;
	iov.iov_len = GNIX_INJECT_SIZE;

	rma_iov.addr = (uint64_t)target;
	rma_iov.len = GNIX_INJECT_SIZE;
	rma_iov.key = mr_key;

	msg.msg_iov = &iov;
	msg.desc = (void **)&loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = target;
	msg.data = (uint64_t)target;

	init_data(source, GNIX_INJECT_SIZE, 0xef);
	init_data(target, GNIX_INJECT_SIZE, 0);

	sz = fi_writemsg(ep[0], &msg, FI_INJECT);
	cr_assert_eq(sz, 0);

	iov.iov_len = GNIX_INJECT_SIZE+1;
	sz = fi_writemsg(ep[0], &msg, FI_INJECT);
	cr_assert_eq(sz, -FI_EINVAL);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);
	rdm_rma_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, GNIX_INJECT_SIZE),
		  "Data mismatch");
}

void do_write_autoreg(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;

	init_data(source, len, 0xab);
	init_data(target, len, 0);
	sz = fi_write(ep[0], source, len,
			 NULL, gni_addr[1], (uint64_t)target, mr_key,
			 target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);
	rdm_rma_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(source, target, len), "Data mismatch");
}

Test(rdm_rma, write_autoreg)
{
	xfer_for_each_size(do_write_autoreg, 8, BUF_SZ);
}

void do_write_autoreg_uncached(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;

	init_data(uc_source, len, 0xab);
	init_data(target, len, 0);
	sz = fi_write(ep[0], uc_source, len,
			 NULL, gni_addr[1], (uint64_t)target, mr_key,
			 target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	rdm_rma_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);
	rdm_rma_check_cntrs(1, 0, 0, 0);

	dbg_printf("got write context event!\n");

	cr_assert(check_data(uc_source, target, len), "Data mismatch");
}

Test(rdm_rma, write_autoreg_uncached)
{
	xfer_for_each_size(do_write_autoreg_uncached, 8, BUF_SZ);
}

void do_write_error(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	struct fi_cq_err_entry err_cqe;

	init_data(source, len, 0xab);
	init_data(target, len, 0);
	sz = fi_write(ep[0], source, len,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, -FI_EAVAIL);

	ret = fi_cq_readerr(send_cq, &err_cqe, 0);
	cr_assert_eq(ret, 1);

	cr_assert((uint64_t)err_cqe.op_context == (uint64_t)target,
		  "Bad error context");
	cr_assert(err_cqe.flags == (FI_RMA | FI_WRITE));
	cr_assert(err_cqe.len == 0, "Bad error len");
	cr_assert(err_cqe.buf == 0, "Bad error buf");
	cr_assert(err_cqe.data == 0, "Bad error data");
	cr_assert(err_cqe.tag == 0, "Bad error tag");
	cr_assert(err_cqe.olen == 0, "Bad error olen");
	cr_assert(err_cqe.err == FI_ECANCELED, "Bad error errno");
	cr_assert(err_cqe.prov_errno == GNI_RC_TRANSACTION_ERROR,
		  "Bad prov errno");
	cr_assert(err_cqe.err_data == NULL, "Bad error provider data");

	rdm_rma_check_cntrs(0, 0, 1, 0);
}

Test(rdm_rma, write_error)
{
	int ret, max_retrans_val = 1;

	ret = gni_domain_ops->set_val(&dom->fid, GNI_MAX_RETRANSMITS,
				      &max_retrans_val);
	cr_assert(!ret, "setval(GNI_MAX_RETRANSMITS)");
	err_inject_enable();

	xfer_for_each_size(do_write_error, 8, BUF_SZ);
}

void do_read_error(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	struct fi_cq_err_entry err_cqe;

	init_data(source, len, 0);
	init_data(target, len, 0xad);
	sz = fi_read(ep[0], source, len,
			loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			(void *)READ_CTX);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, -FI_EAVAIL);

	ret = fi_cq_readerr(send_cq, &err_cqe, 0);
	cr_assert_eq(ret, 1);

	cr_assert((uint64_t)err_cqe.op_context == (uint64_t)READ_CTX,
		  "Bad error context");
	cr_assert(err_cqe.flags == (FI_RMA | FI_READ));
	cr_assert(err_cqe.len == 0, "Bad error len");
	cr_assert(err_cqe.buf == 0, "Bad error buf");
	cr_assert(err_cqe.data == 0, "Bad error data");
	cr_assert(err_cqe.tag == 0, "Bad error tag");
	cr_assert(err_cqe.olen == 0, "Bad error olen");
	cr_assert(err_cqe.err == FI_ECANCELED, "Bad error errno");
	cr_assert(err_cqe.prov_errno == GNI_RC_TRANSACTION_ERROR,
		  "Bad prov errno");
	cr_assert(err_cqe.err_data == NULL, "Bad error provider data");

	rdm_rma_check_cntrs(0, 0, 0, 1);
}

Test(rdm_rma, read_error)
{
	int ret, max_retrans_val = 1;

	ret = gni_domain_ops->set_val(&dom->fid, GNI_MAX_RETRANSMITS,
				      &max_retrans_val);
	cr_assert(!ret, "setval(GNI_MAX_RETRANSMITS)");
	err_inject_enable();

	xfer_for_each_size(do_read_error, 8, BUF_SZ);
}

