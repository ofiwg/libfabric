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
static struct fid_ep *ep[2];
static struct fid_av *av;
static struct fi_info *hints;
static struct fi_info *fi;
void *ep_name[2];
size_t gni_addr[2];
static struct fid_cq *send_cq;
static struct fid_cq *recv_cq;
static struct fi_cq_attr cq_attr;
static struct fid_cntr *write_cntr, *read_cntr, *rcv_cntr;
static struct fi_cntr_attr cntr_attr = {.events = FI_CNTR_EVENTS_COMP,
					.flags = 0};

#define BUF_SZ (64*1024)
char *target;
char *source;
struct fid_mr *rem_mr, *loc_mr;
uint64_t mr_key;

void cntr_setup(void)
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

	attr.type = FI_AV_MAP;
	attr.count = 16;

	ret = fi_av_open(dom, &attr, &av, NULL);
	cr_assert(!ret, "fi_av_open");

	ret = fi_endpoint(dom, fi, &ep[0], NULL);
	cr_assert(!ret, "fi_endpoint");

	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.size = 1024;
	cq_attr.wait_obj = 0;

	ret = fi_cq_open(dom, &cq_attr, &send_cq, 0);
	cr_assert(!ret, "fi_cq_open");

	ret = fi_cq_open(dom, &cq_attr, &recv_cq, 0);
	cr_assert(!ret, "fi_cq_open");

	ret = fi_cntr_open(dom, &cntr_attr, &write_cntr, 0);
	cr_assert(!ret, "fi_cntr_open");

	ret = fi_cntr_open(dom, &cntr_attr, &read_cntr, 0);
	cr_assert(!ret, "fi_cntr_open");

	ret = fi_cntr_open(dom, &cntr_attr, &rcv_cntr, 0);
	cr_assert(!ret, "fi_cntr_open");

	ret = fi_ep_bind(ep[0], &send_cq->fid, FI_SEND);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[0], &recv_cq->fid, FI_RECV);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[0], &write_cntr->fid, FI_WRITE | FI_SEND);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[0], &read_cntr->fid, FI_READ);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[0], &rcv_cntr->fid, FI_RECV);
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

	ret = fi_ep_bind(ep[1], &send_cq->fid, FI_SEND);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[1], &recv_cq->fid, FI_RECV);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[1], &write_cntr->fid, FI_WRITE | FI_SEND);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[1], &read_cntr->fid, FI_READ);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_ep_bind(ep[1], &rcv_cntr->fid, FI_RECV);
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

	mr_key = fi_mr_key(rem_mr);
}

void cntr_teardown(void)
{
	int ret = 0;

	fi_close(&loc_mr->fid);
	fi_close(&rem_mr->fid);

	free(target);
	free(source);

	ret = fi_close(&ep[0]->fid);
	cr_assert(!ret, "failure in closing ep.");

	ret = fi_close(&ep[1]->fid);
	cr_assert(!ret, "failure in closing ep.");

	ret = fi_close(&send_cq->fid);
	cr_assert(!ret, "failure in send cq.");

	ret = fi_close(&recv_cq->fid);
	cr_assert(!ret, "failure in send cq.");

	ret = fi_close(&write_cntr->fid);
	cr_assert(!ret, "failure in write_cntr.");

	ret = fi_close(&read_cntr->fid);
	cr_assert(!ret, "failure in read_cntr.");

	ret = fi_close(&rcv_cntr->fid);
	cr_assert(!ret, "failure in read_cntr.");

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

static void init_data(char *buf, int len, char seed)
{
	int i;

	for (i = 0; i < len; i++)
		buf[i] = seed++;
}

static int check_data(char *buf1, char *buf2, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (buf1[i] != buf2[i]) {
			printf("data mismatch, elem: %d, exp: %x, act: %x\n",
			       i, buf1[i], buf2[i]);
			return 0;
		}
	}

	return 1;
}

static void xfer_for_each_size(void (*xfer)(int len), int slen, int elen)
{
	int i;

	for (i = slen; i <= elen; i *= 2)
		xfer(i);
}

/*******************************************************************************
 * Test RMA functions
 ******************************************************************************/

TestSuite(cntr, .init = cntr_setup, .fini = cntr_teardown,
	  .disabled = false);

static void do_write(int len)
{
	uint64_t old_w_cnt, new_w_cnt;
	uint64_t old_r_cnt, new_r_cnt;
	ssize_t sz;

	init_data(source, len, 0xab);
	init_data(target, len, 0);


	old_w_cnt = fi_cntr_read(write_cntr);
	cr_assert(old_w_cnt >= 0);

	old_r_cnt = fi_cntr_read(read_cntr);
	cr_assert(old_r_cnt >= 0);

	sz = fi_write(ep[0], source, len,
			 loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			 target);
	cr_assert_eq(sz, 0);

	do {
		new_w_cnt = fi_cntr_read(write_cntr);
		cr_assert(new_w_cnt >= 0);
		if (new_w_cnt == (old_w_cnt + 1))
			break;
		pthread_yield();
	} while (1);

	cr_assert(check_data(source, target, len), "Data mismatch");

	new_r_cnt = fi_cntr_read(read_cntr);
	cr_assert(new_r_cnt >= 0);

	/*
	 * no fi_read called so old and new read cnts should be equal
	 */
	cr_assert(new_r_cnt == old_r_cnt);
}

Test(cntr, write)
{
	xfer_for_each_size(do_write, 8, BUF_SZ);
}

static void do_write_wait(int len)
{
	uint64_t old_w_cnt, new_w_cnt;
	uint64_t old_r_cnt, new_r_cnt;
	ssize_t sz;
	const int iters = 1;
	int i;

	init_data(source, len, 0xab);
	init_data(target, len, 0);

	old_w_cnt = fi_cntr_read(write_cntr);
	cr_assert(old_w_cnt >= 0);

	old_r_cnt = fi_cntr_read(read_cntr);
	cr_assert(old_r_cnt >= 0);

	for (i = 0; i < iters; i++) {
		sz = fi_write(ep[0], source, len,
			      loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			      target);
		cr_assert_eq(sz, 0);
	}

	fi_cntr_wait(write_cntr, old_w_cnt+iters, -1);
	new_w_cnt = fi_cntr_read(write_cntr);
	cr_assert(old_w_cnt + iters == new_w_cnt);

	cr_assert(check_data(source, target, len), "Data mismatch");

	new_r_cnt = fi_cntr_read(read_cntr);
	cr_assert(new_r_cnt >= 0);

	/*
	 * no fi_read called so old and new read cnts should be equal
	 */
	cr_assert(new_r_cnt == old_r_cnt);
}

Test(cntr, write_wait)
{
	xfer_for_each_size(do_write_wait, 8, BUF_SZ);
}

static void do_read(int len)
{
	ssize_t sz;
	uint64_t old_w_cnt, new_w_cnt;
	uint64_t old_r_cnt, new_r_cnt;

#define READ_CTX 0x4e3dda1aULL
	init_data(source, len, 0);
	init_data(target, len, 0xad);

	old_w_cnt = fi_cntr_read(write_cntr);
	cr_assert(old_w_cnt >= 0);

	old_r_cnt = fi_cntr_read(read_cntr);
	cr_assert(old_r_cnt >= 0);

	sz = fi_read(ep[0], source, len,
			loc_mr, gni_addr[1], (uint64_t)target, mr_key,
			(void *)READ_CTX);
	cr_assert_eq(sz, 0);

	do {
		new_r_cnt = fi_cntr_read(read_cntr);
		cr_assert(new_r_cnt >= 0);
		if (new_r_cnt == (old_r_cnt + 1))
			break;
		pthread_yield();
	} while (1);

	cr_assert(check_data(source, target, len), "Data mismatch");

	new_w_cnt = fi_cntr_read(write_cntr);
	cr_assert(new_w_cnt >= 0);

	/*
	 * no fi_read called so old and new read cnts should be equal
	 */
	cr_assert(new_w_cnt == old_w_cnt);
}

static void do_read_wait(int len)
{
	int i, iters = 10;
	ssize_t sz;
	uint64_t old_w_cnt, new_w_cnt;
	uint64_t old_r_cnt;

#define READ_CTX 0x4e3dda1aULL
	init_data(source, len, 0);
	init_data(target, len, 0xad);

	old_w_cnt = fi_cntr_read(write_cntr);
	cr_assert(old_w_cnt >= 0);

	old_r_cnt = fi_cntr_read(read_cntr);
	cr_assert(old_r_cnt >= 0);

	for (i = 0; i < iters; i++) {
		sz = fi_read(ep[0], source, len,
				loc_mr, gni_addr[1], (uint64_t)target,
				mr_key, (void *)READ_CTX);
		cr_assert_eq(sz, 0);
	}

	fi_cntr_wait(read_cntr, old_r_cnt + iters, -1);

	cr_assert(check_data(source, target, len), "Data mismatch");

	new_w_cnt = fi_cntr_read(write_cntr);
	cr_assert(new_w_cnt >= 0);

	/*
	 * no fi_read called so old and new read cnts should be equal
	 */
	cr_assert(new_w_cnt == old_w_cnt);
}

Test(cntr, read)
{
	xfer_for_each_size(do_read, 8, BUF_SZ);
}

Test(cntr, read_wait)
{
	xfer_for_each_size(do_read_wait, 8, BUF_SZ);
}


Test(cntr, send_recv)
{
	int ret, i, got_r = 0;
	struct fi_context r_context, s_context;
	struct fi_cq_entry cqe;
	uint64_t old_s_cnt, new_s_cnt;
	uint64_t old_r_cnt, new_r_cnt;
	char s_buffer[128], r_buffer[128];

	old_s_cnt = fi_cntr_read(write_cntr);
	cr_assert(old_s_cnt >= 0);

	old_r_cnt = fi_cntr_read(rcv_cntr);
	cr_assert(old_r_cnt >= 0);

	for (i = 0; i < 16; i++) {
		sprintf(s_buffer, "Hello there iter=%d", i);
		memset(r_buffer, 0, 128);
		ret = fi_recv(ep[1],
			      r_buffer,
			      sizeof(r_buffer),
			      NULL,
			      gni_addr[0],
			      &r_context);
		cr_assert_eq(ret, FI_SUCCESS, "fi_recv");
		ret = fi_send(ep[0],
			      s_buffer,
			      strlen(s_buffer),
			      NULL,
			      gni_addr[1],
			      &s_context);
		cr_assert_eq(ret, FI_SUCCESS, "fi_send");

		while ((ret = fi_cq_read(send_cq, &cqe, 1)) == -FI_EAGAIN)
			pthread_yield();

		cr_assert((cqe.op_context == &r_context) ||
			(cqe.op_context == &s_context), "fi_cq_read");
		got_r = (cqe.op_context == &r_context) ? 1 : 0;

		if (got_r) {
			new_r_cnt = fi_cntr_read(rcv_cntr);
			old_r_cnt++;
			cr_assert(new_r_cnt == old_r_cnt);
		} else {
			new_s_cnt = fi_cntr_read(write_cntr);
			old_s_cnt++;
			cr_assert(new_s_cnt == old_s_cnt);
		}

		while ((ret = fi_cq_read(recv_cq, &cqe, 1)) == -FI_EAGAIN)
			pthread_yield();
		if (got_r)
			cr_assert((cqe.op_context == &s_context), "fi_cq_read");
		else
			cr_assert((cqe.op_context == &r_context), "fi_cq_read");

		if (got_r) {
			new_s_cnt = fi_cntr_read(write_cntr);
			old_s_cnt++;
			cr_assert(new_s_cnt == old_s_cnt);
		} else {
			new_r_cnt = fi_cntr_read(rcv_cntr);
			old_r_cnt++;
			cr_assert(new_r_cnt == old_r_cnt);
		}

		cr_assert(strcmp(s_buffer, r_buffer) == 0, "check message");

		got_r = 0;
	}

}
