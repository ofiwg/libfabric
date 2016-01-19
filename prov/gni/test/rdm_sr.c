/*
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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>
#include <poll.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>

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
#define dbg_printf(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)
#endif

#define NUMEPS 2

static struct fid_fabric *fab;
static struct fid_domain *dom[NUMEPS];
struct fi_gni_ops_domain *gni_domain_ops[NUMEPS];
static struct fid_ep *ep[NUMEPS];
static struct fid_av *av[NUMEPS];
void *ep_name[NUMEPS];
fi_addr_t gni_addr[NUMEPS];
static struct fid_cq *msg_cq[NUMEPS];
static struct fi_info *fi[NUMEPS];
static struct fi_cq_attr cq_attr;
const char *cdm_id[NUMEPS] = { "5000", "5001" };
struct fi_info *hints;
static int using_bnd_ep = 0;

#define BUF_SZ (1<<20)
char *target;
char *source;
char *uc_target;
char *uc_source;
struct fid_mr *rem_mr[NUMEPS], *loc_mr[NUMEPS];
uint64_t mr_key[NUMEPS];

static struct fid_cntr *send_cntr[NUMEPS], *recv_cntr[NUMEPS];
static struct fi_cntr_attr cntr_attr = {
	.events = FI_CNTR_EVENTS_COMP,
	.flags = 0
};
static uint64_t sends[NUMEPS] = {0}, recvs[NUMEPS] = {0},
	send_errs[NUMEPS] = {0}, recv_errs[NUMEPS] = {0};

void rdm_sr_setup_common_eps(void)
{
	int ret = 0, i = 0, j = 0;
	struct fi_av_attr attr;
	size_t addrlen = 0;

	attr.type = FI_AV_MAP;
	attr.count = NUMEPS;

	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cq_attr.size = 1024;
	cq_attr.wait_obj = 0;

	target = malloc(BUF_SZ * 3); /* 3x BUF_SZ for multi recv testing */
	assert(target);

	source = malloc(BUF_SZ);
	assert(source);

	uc_target = malloc(BUF_SZ);
	assert(uc_target);

	uc_source = malloc(BUF_SZ);
	assert(uc_source);

	ret = fi_fabric(fi[0]->fabric_attr, &fab, NULL);
	cr_assert(!ret, "fi_fabric");

	for (; i < NUMEPS; i++) {
		ret = fi_domain(fab, fi[i], dom + i, NULL);
		cr_assert(!ret, "fi_domain");

		ret = fi_open_ops(&dom[i]->fid, FI_GNI_DOMAIN_OPS_1,
				  0, (void **) (gni_domain_ops + i), NULL);

		ret = fi_av_open(dom[i], &attr, av + i, NULL);
		cr_assert(!ret, "fi_av_open");

		ret = fi_endpoint(dom[i], fi[i], ep + i, NULL);
		cr_assert(!ret, "fi_endpoint");

		ret = fi_cq_open(dom[i], &cq_attr, msg_cq + i, 0);
		cr_assert(!ret, "fi_cq_open");

		ret = fi_ep_bind(ep[i], &msg_cq[i]->fid, FI_SEND | FI_RECV);
		cr_assert(!ret, "fi_ep_bind");

		ret = fi_getname(&ep[i]->fid, NULL, &addrlen);
		cr_assert(addrlen > 0);

		ep_name[i] = malloc(addrlen);
		cr_assert(ep_name[i] != NULL);

		ret = fi_getname(&ep[i]->fid, ep_name[i], &addrlen);
		cr_assert(ret == FI_SUCCESS);
	}

	for (i = 0; i < NUMEPS; i++) {
		/* Insert all gni addresses into each av */
		for (j = 0; j < NUMEPS; j++) {
			ret = fi_av_insert(av[i], ep_name[j], 1, &gni_addr[j],
					   0, NULL);
			cr_assert(ret == 1);
		}

		ret = fi_ep_bind(ep[i], &av[i]->fid, 0);
		cr_assert(!ret, "fi_ep_bind");

		ret = fi_enable(ep[i]);
		cr_assert(!ret, "fi_ep_enable");

		ret = fi_cntr_open(dom[i], &cntr_attr, send_cntr + i, 0);
		cr_assert(!ret, "fi_cntr_open");

		ret = fi_ep_bind(ep[i], &send_cntr[i]->fid, FI_SEND);
		cr_assert(!ret, "fi_ep_bind");

		ret = fi_cntr_open(dom[i], &cntr_attr, recv_cntr + i, 0);
		cr_assert(!ret, "fi_cntr_open");

		ret = fi_ep_bind(ep[i], &recv_cntr[i]->fid, FI_RECV);
		cr_assert(!ret, "fi_ep_bind");
	}
}

void rdm_sr_setup_common(void)
{
	int ret = 0, i = 0;

	rdm_sr_setup_common_eps();

	for (i = 0; i < NUMEPS; i++) {
		ret = fi_mr_reg(dom[i], target, 3 * BUF_SZ,
				FI_REMOTE_WRITE, 0, 0, 0, rem_mr + i, &target);
		cr_assert_eq(ret, 0);

		ret = fi_mr_reg(dom[i], source, BUF_SZ,
				FI_REMOTE_WRITE, 0, 0, 0, loc_mr + i, &source);
		cr_assert_eq(ret, 0);

		mr_key[i] = fi_mr_key(rem_mr[i]);
	}
}

void rdm_sr_setup(bool is_noreg)
{
	int ret = 0, i = 0;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->domain_attr->cq_data_size = NUMEPS * 2;
	hints->domain_attr->data_progress = FI_PROGRESS_AUTO;
	hints->mode = ~0;
	hints->caps = is_noreg ? hints->caps : FI_SOURCE;
	hints->fabric_attr->name = strdup("gni");

	/* Get info about fabric services with the provided hints */
	for (; i < NUMEPS; i++) {
		ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi[i]);
		cr_assert(!ret, "fi_getinfo");
	}

	if (is_noreg)
		rdm_sr_setup_common_eps();
	else
		rdm_sr_setup_common();
}

static void rdm_sr_setup_reg(void) {
	rdm_sr_setup(false);
}

static void rdm_sr_setup_noreg(void) {
	rdm_sr_setup(true);
}

void rdm_sr_bnd_ep_setup(void)
{
	int ret = 0, i = 0;
	char my_hostname[HOST_NAME_MAX];

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->domain_attr->cq_data_size = NUMEPS * 2;
	hints->mode = ~0;
	hints->fabric_attr->name = strdup("gni");

	ret = gethostname(my_hostname, sizeof(my_hostname));
	cr_assert(!ret, "gethostname");

	for (; i < NUMEPS; i++) {
		ret = fi_getinfo(FI_VERSION(1, 0), my_hostname,
				 cdm_id[i], 0, hints, fi + i);
		cr_assert(!ret, "fi_getinfo");
	}

	using_bnd_ep = 1;

	rdm_sr_setup_common();
}

static void rdm_sr_teardown_common(bool unreg)
{
	int ret = 0, i = 0;

	free(uc_source);
	free(uc_target);
	free(target);
	free(source);

	for (; i < NUMEPS; i++) {
		fi_close(&recv_cntr[i]->fid);
		fi_close(&send_cntr[i]->fid);

		if (unreg) {
			fi_close(&loc_mr[i]->fid);
			fi_close(&rem_mr[i]->fid);
		}

		ret = fi_close(&ep[i]->fid);
		cr_assert(!ret, "failure in closing ep.");

		ret = fi_close(&msg_cq[i]->fid);
		cr_assert(!ret, "failure in send cq.");

		ret = fi_close(&av[i]->fid);
		cr_assert(!ret, "failure in closing av.");

		ret = fi_close(&dom[i]->fid);
		cr_assert(!ret, "failure in closing domain.");

		fi_freeinfo(fi[i]);

		free(ep_name[i]);
	}

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "failure in closing fabric.");

	fi_freeinfo(hints);
}

static void rdm_sr_teardown(void)
{
	rdm_sr_teardown_common(true);
}

static void rdm_sr_teardown_nounreg(void)
{
	rdm_sr_teardown_common(false);
}


void rdm_sr_init_data(char *buf, int len, char seed)
{
	int i;

	for (i = 0; i < len; i++) {
		buf[i] = seed++;
	}
}

int rdm_sr_check_data(char *buf1, char *buf2, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (buf1[i] != buf2[i]) {
			printf("data mismatch, elem: %d, exp: %hhx, act: %hhx\n"
			       , i, buf1[i], buf2[i]);
			return 0;
		}
	}

	return 1;
}

void rdm_sr_xfer_for_each_size(void (*xfer)(int len), int slen, int elen)
{
	int i;

	for (i = slen; i <= elen; i *= 2) {
		xfer(i);
	}
}

void rdm_sr_check_cqe(struct fi_cq_tagged_entry *cqe, void *ctx,
		      uint64_t flags, void *addr, size_t len,
		      uint64_t data)
{
	cr_assert(cqe->op_context == ctx, "CQE Context mismatch");
	cr_assert(cqe->flags == flags, "CQE flags mismatch");

	if (flags & FI_RECV) {
		cr_assert(cqe->len == len, "CQE length mismatch");
		cr_assert(cqe->buf == addr, "CQE address mismatch");

		if (flags & FI_REMOTE_CQ_DATA)
			cr_assert(cqe->data == data, "CQE data mismatch");
	} else {
		cr_assert(cqe->len == 0, "Invalid CQE length");
		cr_assert(cqe->buf == 0, "Invalid CQE address");
		cr_assert(cqe->data == 0, "Invalid CQE data");
	}

	cr_assert(cqe->tag == 0, "Invalid CQE tag");
}

void rdm_sr_check_cntrs(uint64_t s[], uint64_t r[], uint64_t s_e[],
			uint64_t r_e[])
{
	int i = 0;
	for (; i < NUMEPS; i++) {
		sends[i] += s[i];
		recvs[i] += r[i];
		send_errs[i] += s_e[i];
		recv_errs[i] += r_e[i];

		cr_assert(fi_cntr_read(send_cntr[i]) == sends[i],
			  "Bad send count");
		cr_assert(fi_cntr_read(recv_cntr[i]) == recvs[i],
			  "Bad recv count");
		cr_assert(fi_cntr_readerr(send_cntr[i]) == send_errs[i],
			  "Bad send err count");
		cr_assert(fi_cntr_readerr(recv_cntr[i]) == recv_errs[i],
			  "Bad recv err count");
	}
}

void rdm_sr_err_inject_enable(void)
{
	int ret, err_count_val = 1, i = 0;

	for (; i < NUMEPS; i++) {
		ret = gni_domain_ops[i]->set_val(&dom[i]->fid,
						 GNI_ERR_INJECT_COUNT,
						 &err_count_val);
		cr_assert(!ret, "setval(GNI_ERR_INJECT_COUNT)");
	}
}

void rdm_sr_lazy_dereg_disable(void)
{
	int ret, lazy_dereg_val = 0, i = 0;

	for (; i < NUMEPS; i++) {
		ret = gni_domain_ops[i]->set_val(&dom[i]->fid,
						 GNI_MR_CACHE_LAZY_DEREG,
						 &lazy_dereg_val);
		cr_assert(!ret, "setval(GNI_MR_CACHE_LAZY_DEREG)");
	}
}

/*******************************************************************************
 * Test MSG functions
 ******************************************************************************/

TestSuite(rdm_sr, .init = rdm_sr_setup_reg, .fini = rdm_sr_teardown,
	  .disabled = false);

TestSuite(rdm_sr_noreg, .init = rdm_sr_setup_noreg,
	  .fini = rdm_sr_teardown_nounreg, .disabled = false);

TestSuite(rdm_sr_bnd_ep, .init = rdm_sr_bnd_ep_setup, .fini = rdm_sr_teardown,
	  .disabled = false);

/*
 * ssize_t fi_send(struct fid_ep *ep, void *buf, size_t len,
 *		void *desc, fi_addr_t dest_addr, void *context);
 *
 * ssize_t fi_recv(struct fid_ep *ep, void * buf, size_t len,
 *		void *desc, fi_addr_t src_addr, void *context);
 */
void do_send(int len)
{
	int ret;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	sz = fi_send(ep[0], source, len, loc_mr[0], gni_addr[1], target);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, rem_mr[1], gni_addr[0], source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV), target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, send)
{
	rdm_sr_xfer_for_each_size(do_send, 1, BUF_SZ);
}

Test(rdm_sr, send_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_send, 1, BUF_SZ);
}

/*
ssize_t fi_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		 size_t count, fi_addr_t dest_addr, void *context);
*/
void do_sendv(int len)
{
	int ret;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	struct iovec iov;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	rdm_sr_init_data(source, len, 0x25);
	rdm_sr_init_data(target, len, 0);

	sz = fi_sendv(ep[0], &iov, (void **)loc_mr, 1, gni_addr[1], target);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, rem_mr[0], gni_addr[0], source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV), target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, sendv)
{
	rdm_sr_xfer_for_each_size(do_sendv, 1, BUF_SZ);
}

Test(rdm_sr, sendv_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_sendv, 1, BUF_SZ);
}

/*
ssize_t fi_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
		   uint64_t flags);
*/
void do_sendmsg(int len)
{
	int ret;
	ssize_t sz;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	struct fi_msg msg;
	struct iovec iov;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	msg.context = target;
	msg.data = (uint64_t)target;

	rdm_sr_init_data(source, len, 0xef);
	rdm_sr_init_data(target, len, 0);

	sz = fi_sendmsg(ep[0], &msg, 0);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, rem_mr[0], gni_addr[0], source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV), target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, sendmsg)
{
	rdm_sr_xfer_for_each_size(do_sendmsg, 1, BUF_SZ);
}

Test(rdm_sr, sendmsg_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_sendmsg, 1, BUF_SZ);
}

/*
ssize_t fi_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
		   uint64_t flags);
*/

void do_sendmsgdata(int len)
{
	int ret;
	ssize_t sz;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	struct fi_msg msg;
	struct iovec iov;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	msg.context = target;
	msg.data = (uint64_t)source;

	rdm_sr_init_data(source, len, 0xef);
	rdm_sr_init_data(target, len, 0);

	sz = fi_sendmsg(ep[0], &msg, FI_REMOTE_CQ_DATA);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, rem_mr[0], gni_addr[0], source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV|FI_REMOTE_CQ_DATA),
			 target, len, (uint64_t)source);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, sendmsgdata)
{
	rdm_sr_xfer_for_each_size(do_sendmsgdata, 1, BUF_SZ);
}

Test(rdm_sr, sendmsgdata_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_sendmsgdata, 1, BUF_SZ);
}

/*
ssize_t fi_inject(struct fid_ep *ep, void *buf, size_t len,
		  fi_addr_t dest_addr);
*/
#define INJECT_SIZE 64
void do_inject(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0x23);
	rdm_sr_init_data(target, len, 0);

	sz = fi_inject(ep[0], source, len, gni_addr[1]);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, rem_mr[1], gni_addr[0], source);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(msg_cq[1], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
		/* Manually progress connection to domain 1 */
		fi_cq_read(msg_cq[0], &cqe, 1);
	}

	cr_assert_eq(ret, 1);
	rdm_sr_check_cqe(&cqe, source, (FI_MSG|FI_RECV),
			 target, len, (uint64_t)source);

	dbg_printf("got recv context event!\n");

	/* do progress until send counter is updated */
	while (fi_cntr_read(send_cntr[0]) < 1) {
		pthread_yield();
	}
	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	/* make sure inject does not generate a send competion */
	cr_assert_eq(fi_cq_read(msg_cq[0], &cqe, 1), -FI_EAGAIN);

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, inject)
{
	rdm_sr_xfer_for_each_size(do_inject, 1, INJECT_SIZE);
}

Test(rdm_sr, inject_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_inject, 1, INJECT_SIZE);
}

/*
ssize_t fi_senddata(struct fid_ep *ep, void *buf, size_t len, void *desc,
		    uint64_t data, fi_addr_t dest_addr, void *context);
*/
void do_senddata(int len)
{
	int ret;
	ssize_t sz;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	sz = fi_senddata(ep[0], source, len, loc_mr[0], (uint64_t)source,
			 gni_addr[1], target);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, rem_mr[0], gni_addr[0], source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV|FI_REMOTE_CQ_DATA),
			 target, len, (uint64_t)source);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, senddata)
{
	rdm_sr_xfer_for_each_size(do_senddata, 1, BUF_SZ);
}

Test(rdm_sr, senddata_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_senddata, 1, BUF_SZ);
}

/*
ssize_t fi_injectdata(struct fid_ep *ep, const void *buf, size_t len,
		      uint64_t data, fi_addr_t dest_addr);
*/
void do_injectdata(int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	sz = fi_injectdata(ep[0], source, len, (uint64_t)source, gni_addr[1]);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, rem_mr[0], gni_addr[0], source);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(msg_cq[1], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
		/* Manually progress connection to domain 1 */
		fi_cq_read(msg_cq[0], &cqe, 1);
	}

	rdm_sr_check_cqe(&cqe, source, (FI_MSG|FI_RECV|FI_REMOTE_CQ_DATA),
			 target, len, (uint64_t)source);

	dbg_printf("got recv context event!\n");

	/* don't progress until send counter is updated */
	while (fi_cntr_read(send_cntr[0]) < 1) {
		pthread_yield();
	}

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	/* make sure inject does not generate a send competion */
	cr_assert_eq(fi_cq_read(msg_cq[0], &cqe, 1), -FI_EAGAIN);

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, injectdata)
{
	rdm_sr_xfer_for_each_size(do_injectdata, 1, INJECT_SIZE);
}

Test(rdm_sr, injectdata_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_injectdata, 1, INJECT_SIZE);
}

/*
ssize_t (*recvv)(struct fid_ep *ep, const struct iovec *iov, void **desc,
		 size_t count, fi_addr_t src_addr, void *context);
*/
void do_recvv(int len)
{
	int ret;
	ssize_t sz;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	struct iovec iov;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	sz = fi_send(ep[0], source, len, loc_mr[0], gni_addr[1], target);
	cr_assert_eq(sz, 0);

	iov.iov_base = target;
	iov.iov_len = len;

	sz = fi_recvv(ep[1], &iov, (void **)rem_mr, 1, gni_addr[0], source);
	cr_assert_eq(sz, 0);

	/*  need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV), target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, recvv)
{
	rdm_sr_xfer_for_each_size(do_recvv, 1, BUF_SZ);
}

Test(rdm_sr, recvv_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_recvv, 1, BUF_SZ);
}

/*
ssize_t (*recvmsg)(struct fid_ep *ep, const struct fi_msg *msg,
		   uint64_t flags);
*/
void do_recvmsg(int len)
{
	int ret;
	ssize_t sz;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	struct iovec iov;
	struct fi_msg msg;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	sz = fi_send(ep[0], source, len, loc_mr[0], gni_addr[1], target);
	cr_assert_eq(sz, 0);

	iov.iov_base = target;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = (void **)rem_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[0];
	msg.context = source;
	msg.data = (uint64_t)source;

	sz = fi_recvmsg(ep[1], &msg, 0);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV), target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, recvmsg)
{
	rdm_sr_xfer_for_each_size(do_recvmsg, 1, BUF_SZ);
}

Test(rdm_sr, recvmsg_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_recvmsg, 1, BUF_SZ);
}

Test(rdm_sr_bnd_ep, recvmsg)
{
	rdm_sr_xfer_for_each_size(do_recvmsg, 1, BUF_SZ);
}

void do_send_autoreg(int len)
{
	int ret;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	sz = fi_send(ep[0], source, len, NULL, gni_addr[1], target);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, NULL, gni_addr[0], source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV), target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

Test(rdm_sr, send_autoreg)
{
	rdm_sr_xfer_for_each_size(do_send_autoreg, 1, BUF_SZ);
}

Test(rdm_sr, send_autoreg_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_send_autoreg, 1, BUF_SZ);
}

void do_send_autoreg_uncached(int len)
{
	int ret;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(uc_source, len, 0xab);
	rdm_sr_init_data(uc_target, len, 0);

	sz = fi_send(ep[0], uc_source, len, NULL, gni_addr[1], uc_target);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], uc_target, len, NULL, gni_addr[0], uc_source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, uc_target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, uc_source, (FI_MSG|FI_RECV),
			 uc_target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(uc_source, uc_target, len),
		  "Data mismatch");
}

Test(rdm_sr, send_autoreg_uncached)
{
	rdm_sr_xfer_for_each_size(do_send_autoreg_uncached, 1, BUF_SZ);
}

Test(rdm_sr, send_autoreg_uncached_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_send_autoreg_uncached, 1, BUF_SZ);
}

void do_send_err(int len)
{
	int ret;
	struct fi_cq_tagged_entry s_cqe;
	struct fi_cq_err_entry err_cqe;
	ssize_t sz;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	sz = fi_send(ep[0], source, len, loc_mr[0], gni_addr[1], target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(msg_cq[0], &s_cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, -FI_EAVAIL);

	ret = fi_cq_readerr(msg_cq[0], &err_cqe, 0);
	cr_assert_eq(ret, 1);

	cr_assert((uint64_t)err_cqe.op_context == (uint64_t)target,
		  "Bad error context");
	cr_assert(err_cqe.flags == (FI_MSG | FI_SEND));
	cr_assert(err_cqe.len == 0, "Bad error len");
	cr_assert(err_cqe.buf == 0, "Bad error buf");
	cr_assert(err_cqe.data == 0, "Bad error data");
	cr_assert(err_cqe.tag == 0, "Bad error tag");
	cr_assert(err_cqe.olen == 0, "Bad error olen");
	cr_assert(err_cqe.err == FI_ECANCELED, "Bad error errno");
	cr_assert(err_cqe.prov_errno == GNI_RC_TRANSACTION_ERROR,
		  "Bad prov errno");
	cr_assert(err_cqe.err_data == NULL, "Bad error provider data");

	s_e[0] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);
}

Test(rdm_sr, send_err)
{
	int ret, max_retrans_val = 0, i = 0; /* 0 to force SMSG failure */

	for (; i < NUMEPS; i++) {
		ret = gni_domain_ops[i]->set_val(&dom[i]->fid,
						 GNI_MAX_RETRANSMITS,
						 &max_retrans_val);
		cr_assert(!ret, "setval(GNI_MAX_RETRANSMITS)");
	}
	rdm_sr_err_inject_enable();

	rdm_sr_xfer_for_each_size(do_send_err, 1, BUF_SZ);
}

void do_send_autoreg_uncached_nolazydereg(int len)
{
	int ret;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(uc_source, len, 0xab);
	rdm_sr_init_data(uc_target, len, 0);

	sz = fi_send(ep[0], uc_source, len, NULL, gni_addr[1], uc_target);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], uc_target, len, NULL, gni_addr[0], uc_source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, uc_target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, uc_source, (FI_MSG|FI_RECV),
			 uc_target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(uc_source, uc_target, len),
		  "Data mismatch");
}

Test(rdm_sr_noreg, send_autoreg_uncached_nolazydereg)
{
	rdm_sr_lazy_dereg_disable();
	rdm_sr_xfer_for_each_size(do_send_autoreg_uncached_nolazydereg,
				  1, BUF_SZ);
}

Test(rdm_sr, send_readfrom)
{
	int ret;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	fi_addr_t src_addr;
	int len = 64;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	sz = fi_send(ep[0], source, len, loc_mr[0], gni_addr[1], target);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], target, len, rem_mr[0], gni_addr[0], source);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_readfrom(msg_cq[1], &d_cqe, 1, &src_addr);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV), target, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	cr_assert(src_addr == gni_addr[0], "src_addr mismatch");

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(source, target, len), "Data mismatch");
}

void do_send_buf(void *p, void *t, int len)
{
	int ret;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(p, len, 0xab);
	rdm_sr_init_data(t, len, 0);

	sz = fi_send(ep[0], p, len, loc_mr[0], gni_addr[1], t);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], t, len, rem_mr[0], gni_addr[0], p);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	rdm_sr_check_cqe(&s_cqe, t, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, p, (FI_MSG|FI_RECV), t, len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(p, t, len), "Data mismatch");
}

void do_send_alignment(int len)
{
	int s_off, t_off, l_off;

	for (s_off = 0; s_off < 7; s_off++) {
		for (t_off = 0; t_off < 7; t_off++) {
			for (l_off = 0; l_off < 7; l_off++) {
				do_send_buf(source + s_off,
					    target + t_off,
					    len + l_off);
			}
		}
	}
}

Test(rdm_sr, send_alignment)
{
	rdm_sr_xfer_for_each_size(do_send_alignment, 1, (BUF_SZ-1));
}

Test(rdm_sr, send_alignment_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_send_alignment, 1, (BUF_SZ-1));
}

void do_sendrecv_buf(void *p, void *t, int send_len, int recv_len)
{
	int ret;
	int source_done = 0, dest_done = 0;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	int xfer_len;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	rdm_sr_init_data(p, send_len, 0xab);
	rdm_sr_init_data(t, recv_len, 0);

	sz = fi_send(ep[0], p, send_len, loc_mr[0], gni_addr[1], t);
	cr_assert_eq(sz, 0);

	sz = fi_recv(ep[1], t, recv_len, rem_mr[0], gni_addr[0], p);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			source_done = 1;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			dest_done = 1;
		}
	} while (!(source_done && dest_done));

	xfer_len = MIN(send_len, recv_len);
	rdm_sr_check_cqe(&s_cqe, t, (FI_MSG|FI_SEND), 0, 0, 0);
	rdm_sr_check_cqe(&d_cqe, p, (FI_MSG|FI_RECV), t, xfer_len, 0);

	s[0] = 1; r[1] = 1;
	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");

	cr_assert(rdm_sr_check_data(p, t, xfer_len), "Data mismatch");
}

void do_sendrecv_alignment(int len)
{
	int s_off, t_off, sl_off, rl_off;

	for (s_off = 0; s_off < 8; s_off++) {
		for (t_off = 0; t_off < 8; t_off++) {
			for (sl_off = -7; sl_off < 8; sl_off++) {
				for (rl_off = -7; rl_off < 8; rl_off++) {
					do_sendrecv_buf(source + s_off,
							target + t_off,
							len + sl_off,
							len + rl_off);
				}
			}
		}
	}
}

Test(rdm_sr, sendrecv_alignment)
{
	rdm_sr_xfer_for_each_size(do_sendrecv_alignment, 8*1024, 16*1024);
}

Test(rdm_sr, sendrecv_alignment_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_sendrecv_alignment, 8*1024, 32*1024);
}

void do_multirecv(int len)
{
	int i, ret;
	ssize_t sz;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	struct iovec iov;
	struct fi_msg msg;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};
	int nrecvs = 3;

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	/* Post receives first to force matching in SMSG callback. */
	iov.iov_base = target;
	iov.iov_len = len * nrecvs + 63;

	msg.msg_iov = &iov;
	msg.desc = (void **)rem_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[0];
	msg.context = source;
	msg.data = (uint64_t)source;

	sz = fi_recvmsg(ep[1], &msg, FI_MULTI_RECV);
	cr_assert_eq(sz, 0);

	for (i = 0; i < nrecvs; i++) {
		sz = fi_send(ep[0], source, len, loc_mr[0], gni_addr[1],
			     target);
		cr_assert_eq(sz, 0);
	}

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND),
					 0, 0, 0);
			s[0]++;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			rdm_sr_check_cqe(&d_cqe, source,
					 (FI_MSG|FI_RECV|FI_MULTI_RECV),
					 target + (r[1] * len), len, 0);
			cr_assert(rdm_sr_check_data(source, d_cqe.buf, len),
				  "Data mismatch");
			r[1]++;
		}
	} while (s[0] < nrecvs || r[1] < nrecvs);

	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");
}

Test(rdm_sr, multirecv)
{
	rdm_sr_xfer_for_each_size(do_multirecv, 1, BUF_SZ);
}

Test(rdm_sr, multirecv_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_multirecv, 1, BUF_SZ);
}

void do_multirecv2(int len)
{
	int i, ret;
	ssize_t sz;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	struct iovec iov;
	struct fi_msg msg;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};
	int nrecvs = 3;

	rdm_sr_init_data(source, len, 0xab);
	rdm_sr_init_data(target, len, 0);

	/* Post sends first to force matching in the _gnix_recv() path. */
	for (i = 0; i < nrecvs; i++) {
		sz = fi_send(ep[0], source, len, loc_mr[0], gni_addr[1],
			     target);
		cr_assert_eq(sz, 0);
	}

	/* Progress our sends. */
	for (i = 0; i < 10000; i++) {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND),
					 0, 0, 0);
			s[0]++;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		cr_assert_eq(ret, -FI_EAGAIN);
	}

	iov.iov_base = target;
	iov.iov_len = len * nrecvs + 63;

	msg.msg_iov = &iov;
	msg.desc = (void **)rem_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[0];
	msg.context = source;
	msg.data = (uint64_t)source;

	sz = fi_recvmsg(ep[1], &msg, FI_MULTI_RECV);
	cr_assert_eq(sz, 0);

	/* need to progress both CQs simultaneously for rendezvous */
	do {
		ret = fi_cq_read(msg_cq[0], &s_cqe, 1);
		if (ret == 1) {
			rdm_sr_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND),
					 0, 0, 0);
			s[0]++;
		}
		ret = fi_cq_read(msg_cq[1], &d_cqe, 1);
		if (ret == 1) {
			rdm_sr_check_cqe(&d_cqe, source,
					 (FI_MSG|FI_RECV|FI_MULTI_RECV),
					 target + (r[1] * len), len, 0);
			cr_assert(rdm_sr_check_data(source, d_cqe.buf, len),
				  "Data mismatch");
			r[1]++;
		}
	} while (s[0] < nrecvs || r[1] < nrecvs);

	rdm_sr_check_cntrs(s, r, s_e, r_e);

	dbg_printf("got context events!\n");
}

Test(rdm_sr, multirecv2)
{
	rdm_sr_xfer_for_each_size(do_multirecv2, 1, BUF_SZ);
}

Test(rdm_sr, multirecv2_retrans)
{
	rdm_sr_err_inject_enable();
	rdm_sr_xfer_for_each_size(do_multirecv2, 1, BUF_SZ);
}

