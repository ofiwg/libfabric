/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
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
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <assert.h>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <criterion/criterion.h>
#include "gnix_rdma_headers.h"
#include "common.h"
#include "fi_ext_gni.h"
#include "gnix.h"

#if 1
#define dbg_printf(...)
#else
#define dbg_printf(...)				\
	do {					\
		printf(__VA_ARGS__);		\
		fflush(stdout);			\
	} while (0)
#endif

#define NUMEPS 2
#define BUF_SZ (1<<20)
#define IOV_CNT (4)

static struct fid_fabric *fab;
static struct fid_domain *dom[NUMEPS];
static struct fid_ep *sep[NUMEPS];
static struct fid_av *av[NUMEPS];
static struct fid_av *t_av;
static void *ep_name[NUMEPS];
fi_addr_t gni_addr[NUMEPS];
static struct fi_cq_attr cq_attr;
struct fi_info *hints;
static struct fi_info *fi[NUMEPS];
static struct fid_ep *sep[NUMEPS];

char *target;
char *source;
struct iovec *src_iov, *dest_iov;
char *iov_src_buf, *iov_dest_buf;
struct fid_mr *rem_mr[NUMEPS], *loc_mr[NUMEPS];
struct fid_mr *iov_dest_buf_mr[NUMEPS], *iov_src_buf_mr[NUMEPS];
uint64_t mr_key[NUMEPS];

static int ctx_cnt = 4;
static int rx_ctx_bits;
static struct fid_ep **tx_ep[NUMEPS], **rx_ep[NUMEPS];
static struct fid_cq **tx_cq[NUMEPS];
static struct fid_cq **rx_cq[NUMEPS];
static fi_addr_t *rx_addr;
static struct fid_cntr *send_cntr[NUMEPS], *recv_cntr[NUMEPS];
static struct fi_cntr_attr cntr_attr = {
	.events = FI_CNTR_EVENTS_COMP,
	.flags = 0
};
static uint64_t sends[NUMEPS] = {0}, recvs[NUMEPS] = {0},
	send_errs[NUMEPS] = {0}, recv_errs[NUMEPS] = {0};

void sep_setup(void)
{
	int ret, i, j;
	struct fi_av_attr av_attr = {0};
	size_t addrlen = 0;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_ATOMIC | FI_RMA | FI_MSG | FI_NAMED_RX_CTX;
	hints->mode = FI_LOCAL_MR;
	hints->domain_attr->cq_data_size = NUMEPS * 2;
	hints->domain_attr->data_progress = FI_PROGRESS_AUTO;
	hints->domain_attr->mr_mode = FI_MR_BASIC;
	hints->fabric_attr->prov_name = strdup("gni");
	hints->ep_attr->tx_ctx_cnt = ctx_cnt;
	hints->ep_attr->rx_ctx_cnt = ctx_cnt;

	for (i = 0; i < NUMEPS; i++) {
		ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi[i]);
		cr_assert(!ret, "fi_getinfo");

		tx_cq[i] = calloc(ctx_cnt, sizeof(*tx_cq));
		rx_cq[i] = calloc(ctx_cnt, sizeof(*rx_cq));
		tx_ep[i] = calloc(ctx_cnt, sizeof(*tx_ep));
		rx_ep[i] = calloc(ctx_cnt, sizeof(*rx_ep));
		if (!tx_cq[i] || !tx_cq[i] ||
		    !tx_ep[i] || !rx_ep[i]) {
			cr_assert(0, "calloc");
		}
	}

	ctx_cnt = MIN(ctx_cnt, fi[0]->domain_attr->rx_ctx_cnt);
	ctx_cnt = MIN(ctx_cnt, fi[0]->domain_attr->tx_ctx_cnt);
	cr_assert(ctx_cnt, "ctx_cnt is 0");

	ret = fi_fabric(fi[0]->fabric_attr, &fab, NULL);
	cr_assert(!ret, "fi_fabric");

	while (ctx_cnt >> ++rx_ctx_bits);
	av_attr.rx_ctx_bits = rx_ctx_bits;
	av_attr.type = FI_AV_MAP;
	av_attr.count = NUMEPS;

	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cq_attr.size = 1024;
	cq_attr.wait_obj = FI_WAIT_NONE;

	rx_addr = calloc(ctx_cnt, sizeof(*rx_addr));
	target = calloc(BUF_SZ, 1);
	source = calloc(BUF_SZ, 1);
	iov_src_buf = malloc(BUF_SZ * IOV_CNT);
	iov_dest_buf = malloc(BUF_SZ * IOV_CNT);
	src_iov = malloc(sizeof(struct iovec) * IOV_CNT);
	dest_iov = malloc(sizeof(struct iovec) * IOV_CNT);

	if (!rx_addr || !target || !source || !iov_src_buf || !iov_dest_buf ||
	    !src_iov || !dest_iov) {
		cr_assert(0, "allocation");
	}

	for (i = 0; i < IOV_CNT; i++) {
		src_iov[i].iov_base = malloc(BUF_SZ);
		assert(src_iov[i].iov_base != NULL);

		dest_iov[i].iov_base = malloc(BUF_SZ * 3);
		assert(dest_iov[i].iov_base != NULL);
	}

	for (i = 0; i < NUMEPS; i++) {
		fi[i]->ep_attr->tx_ctx_cnt = ctx_cnt;
		fi[i]->ep_attr->rx_ctx_cnt = ctx_cnt;

		ret = fi_domain(fab, fi[i], &dom[i], NULL);
		cr_assert(!ret, "fi_domain");

		ret = fi_scalable_ep(dom[i], fi[i], &sep[i], NULL);
		cr_assert(!ret, "fi_scalable_ep");

		ret = fi_av_open(dom[i], &av_attr, &av[i], NULL);
		cr_assert(!ret, "fi_av_open");

		ret = fi_cntr_open(dom[i], &cntr_attr, &send_cntr[i], 0);
		cr_assert(!ret, "fi_cntr_open");

		ret = fi_cntr_open(dom[i], &cntr_attr, &recv_cntr[i], 0);
		cr_assert(!ret, "fi_cntr_open");

		for (j = 0; j < ctx_cnt; j++) {
			ret = fi_tx_context(sep[i], j, NULL, &tx_ep[i][j],
					    NULL);
			cr_assert(!ret, "fi_tx_context");

			ret = fi_cq_open(dom[i], &cq_attr, &tx_cq[i][j],
					 NULL);
			cr_assert(!ret, "fi_cq_open");

			ret = fi_rx_context(sep[i], j, NULL, &rx_ep[i][j],
					    NULL);
			cr_assert(!ret, "fi_rx_context");

			ret = fi_cq_open(dom[i], &cq_attr, &rx_cq[i][j],
					 NULL);
			cr_assert(!ret, "fi_cq_open");
		}

		ret = fi_scalable_ep_bind(sep[i], &av[i]->fid, 0);
		cr_assert(!ret, "fi_scalable_ep_bind");

		for (j = 0; j < ctx_cnt; j++) {
			ret = fi_ep_bind(tx_ep[i][j], &tx_cq[i][j]->fid,
					 FI_TRANSMIT);
			cr_assert(!ret, "fi_ep_bind");

			ret = fi_ep_bind(tx_ep[i][j], &send_cntr[i]->fid,
					 FI_SEND | FI_WRITE);
			cr_assert(!ret, "fi_ep_bind");

			ret = fi_enable(tx_ep[i][j]);
			cr_assert(!ret, "fi_enable");

			ret = fi_ep_bind(rx_ep[i][j], &rx_cq[i][j]->fid,
					 FI_RECV);
			cr_assert(!ret, "fi_ep_bind");

			ret = fi_ep_bind(rx_ep[i][j], &recv_cntr[i]->fid,
					 FI_RECV | FI_READ);
			cr_assert(!ret, "fi_ep_bind");

			ret = fi_enable(rx_ep[i][j]);
			cr_assert(!ret, "fi_enable");

		}
	}

	for (i = 0; i < NUMEPS; i++) {
		ret = fi_enable(sep[i]);
		cr_assert(!ret, "fi_enable");

		ret = fi_getname(&sep[i]->fid, NULL, &addrlen);
		cr_assert(addrlen > 0);

		ep_name[i] = malloc(addrlen);
		cr_assert(ep_name[i] != NULL);

		ret = fi_getname(&sep[i]->fid, ep_name[i], &addrlen);
		cr_assert(ret == FI_SUCCESS);

		ret = fi_mr_reg(dom[i], target, BUF_SZ, FI_REMOTE_WRITE,
				0, 0, 0, &rem_mr[i], &target);
		cr_assert_eq(ret, 0);

		ret = fi_mr_reg(dom[i], source, BUF_SZ, FI_REMOTE_WRITE,
				0, 0, 0, &loc_mr[i], &source);
		cr_assert_eq(ret, 0);

		mr_key[i] = fi_mr_key(rem_mr[i]);

		ret = fi_mr_reg(dom[i], iov_dest_buf, IOV_CNT * BUF_SZ,
				FI_REMOTE_WRITE, 0, 0, 0, iov_dest_buf_mr + i,
				&iov_dest_buf);
		cr_assert_eq(ret, 0);

		ret = fi_mr_reg(dom[i], iov_src_buf, IOV_CNT * BUF_SZ,
				FI_REMOTE_WRITE, 0, 0, 0, iov_src_buf_mr + i,
				&iov_src_buf);
		cr_assert_eq(ret, 0);

	}

	for (i = 0; i < NUMEPS; i++) {
		for (j = 0; j < NUMEPS; j++) {
			ret = fi_av_insert(av[i], ep_name[j], 1, &gni_addr[j],
					   0, NULL);
			cr_assert(ret == 1);
		}
	}

	for (i = 0; i < ctx_cnt; i++) {
		rx_addr[i] = fi_rx_addr(gni_addr[1], i, rx_ctx_bits);
		dbg_printf("fi_rx_addr[%d] %016lx\n", i, rx_addr[i]);
	}
}

static void sep_teardown(void)
{
	int ret, i, j;

	for (i = 0; i < NUMEPS; i++) {
		fi_close(&recv_cntr[i]->fid);
		fi_close(&send_cntr[i]->fid);

		for (j = 0; j < ctx_cnt; j++) {
			ret = fi_close(&tx_ep[i][j]->fid);
			cr_assert(!ret, "failure closing tx_ep.");

			ret = fi_close(&rx_ep[i][j]->fid);
			cr_assert(!ret, "failure closing rx_ep.");

			ret = fi_close(&tx_cq[i][j]->fid);
			cr_assert(!ret, "failure closing tx cq.");

			ret = fi_close(&rx_cq[i][j]->fid);
			cr_assert(!ret, "failure closing rx cq.");
		}

		cr_assert(1);

		ret = fi_close(&sep[i]->fid);
		cr_assert(!ret, "failure in closing ep.");

		ret = fi_close(&av[i]->fid);
		cr_assert(!ret, "failure in closing av.");

		fi_close(&loc_mr[i]->fid);
		fi_close(&rem_mr[i]->fid);

		ret = fi_close(&dom[i]->fid);
		cr_assert(!ret, "failure in closing domain.");

		free(tx_ep[i]);
		free(rx_ep[i]);
		free(ep_name[i]);
		fi_freeinfo(fi[i]);
	}

	for (i = 0; i < IOV_CNT; i++) {
		free(src_iov[i].iov_base);
		free(dest_iov[i].iov_base);
	}

	free(src_iov);
	free(dest_iov);
	free(iov_src_buf);
	free(iov_dest_buf);

	fi_freeinfo(hints);
	free(target);
	free(source);

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "failure in closing fabric.");
}

static void
sep_init_data(char *buf, int len, char seed)
{
	int i;

	for (i = 0; i < len; i++)
		buf[i] = seed++;
}

static int
sep_check_data(char *buf1, char *buf2, int len)
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

static void
wait_for_cqs(struct fid_cq *scq, struct fid_cq *dcq,
		struct fi_cq_tagged_entry *scqe,
		struct fi_cq_tagged_entry *dcqe)
{
	int ret;
	int s_done = 0, d_done = 0;

	do {
		ret = fi_cq_read(scq, scqe, 1);
		if (ret == 1) {
			s_done = 1;
		}

		ret = fi_cq_read(dcq, dcqe, 1);
		if (ret == 1) {
			d_done = 1;
		}
	} while (!(s_done && d_done));
}

static void
xfer_each_size(void (*xfer)(int index, int len), int index, int slen, int elen)
{
	int i;

	for (i = slen; i <= elen; i *= 2) {
		xfer(index, i);
	}
}

static void
sep_check_cqe(struct fi_cq_tagged_entry *cqe, void *ctx,
		uint64_t flags, void *addr, size_t len,
		uint64_t data, bool buf_is_non_null)
{
	cr_assert(cqe->op_context == ctx, "CQE Context mismatch");
	cr_assert(cqe->flags == flags, "CQE flags mismatch");

	if (flags & FI_RECV) {
		cr_assert(cqe->len == len, "CQE length mismatch");

		if (buf_is_non_null)
			cr_assert(cqe->buf == addr, "CQE address mismatch");
		else
			cr_assert(cqe->buf == NULL, "CQE address mismatch");


		if (flags & FI_REMOTE_CQ_DATA)
			cr_assert(cqe->data == data, "CQE data mismatch");
	} else {
		cr_assert(cqe->len == 0, "Invalid CQE length");
		cr_assert(cqe->buf == 0, "Invalid CQE address");
		cr_assert(cqe->data == 0, "Invalid CQE data");
	}

	cr_assert(cqe->tag == 0, "Invalid CQE tag");
}

static void
sep_check_tcqe(struct fi_cq_tagged_entry *tcqe, void *ctx,
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

static void
sep_check_cntrs(uint64_t s[], uint64_t r[], uint64_t s_e[],
		uint64_t r_e[])
{
	int i = 0;

	for (; i < NUMEPS; i++) {
		sends[i] += s[i];
		recvs[i] += r[i];
		send_errs[i] += s_e[i];
		recv_errs[i] += r_e[i];

		cr_assert(fi_cntr_read(send_cntr[i]) == sends[i],
			  "Bad send count i:%d send_cntr:%ld sends:%ld",
			  i, fi_cntr_read(send_cntr[i]), sends[i]);
		cr_assert(fi_cntr_read(recv_cntr[i]) == recvs[i],
			  "Bad recv count");
		cr_assert(fi_cntr_readerr(send_cntr[i]) == send_errs[i],
			  "Bad send err count");
		cr_assert(fi_cntr_readerr(recv_cntr[i]) == recv_errs[i],
			  "Bad recv err count");
	}
}

static int
sep_check_iov_data(struct iovec *iov_buf, char *buf, size_t cnt, size_t buf_len)
{
	size_t i, j, cum_len = 0, len, iov_idx;

	for (i = 0; i < cnt; i++) {
		cum_len += iov_buf[i].iov_len;
	}

	len = MIN(cum_len, buf_len);

	cum_len = iov_buf[0].iov_len;

	for (i = j = iov_idx = 0; j < len; j++, iov_idx++) {

		if (j == cum_len) {
			i++, iov_idx = 0;
			cum_len += iov_buf[i].iov_len;

			if (i >= cnt)
				break;
		}

		if (((char *)iov_buf[i].iov_base)[iov_idx] != buf[j]) {
			printf("data mismatch, iov_index: %lu, elem: %lu, "
			       "iov_buf_len: %lu, "
			       " iov_buf: %hhx, buf: %hhx\n", i, j,
			       iov_buf[i].iov_len,
			       ((char *)iov_buf[i].iov_base)[iov_idx],
			       buf[j]);
			return 0;
		}
	}

	return 1;
}

static int
check_iov_data(struct iovec *ib, struct iovec *ob, size_t cnt)
{
	size_t i;

	for (i = 0; i < cnt; i++) {
		if (memcmp(ib[i].iov_base, ob[i].iov_base, ib[i].iov_len)) {
			printf("data mismatch, ib:%x ob:%x\n",
			       *(char *)ib[i].iov_base,
			       *(char *)ob[i].iov_base);
			return 0;
		}
	}

	return 1;
}

/*******************************************************************************
 * Test MSG functions
 ******************************************************************************/

TestSuite(scalable, .init = sep_setup, .fini = sep_teardown);

Test(scalable, bind)
{
	int ret;
	struct fi_av_attr av_attr = {0};

	/* test if bind fails */
	ret = fi_ep_bind(tx_ep[0][0], &tx_cq[0][0]->fid,
			 FI_TRANSMIT);
	cr_assert(ret, "fi_ep_bind should fail");

	ret = fi_ep_bind(rx_ep[0][0], &rx_cq[0][0]->fid,
			 FI_TRANSMIT);
	cr_assert(ret, "fi_ep_bind should fail");

	/* test for inserting an ep_name that doesn't fit in the AV */
	av_attr.type = FI_AV_MAP;
	av_attr.count = NUMEPS;
	av_attr.rx_ctx_bits = 1;

	ret = fi_av_open(dom[0], &av_attr, &t_av, NULL);
	cr_assert(!ret, "fi_av_open");
	ret = fi_av_insert(t_av, ep_name[0], 1, &gni_addr[0], 0, NULL);
	cr_assert(ret == -FI_EINVAL);
	ret = fi_close(&t_av->fid);
	cr_assert(!ret, "failure in closing av.");
}

/*
 * ssize_t fi_send(struct fid_ep *ep, void *buf, size_t len,
 *		void *desc, fi_addr_t dest_addr, void *context);
 *
 * ssize_t fi_recv(struct fid_ep *ep, void * buf, size_t len,
 *		void *desc, fi_addr_t src_addr, void *context);
 */
static void sep_send_recv(int index, int len)
{
	ssize_t ret;
	struct fi_cq_tagged_entry cqe;

	sep_init_data(source, len, 0xab + index);
	sep_init_data(target, len, 0);

	ret = fi_send(tx_ep[0][index], source, len, loc_mr[0],
		      rx_addr[index], target);
	cr_assert(ret == 0, "fi_send failed err:%ld", ret);

	ret = fi_recv(rx_ep[1][index], target, len, rem_mr[0],
		      FI_ADDR_UNSPEC, source);
	cr_assert(ret == 0, "fi_recv failed err:%ld", ret);

	wait_for_cqs(tx_cq[0][index], rx_cq[1][index], &cqe, &cqe);

	ret = sep_check_data(source, target, 8);
	cr_assert(ret == 1, "Data check failed");
}

Test(scalable, sr)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_send_recv, i, 1, BUF_SZ);
	}
}

/*
ssize_t fi_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		 size_t count, fi_addr_t dest_addr, void *context);
*/
static void sep_sendv(int index, int len)
{
	int i, iov_cnt;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	ssize_t sz;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	for (iov_cnt = 1; iov_cnt <= IOV_CNT; iov_cnt++) {
		for (i = 0; i < iov_cnt; i++) {
			sep_init_data(src_iov[i].iov_base, len, 0x25);
			src_iov[i].iov_len = len;
		}
		sep_init_data(iov_dest_buf, len * iov_cnt, 0);

		sz = fi_sendv(tx_ep[0][index], src_iov, NULL, iov_cnt,
				rx_addr[index], iov_dest_buf);
		cr_assert_eq(sz, 0);

		sz = fi_recv(rx_ep[1][index], iov_dest_buf, len * iov_cnt,
			     iov_dest_buf_mr[0], FI_ADDR_UNSPEC, src_iov);
		cr_assert_eq(sz, 0);

		/* reset cqe */
		s_cqe.op_context = s_cqe.buf = (void *) -1;
		s_cqe.flags = s_cqe.len = s_cqe.data = s_cqe.tag = UINT_MAX;
		d_cqe.op_context = d_cqe.buf = (void *) -1;
		d_cqe.flags = d_cqe.len = d_cqe.data = d_cqe.tag = UINT_MAX;

		wait_for_cqs(tx_cq[0][index], rx_cq[1][index], &s_cqe, &d_cqe);
		sep_check_cqe(&s_cqe, iov_dest_buf, (FI_MSG|FI_SEND),
				 0, 0, 0, false);
		sep_check_cqe(&d_cqe, src_iov, (FI_MSG|FI_RECV), iov_dest_buf,
				 len * iov_cnt, 0, false);

		s[0] = 1; r[1] = 1;
		sep_check_cntrs(s, r, s_e, r_e);
		cr_assert(sep_check_iov_data(src_iov, iov_dest_buf, iov_cnt,
			len * iov_cnt), "Data mismatch");
	}
}

Test(scalable, sendv)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_sendv, i, 1, BUF_SZ);
	}
}

static void sep_recvv(int index, int len)
{
	int i, iov_cnt;
	ssize_t sz;
	struct fi_cq_tagged_entry s_cqe, d_cqe;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	for (iov_cnt = 1; iov_cnt <= IOV_CNT; iov_cnt++) {
		for (i = 0; i < iov_cnt; i++) {
			sep_init_data(src_iov[i].iov_base, len, 0x25 + index);
			src_iov[i].iov_len = len;
		}

		for (i = 0; i < iov_cnt; i++) {
			sep_init_data(dest_iov[i].iov_base, len, 0);
			dest_iov[i].iov_len = len;
		}

		sz = fi_sendv(tx_ep[0][index], src_iov, NULL, iov_cnt,
				rx_addr[index], iov_dest_buf);
		cr_assert_eq(sz, 0);

		sz = fi_recvv(rx_ep[1][index], dest_iov, NULL, iov_cnt,
				FI_ADDR_UNSPEC, iov_src_buf);
		cr_assert_eq(sz, 0);

		/* reset cqe */
		s_cqe.op_context = s_cqe.buf = (void *) -1;
		s_cqe.flags = s_cqe.len = s_cqe.data = s_cqe.tag = UINT_MAX;
		d_cqe.op_context = d_cqe.buf = (void *) -1;
		d_cqe.flags = d_cqe.len = d_cqe.data = d_cqe.tag = UINT_MAX;

		wait_for_cqs(tx_cq[0][index], rx_cq[1][index], &s_cqe, &d_cqe);
		sep_check_cqe(&s_cqe, iov_dest_buf, (FI_MSG|FI_SEND),
				 0, 0, 0, false);
		sep_check_cqe(&d_cqe, iov_src_buf, (FI_MSG|FI_RECV),
				iov_dest_buf, len * iov_cnt, 0, false);

		s[0] = 1; r[1] = 1;
		sep_check_cntrs(s, r, s_e, r_e);
		cr_assert(check_iov_data(src_iov, dest_iov, iov_cnt),
			  "Data mismatch");
	}
}

Test(scalable, recvv)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_recvv, i, 1, BUF_SZ);
	}
}

static void sep_sendmsg(int index, int len)
{
	ssize_t sz;
	struct fi_cq_tagged_entry s_cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					    (void *) -1, UINT_MAX, UINT_MAX };
	struct fi_cq_tagged_entry d_cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					    (void *) -1, UINT_MAX, UINT_MAX };
	struct fi_msg msg;
	struct iovec iov;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = rx_addr[index];
	msg.context = target;
	msg.data = (uint64_t)target;

	sep_init_data(source, len, 0xd0 + index);
	sep_init_data(target, len, 0);

	sz = fi_sendmsg(tx_ep[0][index], &msg, 0);
	cr_assert_eq(sz, 0);

	sz = fi_recv(rx_ep[1][index], target, len, rem_mr[0],
		     FI_ADDR_UNSPEC, source);
	cr_assert_eq(sz, 0);

	wait_for_cqs(tx_cq[0][index], rx_cq[1][index], &s_cqe, &d_cqe);
	sep_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0, false);
	sep_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV), target, len, 0,
			false);

	s[0] = 1; r[1] = 1;
	sep_check_cntrs(s, r, s_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, sendmsg)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_sendmsg, i, 1, BUF_SZ);
	}
}

void sep_sendmsgdata(int index, int len)
{
	ssize_t sz;
	struct fi_cq_tagged_entry s_cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					    (void *) -1, UINT_MAX, UINT_MAX };
	struct fi_cq_tagged_entry d_cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					    (void *) -1, UINT_MAX, UINT_MAX };
	struct fi_msg msg;
	struct iovec iov;
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = rx_addr[index];
	msg.context = target;
	msg.data = (uint64_t)source;

	sep_init_data(source, len, 0xe0 + index);
	sep_init_data(target, len, 0);

	sz = fi_sendmsg(tx_ep[0][index], &msg, FI_REMOTE_CQ_DATA);
	cr_assert_eq(sz, 0);

	sz = fi_recv(rx_ep[1][index], target, len, rem_mr[0],
		     FI_ADDR_UNSPEC, source);
	cr_assert_eq(sz, 0);

	wait_for_cqs(tx_cq[0][index], rx_cq[1][index], &s_cqe, &d_cqe);
	sep_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0, false);
	sep_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV|FI_REMOTE_CQ_DATA),
		      target, len, (uint64_t)source, false);

	s[0] = 1; r[1] = 1;
	sep_check_cntrs(s, r, s_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, sendmsgdata)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_sendmsgdata, i, 1, BUF_SZ);
	}
}

#define INJECT_SIZE 64
void sep_inject(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	sep_init_data(source, len, 0x13 + index);
	sep_init_data(target, len, 0);

	sz = fi_inject(tx_ep[0][index], source, len, rx_addr[index]);
	cr_assert_eq(sz, 0);

	sz = fi_recv(rx_ep[1][index], target, len, rem_mr[1],
		     FI_ADDR_UNSPEC, source);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(rx_cq[1][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
		/* Manually progress connection to domain 1 */
		fi_cq_read(tx_cq[0][index], &cqe, 1);
	}

	cr_assert_eq(ret, 1);
	sep_check_cqe(&cqe, source, (FI_MSG|FI_RECV),
			 target, len, (uint64_t)source, false);

	/* do progress until send counter is updated */
	while (fi_cntr_read(send_cntr[0]) < 1) {
		pthread_yield();
	}
	s[0] = 1; r[1] = 1;
	sep_check_cntrs(s, r, s_e, r_e);

	/* make sure inject does not generate a send competion */
	cr_assert_eq(fi_cq_read(tx_cq[0][index], &cqe, 1), -FI_EAGAIN);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, inject)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_inject, i, 1, INJECT_SIZE);
	}
}

/*
ssize_t fi_senddata(struct fid_ep *ep, void *buf, size_t len, void *desc,
		    uint64_t data, fi_addr_t dest_addr, void *context);
*/
void sep_senddata(int index, int len)
{
	ssize_t sz;
	struct fi_cq_tagged_entry s_cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					    (void *) -1, UINT_MAX, UINT_MAX };
	struct fi_cq_tagged_entry d_cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					    (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	sep_init_data(source, len, 0xab + index);
	sep_init_data(target, len, 0);

	sz = fi_senddata(tx_ep[0][index], source, len, loc_mr[0],
			 (uint64_t)source, rx_addr[index], target);
	cr_assert_eq(sz, 0);

	sz = fi_recv(rx_ep[1][index], target, len, rem_mr[0],
		     FI_ADDR_UNSPEC, source);
	cr_assert_eq(sz, 0);

	wait_for_cqs(tx_cq[0][index], rx_cq[1][index], &s_cqe, &d_cqe);
	sep_check_cqe(&s_cqe, target, (FI_MSG|FI_SEND), 0, 0, 0, false);
	sep_check_cqe(&d_cqe, source, (FI_MSG|FI_RECV|FI_REMOTE_CQ_DATA),
			 target, len, (uint64_t)source, false);

	s[0] = 1; r[1] = 1;
	sep_check_cntrs(s, r, s_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, senddata)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_senddata, i, 1, INJECT_SIZE);
	}
}

/*
ssize_t fi_injectdata(struct fid_ep *ep, const void *buf, size_t len,
		      uint64_t data, fi_addr_t dest_addr);
*/
void sep_injectdata(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t s[NUMEPS] = {0}, r[NUMEPS] = {0}, s_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	sep_init_data(source, len, 0x9b + index);
	sep_init_data(target, len, 0);

	sz = fi_injectdata(tx_ep[0][index], source, len, (uint64_t)source,
			   rx_addr[index]);
	cr_assert_eq(sz, 0);

	sz = fi_recv(rx_ep[1][index], target, len, rem_mr[0],
		     FI_ADDR_UNSPEC, source);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(rx_cq[1][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
		/* Manually progress connection to domain 1 */
		fi_cq_read(tx_cq[0][index], &cqe, 1);
	}
	sep_check_cqe(&cqe, source, (FI_MSG|FI_RECV|FI_REMOTE_CQ_DATA),
			 target, len, (uint64_t)source, false);

	/* don't progress until send counter is updated */
	while (fi_cntr_read(send_cntr[0]) < 1) {
		pthread_yield();
	}

	s[0] = 1; r[1] = 1;
	sep_check_cntrs(s, r, s_e, r_e);

	/* make sure inject does not generate a send competion */
	cr_assert_eq(fi_cq_read(tx_cq[0][index], &cqe, 1), -FI_EAGAIN);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, injectdata)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_injectdata, i, 1, INJECT_SIZE);
	}
}

void sep_read(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t w[2] = {0}, r[2] = {0}, w_e[2] = {0}, r_e[2] = {0};

#define READ_CTX 0x4e3dda1aULL
	sep_init_data(source, len, 0);
	sep_init_data(target, len, 0xad);

	sz = fi_read(tx_ep[0][index], source, len,
		     loc_mr[0], rx_addr[index], (uint64_t)target, mr_key[1],
		     (void *)READ_CTX);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, (void *)READ_CTX, FI_RMA | FI_READ, 0);

	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, read)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_read, i, 8, BUF_SZ);
	}
}

void sep_readv(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	struct iovec iov;
	uint64_t w[2] = {0}, r[2] = {0}, w_e[2] = {0}, r_e[2] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	sep_init_data(target, len, 0x25);
	sep_init_data(source, len, 0);
	sz = fi_readv(tx_ep[0][index], &iov, (void **)loc_mr, 1,
		      rx_addr[index], (uint64_t)target, mr_key[1],
		      target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_RMA | FI_READ, 0);

	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, readv)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_readv, i, 8, BUF_SZ);
	}
}

void sep_readmsg(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	struct iovec iov;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;
	uint64_t w[2] = {0}, r[2] = {0}, w_e[2] = {0}, r_e[2] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	rma_iov.addr = (uint64_t)target;
	rma_iov.len = len;
	rma_iov.key = mr_key[1];

	msg.msg_iov = &iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = rx_addr[index];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = target;
	msg.data = (uint64_t)target;

	sep_init_data(target, len, 0xe0 + index);
	sep_init_data(source, len, 0);
	sz = fi_readmsg(tx_ep[0][index], &msg, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_RMA | FI_READ, 0);

	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, readmsg)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_readmsg, i, 8, BUF_SZ);
	}
}

void sep_write(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t w[2] = {0}, r[2] = {0}, w_e[2] = {0}, r_e[2] = {0};

	sep_init_data(source, len, 0xab);
	sep_init_data(target, len, 0);

	sz = fi_write(tx_ep[0][index], source, len, loc_mr[0], rx_addr[index],
		      (uint64_t)target, mr_key[1], target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);

	w[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, write)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_write, i, 8, BUF_SZ);
	}
}

void sep_writev(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	struct iovec iov;
	uint64_t w[2] = {0}, r[2] = {0}, w_e[2] = {0}, r_e[2] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	sep_init_data(source, len, 0x25 + index);
	sep_init_data(target, len, 0);

	sz = fi_writev(tx_ep[0][index], &iov, (void **)loc_mr, 1,
		       gni_addr[1], (uint64_t)target, mr_key[1],
		       target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);

	w[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, writev)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_writev, i, 8, BUF_SZ);
	}
}

void sep_writemsg(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	struct iovec iov;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;
	uint64_t w[2] = {0}, r[2] = {0}, w_e[2] = {0}, r_e[2] = {0};

	iov.iov_base = source;
	iov.iov_len = len;

	rma_iov.addr = (uint64_t)target;
	rma_iov.len = len;
	rma_iov.key = mr_key[1];

	msg.msg_iov = &iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = target;
	msg.data = (uint64_t)target;

	sep_init_data(source, len, 0xe4 + index);
	sep_init_data(target, len, 0);
	sz = fi_writemsg(tx_ep[0][index], &msg, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);

	w[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");
}

Test(scalable, writemsg)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_writemsg, i, 8, BUF_SZ);
	}
}

void sep_inject_write(int index, int len)
{
	ssize_t sz;
	int ret, i;
	struct fi_cq_tagged_entry cqe;

	sep_init_data(source, len, 0x33);
	sep_init_data(target, len, 0);
	sz = fi_inject_write(tx_ep[0][index], source, len,
			     rx_addr[index], (uint64_t)target, mr_key[1]);
	cr_assert_eq(sz, 0, "fi_inject_write returned %ld (%s)", sz,
		     fi_strerror(-sz));

	for (i = 0; i < len; i++) {
		while (source[i] != target[i]) {
			/* for progress */
			ret = fi_cq_read(tx_cq[0][index], &cqe, 1);
			cr_assert(ret == -FI_EAGAIN || ret == -FI_EAVAIL,
				  "Received unexpected event\n");

			pthread_yield();
		}
	}
}

Test(scalable, injectwrite)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_inject_write, i, 8, INJECT_SIZE);
	}
}

void sep_writedata(int index, int len)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	struct fi_cq_tagged_entry dcqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t w[2] = {0}, r[2] = {0}, w_e[2] = {0}, r_e[2] = {0};


#define WRITE_DATA 0x5123da1a145
	sep_init_data(source, len, 0x43 + index);
	sep_init_data(target, len, 0);
	sz = fi_writedata(tx_ep[0][index], source, len, loc_mr[0], WRITE_DATA,
			  rx_addr[index], (uint64_t)target, mr_key[1],
			  target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_RMA | FI_WRITE, 0);

	w[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	cr_assert(sep_check_data(source, target, len), "Data mismatch");

	while ((ret = fi_cq_read(rx_cq[1][index], &dcqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}
	cr_assert(ret != FI_SUCCESS, "Missing remote data");

	sep_check_tcqe(&dcqe, NULL,
		       (FI_RMA | FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA),
		       WRITE_DATA);
}

Test(scalable, writedata)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_writedata, i, 8, BUF_SZ);
	}
}

#define INJECTWRITE_DATA 0xdededadadeadbeaf
void sep_inject_writedata(int index, int len)
{
	ssize_t sz;
	int ret, i;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	struct fi_cq_tagged_entry dcqe = { (void *) -1, UINT_MAX, UINT_MAX,
					   (void *) -1, UINT_MAX, UINT_MAX };

	sep_init_data(source, len, 0x53 + index);
	sep_init_data(target, len, 0);
	sz = fi_inject_writedata(tx_ep[0][index], source, len, INJECTWRITE_DATA,
				 rx_addr[index], (uint64_t)target, mr_key[1]);
	cr_assert_eq(sz, 0);

	for (i = 0; i < len; i++) {
		while (source[i] != target[i]) {
			/* for progress */
			ret = fi_cq_read(tx_cq[0][index], &cqe, 1);
			cr_assert(ret == -FI_EAGAIN || ret == -FI_EAVAIL,
				  "Received unexpected event\n");

			pthread_yield();
		}
	}

	while ((ret = fi_cq_read(rx_cq[1][index], &dcqe, 1)) == -FI_EAGAIN) {
		ret = fi_cq_read(tx_cq[0][index], &cqe, 1); /* for progress */
		pthread_yield();
	}
	cr_assert(ret != FI_SUCCESS, "Missing remote data");

	sep_check_tcqe(&dcqe, NULL,
		      (FI_RMA | FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA),
		      INJECTWRITE_DATA);
}

Test(scalable, inject_writedata)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_inject_writedata, i, 8, INJECT_SIZE);
	}
}

#define SOURCE_DATA	0xBBBB0000CCCCULL
#define TARGET_DATA	0xAAAA0000DDDDULL
void sep_atomic(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	/* u64 */
	*((uint64_t *)source) = SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_atomic(tx_ep[0][index], source, 1,
		       loc_mr[0], rx_addr[index], (uint64_t)target, mr_key[1],
		       FI_UINT64, FI_ATOMIC_WRITE, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);

	w[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");

}

Test(scalable, atomic)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic(i);
	}
}

void sep_atomic_v(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t min;
	struct fi_ioc iov;
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	iov.addr = source;
	iov.count = 1;

	/* i64 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_atomicv(tx_ep[0][index], &iov, (void **)loc_mr, 1,
			rx_addr[index], (uint64_t)target, mr_key[1],
			FI_INT64, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);

	w[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
		SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
}

Test(scalable, atomicv)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_v(i);
	}
}

#define U32_MASK	0xFFFFFFFFULL
void sep_atomic_msg(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t min;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov;
	struct fi_rma_ioc rma_iov;
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	msg_iov.addr = source;
	msg_iov.count = 1;
	msg.msg_iov = &msg_iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	rma_iov.addr = (uint64_t)target;
	rma_iov.count = 1;
	rma_iov.key = mr_key[1];
	msg.rma_iov = &rma_iov;
	msg.context = target;
	msg.op = FI_MIN;

	/* i32 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	msg.datatype = FI_INT32;
	sz = fi_atomicmsg(tx_ep[0][index], &msg, 0);
	cr_assert_eq(sz, 0);

	/* reset cqe */
	cqe.op_context = cqe.buf = (void *) -1;
	cqe.flags = cqe.len = cqe.data = cqe.tag = UINT_MAX;
	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_WRITE, 0);

	w[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	min = ((int32_t)SOURCE_DATA < (int32_t)TARGET_DATA) ?
		SOURCE_DATA : TARGET_DATA;
	min = (min & U32_MASK) | (TARGET_DATA & (U32_MASK << 32));
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
}

Test(scalable, atomicmsg)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_msg(i);
	}
}

void sep_atomic_inject(int index)
{
	int ret, loops;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe;
	uint64_t min;

	/* i64 */
	*((int64_t *)source) = SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	sz = fi_inject_atomic(tx_ep[0][index], source, 1,
			      gni_addr[1], (uint64_t)target, mr_key[1],
			      FI_INT64, FI_MIN);
	cr_assert_eq(sz, 0);

	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
		SOURCE_DATA : TARGET_DATA;
	loops = 0;
	while (*((int64_t *)target) != min) {
		ret = fi_cq_read(tx_cq[0][index], &cqe, 1); /* for progress */
		cr_assert(ret == -FI_EAGAIN,
			  "Received unexpected event\n");

		pthread_yield();
		cr_assert(++loops < 10000, "Data mismatch");
	}
}

Test(scalable, atomicinj)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_inject(i);
	}
}

#define FETCH_SOURCE_DATA	0xACEDACEDULL
void sep_atomic_rw(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t operand = SOURCE_DATA;
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_fetch_atomic(tx_ep[0][index], &operand, 1, NULL, source,
			     loc_mr[0], rx_addr[index], (uint64_t)target,
			     mr_key[1], FI_UINT64, FI_SUM, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);

	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	ret = *((uint64_t *)target) == (SOURCE_DATA + TARGET_DATA);
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");
}

Test(scalable, atomicrw)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_rw(i);
	}
}

void sep_atomic_rwv(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t min;
	uint64_t operand = SOURCE_DATA;
	struct fi_ioc iov, r_iov;
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	iov.count = 1;
	r_iov.count = 1;

	/* i64 */
	*((int64_t *)source) = FETCH_SOURCE_DATA;
	*((int64_t *)target) = TARGET_DATA;
	iov.addr = &operand;
	r_iov.addr = source;
	sz = fi_fetch_atomicv(tx_ep[0][index], &iov, NULL, 1,
			      &r_iov, (void **)loc_mr, 1,
			      gni_addr[1], (uint64_t)target, mr_key[1],
			      FI_INT64, FI_MIN, target);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);

	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
		SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	ret = *((int64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");
}

Test(scalable, atomicrwv)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_rw(i);
	}
}

void sep_atomic_rwmsg(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t min;
	uint64_t operand = SOURCE_DATA;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov, res_iov;
	struct fi_rma_ioc rma_iov;
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	msg_iov.count = 1;
	msg.msg_iov = &msg_iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	rma_iov.addr = (uint64_t)target;
	rma_iov.count = 1;
	rma_iov.key = mr_key[1];
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
	sz = fi_fetch_atomicmsg(tx_ep[0][index], &msg, &res_iov,
				(void **)loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);

	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	min = ((int64_t)SOURCE_DATA < (int64_t)TARGET_DATA) ?
		SOURCE_DATA : TARGET_DATA;
	ret = *((int64_t *)target) == min;
	cr_assert(ret, "Data mismatch");
	ret = *((int64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");
}

Test(scalable, atomicrwmsg)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_rwmsg(i);
	}
}

void sep_atomic_compwrite(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t operand = SOURCE_DATA, op2 = TARGET_DATA;
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	/* u64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	sz = fi_compare_atomic(tx_ep[0][index], &operand, 1, NULL, &op2, NULL,
			       source, loc_mr[0], rx_addr[index],
			       (uint64_t)target, mr_key[1], FI_UINT64,
			       FI_CSWAP, target);
	cr_assert_eq(sz, 0, "fi_compare_atomic returned %ld (%s)", sz,
		     fi_strerror(-sz));

	/* reset cqe */
	cqe.op_context = cqe.buf = (void *) -1;
	cqe.flags = cqe.len = cqe.data = cqe.tag = UINT_MAX;
	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);

	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");
}

Test(scalable, atomiccompwrite)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_compwrite(i);
	}
}

#define SOURCE_DATA_FP	0.83203125
#define TARGET_DATA_FP	0.83984375
void sep_atomic_compwritev(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	double operand_dp, op2_dp;
	struct fi_ioc iov, r_iov, c_iov;
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	iov.count = 1;
	r_iov.count = 1;
	c_iov.count = 1;

	/* double */
	*((double *)&operand_dp) = SOURCE_DATA_FP;
	*((double *)&op2_dp) = TARGET_DATA_FP;
	*((double *)source) = FETCH_SOURCE_DATA;
	*((double *)target) = TARGET_DATA_FP;
	iov.addr = &operand_dp;
	r_iov.addr = source;
	c_iov.addr = &op2_dp;
	sz = fi_compare_atomicv(tx_ep[0][index],
				&iov, NULL, 1,
				&c_iov, NULL, 1,
				&r_iov, (void **)loc_mr, 1,
				gni_addr[1], (uint64_t)target, mr_key[1],
				FI_DOUBLE, FI_CSWAP, target);
	cr_assert_eq(sz, 0);

	/* reset cqe */
	cqe.op_context = cqe.buf = (void *) -1;
	cqe.flags = cqe.len = cqe.data = cqe.tag = UINT_MAX;

	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	ret = *((double *)target) == (double)SOURCE_DATA_FP;
	cr_assert(ret, "Data mismatch");
	ret = *((double *)source) == (double)TARGET_DATA_FP;
	cr_assert(ret, "Fetch data mismatch");
}

Test(scalable, atomiccompwritev)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_compwritev(i);
	}
}

void sep_atomic_compwritemsg(int index)
{
	int ret;
	ssize_t sz;
	struct fi_cq_tagged_entry cqe = { (void *) -1, UINT_MAX, UINT_MAX,
					  (void *) -1, UINT_MAX, UINT_MAX };
	uint64_t operand = SOURCE_DATA, op2 = TARGET_DATA;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov, res_iov, cmp_iov;
	struct fi_rma_ioc rma_iov;
	uint64_t w[NUMEPS] = {0}, r[NUMEPS] = {0}, w_e[NUMEPS] = {0};
	uint64_t r_e[NUMEPS] = {0};

	msg_iov.count = 1;
	msg.msg_iov = &msg_iov;
	msg.desc = (void **)loc_mr;
	msg.iov_count = 1;
	msg.addr = gni_addr[1];
	rma_iov.addr = (uint64_t)target;
	rma_iov.count = 1;
	rma_iov.key = mr_key[1];
	msg.rma_iov = &rma_iov;
	msg.context = target;
	msg.op = FI_CSWAP;

	res_iov.count = 1;
	cmp_iov.count = 1;

	/* i64 */
	*((uint64_t *)source) = FETCH_SOURCE_DATA;
	*((uint64_t *)target) = TARGET_DATA;
	msg_iov.addr = &operand;
	msg.datatype = FI_INT64;
	res_iov.addr = source;
	cmp_iov.addr = &op2;
	sz = fi_compare_atomicmsg(tx_ep[0][index], &msg, &cmp_iov, NULL, 1,
				  &res_iov, (void **)loc_mr, 1, 0);
	cr_assert_eq(sz, 0);

	/* reset cqe */
	cqe.op_context = cqe.buf = (void *) -1;
	cqe.flags = cqe.len = cqe.data = cqe.tag = UINT_MAX;
	while ((ret = fi_cq_read(tx_cq[0][index], &cqe, 1)) == -FI_EAGAIN) {
		pthread_yield();
	}

	cr_assert_eq(ret, 1);
	sep_check_tcqe(&cqe, target, FI_ATOMIC | FI_READ, 0);
	r[0] = 1;
	sep_check_cntrs(w, r, w_e, r_e);
	ret = *((uint64_t *)target) == SOURCE_DATA;
	cr_assert(ret, "Data mismatch");
	ret = *((uint64_t *)source) == TARGET_DATA;
	cr_assert(ret, "Fetch data mismatch");
}

Test(scalable, atomiccompwritemsg)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		sep_atomic_compwritemsg(i);
	}
}

void sep_invalid_compare_atomic(enum fi_datatype dt, enum fi_op op)
{
	ssize_t sz;
	size_t count;
	uint64_t operand, op2;

	if (!supported_compare_atomic_ops[op][dt]) {
		sz = fi_compare_atomic(tx_ep[0][0], &operand, 1, NULL,
				       &op2, NULL, source, loc_mr,
				       rx_addr[0], (uint64_t)target, mr_key[1],
				       dt, op, target);
		cr_assert(sz == -FI_ENOENT);

		sz = fi_compare_atomicvalid(tx_ep[0][0], dt, op, &count);
		cr_assert(sz == -FI_ENOENT, "fi_atomicvalid() succeeded\n");
	} else {
		sz = fi_compare_atomicvalid(tx_ep[0][0], dt, op, &count);
		cr_assert(!sz, "fi_atomicvalid() failed\n");
		cr_assert(count == 1, "fi_atomicvalid(): bad count\n");
	}
}

Test(scalable, atomic_invalid_compare)
{
	int i, j;

	for (i = 0; i < FI_ATOMIC_OP_LAST; i++) {
		for (j = 0; j < FI_DATATYPE_LAST; j++) {
			sep_invalid_compare_atomic(j, i);
		}
	}
}

void sep_invalid_fetch_atomic(enum fi_datatype dt, enum fi_op op)
{
	ssize_t sz;
	size_t count;
	uint64_t operand;

	if (!supported_fetch_atomic_ops[op][dt]) {
		sz = fi_fetch_atomic(tx_ep[0][0], &operand, 1, NULL,
				     source, loc_mr[0],
				     rx_addr[0], (uint64_t)target, mr_key[1],
				     dt, op, target);
		cr_assert(sz == -FI_ENOENT);

		sz = fi_fetch_atomicvalid(tx_ep[0][0], dt, op, &count);
		cr_assert(sz == -FI_ENOENT, "fi_atomicvalid() succeeded\n");
	} else {
		sz = fi_fetch_atomicvalid(tx_ep[0][0], dt, op, &count);
		cr_assert(!sz, "fi_atomicvalid() failed\n");
		cr_assert(count == 1, "fi_atomicvalid(): bad count\n");
	}
}

Test(scalable, atomic_invalid_fetch)
{
	int i, j;

	for (i = 0; i < FI_ATOMIC_OP_LAST; i++) {
		for (j = 0; j < FI_DATATYPE_LAST; j++) {
			sep_invalid_fetch_atomic(j, i);
		}
	}
}
