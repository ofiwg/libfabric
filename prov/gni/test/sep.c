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

#define BUF_SZ (1<<20)
char *target;
char *source;
struct fid_mr *rem_mr[NUMEPS], *loc_mr[NUMEPS];
uint64_t mr_key[NUMEPS];

/* from scalable_ep.c */
static int ctx_cnt = 4;
static int rx_ctx_bits;
static struct fid_ep **tx_ep[NUMEPS], **rx_ep[NUMEPS];
static struct fid_cq **txcq_array[NUMEPS];
static struct fid_cq **rxcq_array[NUMEPS];
static fi_addr_t *rx_addr;

void sep_setup(void)
{
	int ret, i, j;
	struct fi_av_attr av_attr = {0};
	size_t addrlen = 0;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_NAMED_RX_CTX;
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

		txcq_array[i] = calloc(ctx_cnt, sizeof(*txcq_array));
		rxcq_array[i] = calloc(ctx_cnt, sizeof(*rxcq_array));
		tx_ep[i] = calloc(ctx_cnt, sizeof(*tx_ep));
		rx_ep[i] = calloc(ctx_cnt, sizeof(*rx_ep));
		if (!txcq_array[i] || !txcq_array[i] ||
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

	if (!rx_addr || !target || !source) {
		cr_assert(0, "calloc");
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

		for (j = 0; j < ctx_cnt; j++) {
			ret = fi_tx_context(sep[i], j, NULL, &tx_ep[i][j],
					    NULL);
			cr_assert(!ret, "fi_tx_context");

			ret = fi_cq_open(dom[i], &cq_attr, &txcq_array[i][j],
					 NULL);
			cr_assert(!ret, "fi_cq_open");

			ret = fi_rx_context(sep[i], j, NULL, &rx_ep[i][j],
					    NULL);
			cr_assert(!ret, "fi_rx_context");

			ret = fi_cq_open(dom[i], &cq_attr, &rxcq_array[i][j],
					 NULL);
			cr_assert(!ret, "fi_cq_open");
		}

		ret = fi_scalable_ep_bind(sep[i], &av[i]->fid, 0);
		cr_assert(!ret, "fi_scalable_ep_bind");

		for (j = 0; j < ctx_cnt; j++) {
			ret = fi_ep_bind(tx_ep[i][j], &txcq_array[i][j]->fid,
					 FI_SEND);
			cr_assert(!ret, "fi_ep_bind");

			ret = fi_ep_bind(tx_ep[i][j], &rxcq_array[i][j]->fid,
					 FI_RECV);
			cr_assert(!ret, "fi_ep_bind");

			ret = fi_enable(tx_ep[i][j]);
			cr_assert(!ret, "fi_enable");

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

	/* test for inserting an ep_name that doesn't fit in the AV */
	av_attr.rx_ctx_bits = 1;
	ret = fi_av_open(dom[0], &av_attr, &t_av, NULL);
	cr_assert(!ret, "fi_av_open");
	ret = fi_av_insert(t_av, ep_name[0], 1, &gni_addr[0], 0, NULL);
	cr_assert(ret == -FI_EINVAL);
	ret = fi_close(&t_av->fid);
	cr_assert(!ret, "failure in closing av.");
}

static void sep_teardown(void)
{
	int ret, i, j;

	for (i = 0; i < NUMEPS; i++) {
		for (j = 0; j < ctx_cnt; j++) {
			ret = fi_close(&tx_ep[i][j]->fid);
			cr_assert(!ret, "failure closing tx_ep.");

			ret = fi_close(&rx_ep[i][j]->fid);
			cr_assert(!ret, "failure closing rx_ep.");

			ret = fi_close(&txcq_array[i][j]->fid);
			cr_assert(!ret, "failure closing tx cq.");

			ret = fi_close(&rxcq_array[i][j]->fid);
			cr_assert(!ret, "failure closing rx cq.");
		}

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

	fi_freeinfo(hints);
	free(target);
	free(source);

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "failure in closing fabric.");
}

void sep_init_data(char *buf, int len, char seed)
{
	int i;

	for (i = 0; i < len; i++)
		buf[i] = seed++;
}

int sep_check_data(char *buf1, char *buf2, int len)
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

static void wait_for_cqs(struct fid_cq *scq, struct fid_cq *dcq)
{
	struct fi_cq_tagged_entry cqe;
	int ret;
	int s_done = 0, d_done = 0;

	do {
		ret = fi_cq_read(scq, &cqe, 1);
		if (ret == 1) {
			s_done = 1;
		}

		ret = fi_cq_read(dcq, &cqe, 1);
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

/*******************************************************************************
 * Test MSG functions
 ******************************************************************************/

TestSuite(scalable, .init = sep_setup, .fini = sep_teardown,
	  .disabled = false);

/*
 * ssize_t fi_send(struct fid_ep *ep, void *buf, size_t len,
 *		void *desc, fi_addr_t dest_addr, void *context);
 *
 * ssize_t fi_recv(struct fid_ep *ep, void * buf, size_t len,
 *		void *desc, fi_addr_t src_addr, void *context);
 */
void sep_send_recv(int index, int len)
{
	ssize_t ret;

	sep_init_data(source, len, 0xab);
	sep_init_data(target, len, 0);

	ret = fi_send(tx_ep[0][index], source, len, loc_mr[0],
		      rx_addr[index], target);
	cr_assert(ret == 0, "fi_send failed err:%ld", ret);

	ret = fi_recv(rx_ep[1][index], target, len, rem_mr[0],
		      FI_ADDR_UNSPEC, source);
	cr_assert(ret == 0, "fi_recv failed err:%ld", ret);

	wait_for_cqs(txcq_array[0][index], rxcq_array[1][index]);

	ret = sep_check_data(source, target, 8);
	cr_assert(ret == 1, "Data check failed");
}

Test(scalable, sr)
{
	int i;

	for (i = 0; i < ctx_cnt; i++) {
		xfer_each_size(sep_send_recv, i, 1, BUF_SZ);
	}

	dbg_printf("scalable done\n");
}
