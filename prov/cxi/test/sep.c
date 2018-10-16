/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <limits.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include "cxip.h"
#include "cxip_test_common.h"

/* NOTE: There is a memory mapping problem with unaligned memory, cause and
 * solution currently unknown. The workaround is to use page-aligned memory.
 * The advancing_amo_fetch test is a reproducer for the problem.
 */
// TODO: remove workaround when problem is fixed.
#define	WORKAROUND
#ifdef	WORKAROUND
static inline void *CALLOC(size_t size)
{
	void *mem;

	mem = aligned_alloc(C_PAGE_SIZE, size);
	if (mem)
		memset(mem, 0, size);
	return mem;
}
#else
static inline void *CALLOC(size_t size)
{
	return calloc(1, size);
}
#endif

TestSuite(sep, .init = cxit_setup_ep, .fini = cxit_teardown_ep);

/* Test basic SEP creation */
Test(sep, simple)
{
	cxit_create_sep();
	cxit_dump_attr(NULL);
	cxit_destroy_sep();
}

/* Test invalid syntax */
Test(sep, invalid_args)
{
	struct fi_info info = *cxit_fi;
	int ret;

	ret = fi_scalable_ep(cxit_domain, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = fi_scalable_ep(cxit_domain, &info, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = fi_scalable_ep(cxit_domain, NULL, &cxit_ep, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	info.ep_attr = NULL;

	ret = fi_scalable_ep(cxit_domain, &info, &cxit_ep, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	/* Currently don't support scalable endpoints doing tagged sends */
	info = *cxit_fi;
	info.caps |= (FI_TAGGED | FI_SEND);
	ret = fi_scalable_ep(cxit_domain, &info, &cxit_ep, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
}

/**
 * Test all of the endpoint types for scalable endpoints.
 *
 */
struct sep_test_params {
	void *context;
	enum fi_ep_type type;
	int retval;
};

static struct sep_test_params sep_params[] = {
	{.type = FI_EP_UNSPEC,
		.retval = -FI_ENOPROTOOPT},
	{.type = FI_EP_MSG,
		.retval = -FI_ENOPROTOOPT},
	{.type = FI_EP_DGRAM,
		.retval = -FI_ENOPROTOOPT},
	{.type = FI_EP_SOCK_STREAM,
		.retval = -FI_ENOPROTOOPT},
	{.type = FI_EP_SOCK_DGRAM,
		.retval = -FI_ENOPROTOOPT},
	{.type = FI_EP_RDM,
		.retval = FI_SUCCESS},
	{.type = FI_EP_RDM,
		.context = (void *)0xabcdef,
		.retval = FI_SUCCESS},
};

ParameterizedTestParameters(sep, fi_sep_types)
{
	size_t param_sz;

	param_sz = ARRAY_SIZE(sep_params);
	return cr_make_param_array(struct sep_test_params, sep_params,
				   param_sz);
}

ParameterizedTest(struct sep_test_params *param, sep, fi_sep_types)
{
	int ret;
	struct cxip_ep *cep;

	cxit_fi->ep_attr->type = param->type;
	cxit_fi->caps &= ~(FI_TAGGED | FI_SEND);
	cxit_ep = NULL;
	ret = fi_scalable_ep(cxit_domain, cxit_fi, &cxit_ep, param->context);
	cr_assert_eq(ret, param->retval,
		     "fi_endpoint() error for type %d. %d != %d",
		     param->type, ret, param->retval);

	if (ret != FI_SUCCESS)
		return;

	cr_assert_not_null(cxit_ep);
	cr_expect_eq(cxit_ep->fid.fclass, FI_CLASS_SEP);
	cr_expect_eq(cxit_ep->fid.context, param->context);
	cep = container_of(cxit_ep, struct cxip_ep, ep);
	cr_expect_not_null(cep->ep_obj);

	cxit_destroy_ep();
}

Test(sep, bind_null_obj)
{
	int ret;

	cxit_create_sep();

	/* Bind a NULL object */
	ret = fi_scalable_ep_bind(cxit_sep, NULL, 0);
	cr_assert_eq(ret, -FI_EINVAL);

	cxit_destroy_sep();
}

Test(sep, ctx_null_args)
{
	int ret;

	cxit_create_sep();

	ret = fi_tx_context(cxit_sep, 0, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = fi_rx_context(cxit_sep, 0, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	cxit_destroy_sep();
}

Test(sep, ctx_inval_idx)
{
	struct fid_ep *fid = NULL;
	int index;
	int ret;

	cxit_create_sep();

	/* Induce index out of range error */
	index = CXIP_EP_MAX_TX_CNT;
	ret = fi_tx_context(cxit_sep, index, NULL, &fid, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_tx_context bad idx. %d", ret);

	index = CXIP_EP_MAX_RX_CNT;
	ret = fi_rx_context(cxit_sep, index, NULL, &fid, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_rx_context bad idx. %d", ret);

	/* Index is signed, check negative values */
	index = -1;
	ret = fi_tx_context(cxit_sep, index, NULL, &fid, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_tx_context neg idx. %d", ret);

	index = -1;
	ret = fi_rx_context(cxit_sep, index, NULL, &fid, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_rx_context neg idx. %d", ret);

	cxit_destroy_sep();
}

Test(sep, ctx_tx)
{
	int ret;
	struct cxip_ep *cxi_ep;
	struct cxip_tx_ctx *tx_ctx;
	struct fid_ep *tx_ep = NULL;
	struct fid_ep *tx_ep2 = NULL;
	void *context = &ret;
	struct fi_tx_attr *attr = NULL;
	int idx, i;

	cxit_create_sep();

	cxi_ep = container_of(cxit_sep, struct cxip_ep, ep.fid);

	idx = 1;
	ret = fi_tx_context(cxit_sep, idx, attr, &tx_ep, context);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tx_context bad tx. %d", ret);
	cr_assert_not_null(tx_ep);

	/* Should fail an attempt to reuse the index */
	ret = fi_tx_context(cxit_sep, idx, attr, &tx_ep2, context);
	cr_assert_eq(ret, -FI_EADDRINUSE, "fi_tx_context bad tx. %d", ret);
	cr_assert_null(tx_ep2);

	/* Validate TX ctx */
	tx_ctx = container_of(tx_ep, struct cxip_tx_ctx, fid.ctx);
	cr_assert_eq(tx_ctx->ep_obj, cxi_ep->ep_obj);
	cr_assert_eq(tx_ctx->domain, cxi_ep->ep_obj->domain);
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->ep_obj->num_tx_ctx), 1);
	cr_assert_eq(tx_ctx->fid.ctx.fid.fclass, FI_CLASS_TX_CTX);
	cr_assert_eq(tx_ctx->fclass, FI_CLASS_TX_CTX);
	cr_assert_eq(tx_ctx->fid.ctx.fid.context, context);

	/* Make sure this went where we wanted it */
	cr_assert_not_null(tx_ctx->ep_obj->tx_array);
	cr_assert_null(tx_ctx->ep_obj->tx_ctx);
	for (i = 0; i < tx_ctx->ep_obj->ep_attr.tx_ctx_cnt; i++) {
		struct cxip_tx_ctx *ctx = tx_ctx->ep_obj->tx_array[i];
		struct cxip_tx_ctx *exp = (i == idx) ? tx_ctx : NULL;

		cr_assert_eq(ctx, exp,
			     "mismatch on index %d, exp=%p, saw=%p\n",
			     i, exp, ctx);
	}

	/* Close should fail with FI_EBUSY */
	ret = fi_close(&cxit_sep->fid);
	cr_assert_eq(ret, -FI_EBUSY, "fi_close tx_ep. %d", ret);

	/* Close the tx ctx */
	ret = fi_close(&tx_ep->fid);
	cr_assert_eq(ret, FI_SUCCESS, "fi_close tx_ep. %d", ret);

	cxit_destroy_sep();
}

Test(sep, ctx_rx)
{
	int ret;
	struct cxip_ep *cxi_ep;
	struct cxip_rx_ctx *rx_ctx;
	struct fid_ep *rx_ep = NULL;
	struct fid_ep *rx_ep2 = NULL;
	void *context = &ret;
	struct fi_rx_attr *attr = NULL;
	int idx, i;

	cxit_create_sep();

	cxi_ep = container_of(cxit_sep, struct cxip_ep, ep.fid);

	idx = 2;
	ret = fi_rx_context(cxit_sep, idx, attr, &rx_ep, context);
	cr_assert_eq(ret, FI_SUCCESS, "fi_rx_context bad rx. %d", ret);
	cr_assert_not_null(rx_ep);

	/* Should fail an attempt to reuse the index */
	ret = fi_rx_context(cxit_sep, idx, attr, &rx_ep2, context);
	cr_assert_eq(ret, -FI_EADDRINUSE, "fi_rx_context bad tx. %d", ret);
	cr_assert_null(rx_ep2);

	/* Validate RX ctx */
	rx_ctx = container_of(rx_ep, struct cxip_rx_ctx, ctx);
	cr_assert_eq(rx_ctx->ep_obj, cxi_ep->ep_obj);
	cr_assert_eq(rx_ctx->domain, cxi_ep->ep_obj->domain);
	cr_assert_eq(rx_ctx->min_multi_recv, cxi_ep->ep_obj->min_multi_recv);
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->ep_obj->num_rx_ctx), 1);
	cr_assert_eq(rx_ctx->ctx.fid.fclass, FI_CLASS_RX_CTX);
	cr_assert_eq(rx_ctx->ctx.fid.context, context);

	/* Make sure this went where we wanted it */
	cr_assert_not_null(rx_ctx->ep_obj->rx_array);
	cr_assert_null(rx_ctx->ep_obj->rx_ctx);
	for (i = 0; i < rx_ctx->ep_obj->ep_attr.rx_ctx_cnt; i++) {
		struct cxip_rx_ctx *ctx = rx_ctx->ep_obj->rx_array[i];
		struct cxip_rx_ctx *exp = (i == idx) ? rx_ctx : NULL;

		cr_assert_eq(ctx, exp,
			     "mismatch on index %d, exp=%p, saw=%p\n",
			     i, exp, ctx);
	}

	/* Close should fail with FI_EBUSY */
	ret = fi_close(&cxit_sep->fid);
	cr_assert_eq(ret, -FI_EBUSY, "fi_close tx_ep. %d", ret);

	/* Close the rx ctx */
	ret = fi_close(&rx_ep->fid);
	cr_assert_eq(ret, FI_SUCCESS, "fi_close rx_ep. %d", ret);

	cxit_destroy_sep();
}

/* Default and maximum -- pid_granule can be < 256 */
#define	CXIP_MAX_MR_CNT	CXIP_PID_MR_CNT(256)

struct fid_ep *cxit_sep_tx[CXIP_EP_MAX_TX_CNT] = {};
struct fid_ep *cxit_sep_rx[CXIP_EP_MAX_RX_CNT] = {};
struct fid_cq *cxit_sep_tx_cq[CXIP_EP_MAX_TX_CNT] = {};
struct fid_cq *cxit_sep_rx_cq[CXIP_EP_MAX_RX_CNT] = {};
fi_addr_t cxit_sep_rx_addr[CXIP_EP_MAX_RX_CNT] = {};
uint8_t *cxit_sep_rx_buf[CXIP_EP_MAX_RX_CNT] = {};
uint8_t *cxit_sep_mr_buf[CXIP_MAX_MR_CNT] = {};
struct fid_mr *cxit_sep_mr[CXIP_MAX_MR_CNT] = {};
int cxit_sep_tx_cnt;
int cxit_sep_rx_cnt;
int cxit_sep_mr_cnt;
int cxit_sep_buf_size;

void cxit_setup_sep(int ntx, int nrx, int nmr, int buf_size)
{
	struct cxip_av *av;
	int ret;
	int i, j;
	size_t addrlen;
	fi_addr_t dest_addr;

	/* Set counts */
	cxit_sep_tx_cnt = ntx;
	cxit_sep_rx_cnt = nrx;
	cxit_sep_mr_cnt = nmr;
	cxit_sep_buf_size = buf_size;

	/* Request required capabilities for RMA */
	cxit_setup_getinfo();
	cxit_fi_hints->fabric_attr->prov_name = strdup(cxip_prov_name);
	cxit_fi_hints->caps = FI_WRITE | FI_READ;
	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_av_attr.type = FI_AV_TABLE;

	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	cxit_fi_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

	cxit_setup_ep();

	/* Set up RMA objects */
	cxit_create_sep();	// cxit_sep

	/* Create and bind the AV. This must be done before enabling the SEP or
	 * any of the TX/RX contexts.
	 */
	cxit_av_attr.rx_ctx_bits = CXIP_EP_MAX_CTX_BITS;
	cxit_create_av();	// cxit_av

	av = container_of(cxit_av, struct cxip_av, av_fid);
	cr_assert_eq(av->rx_ctx_bits, CXIP_EP_MAX_CTX_BITS,
		     "av->rx_ctx_bits = %d\n", av->rx_ctx_bits);

	ret = fi_ep_bind(cxit_sep, &cxit_av->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "bad retval = %d\n", ret);

	/* Create TX contexts */
	for (i = 0; i < cxit_sep_tx_cnt; i++) {
		ret = fi_tx_context(cxit_sep, i, NULL, &cxit_sep_tx[i],
				    (void *)&cxit_sep_tx[i]);
		cr_assert_eq(ret, FI_SUCCESS, "bad TX[%d] = %d\n",
			     i, ret);
		cr_assert_not_null(cxit_sep_tx[i]);

		for (j = 0; j < i-1; j++)
			cr_assert_neq(cxit_sep_tx[j], cxit_sep_tx[i]);
	}

	/* Create RX contexts */
	for (i = 0; i < cxit_sep_rx_cnt; i++) {
		ret = fi_rx_context(cxit_sep, i, NULL, &cxit_sep_rx[i],
				    (void *)&cxit_sep_rx[i]);
		cr_assert_eq(ret, FI_SUCCESS, "bad RX[%d] = %d\n",
			     i, ret);
		cr_assert_not_null(cxit_sep_rx[i]);

		for (j = 0; j < i-1; j++)
			cr_assert_neq(cxit_sep_rx[j], cxit_sep_rx[i]);

	}

	/* Create and bind CQ objects, one per TX and RX. This must be done
	 * before enabling any of the TX/RX contexts.
	 */
	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_rx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	for (i = 0; i < cxit_sep_tx_cnt; i++) {
		ret = fi_cq_open(cxit_domain, &cxit_tx_cq_attr,
				 &cxit_sep_tx_cq[i], NULL);
		cr_assert_eq(ret, FI_SUCCESS, "bad TXCQ[%d] = %d\n",
			     i, ret);

		ret = fi_scalable_ep_bind(cxit_sep_tx[i],
					  &cxit_sep_tx_cq[i]->fid, FI_SEND);
		cr_assert_eq(ret, FI_SUCCESS, "bad TX bind[%d] = %d\n",
			     i, ret);
	}
	for (i = 0; i < cxit_sep_rx_cnt; i++) {
		ret = fi_cq_open(cxit_domain, &cxit_rx_cq_attr,
				 &cxit_sep_rx_cq[i], NULL);
		cr_assert_eq(ret, FI_SUCCESS, "bad RXCQ[%d] = %d\n",
			     i, ret);

		ret = fi_scalable_ep_bind(cxit_sep_rx[i],
					  &cxit_sep_rx_cq[i]->fid, FI_RECV);
		cr_assert_eq(ret, FI_SUCCESS, "bad RX bind[%d] = %d\n",
			     i, ret);
	}

	/* Enable the contexts */
	for (i = 0; i < cxit_sep_tx_cnt; i++) {
		ret = fi_enable(cxit_sep_tx[i]);
		cr_assert(ret == FI_SUCCESS, "bad TX enable[%d] = %d\n",
			  i, ret);
	}
	for (i = 0; i < cxit_sep_rx_cnt; i++) {
		ret = fi_enable(cxit_sep_rx[i]);
		cr_assert(ret == FI_SUCCESS, "bad RX enable[%d] = %d\n",
			  i, ret);
	}

	/* Enable the SEP (should be a no-op) */
	ret = fi_enable(cxit_sep);
	cr_assert(ret == FI_SUCCESS);

	/* Find assigned Endpoint address. Address was assigned during enable.
	 */
	addrlen = sizeof(cxit_ep_addr);
	ret = fi_getname(&cxit_sep->fid, &cxit_ep_addr, &addrlen);
	cr_assert(ret == FI_SUCCESS);
	cr_assert(addrlen == sizeof(cxit_ep_addr));

	/* Insert local address into AV to prepare to send to self. This returns
	 * an index into the AV table, and since it is the first (and only)
	 * address in this test, it should return 0 as the dest_addr.
	 */
	ret = fi_av_insert(cxit_av, (void *)&cxit_ep_addr, 1, &dest_addr, 0,
			   NULL);
	cr_assert(ret == 1);
	cr_assert((uint64_t)dest_addr == 0);

	/* Create an array of extended RX fi_addr values for the application to
	 * use, one for each target RX.
	 */
	for (i = 0; i < cxit_sep_rx_cnt; i++) {
		cxit_sep_rx_addr[i] = fi_rx_addr(dest_addr, i, av->rx_ctx_bits);
	}

	/* Post a single buffer to each RX */
	for (i = 0; i < cxit_sep_rx_cnt; i++) {
		cxit_sep_rx_buf[i] = CALLOC(cxit_sep_buf_size);
		ret = fi_trecv(cxit_sep_rx[i], cxit_sep_rx_buf[i],
			       cxit_sep_buf_size, NULL,
			       FI_ADDR_UNSPEC, 0, 0, NULL);
		cr_assert(ret == FI_SUCCESS, "bad RX post[%d] = %d\n",
			  i, ret);
	}

	/* Create MRs */
	for (i = 0; i < cxit_sep_mr_cnt; i++) {
		cxit_sep_mr_buf[i] = CALLOC(cxit_sep_buf_size);
		cr_assert_not_null(cxit_sep_mr_buf[i]);

		ret = fi_mr_reg(cxit_domain, cxit_sep_mr_buf[i],
				cxit_sep_buf_size,
				FI_REMOTE_READ|FI_REMOTE_WRITE,
				0, i, 0, &cxit_sep_mr[i], NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg[%d]=%d", i, ret);

		ret = fi_mr_bind(cxit_sep_mr[i], &cxit_sep->fid, 0);
		cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind[%d]=%d", i, ret);

		ret = fi_mr_enable(cxit_sep_mr[i]);
		cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable[%d]=%d", i, ret);
	}
}

void cxit_teardown_sep(void)
{
	int ret;
	int i;

	for (i = 0; i < cxit_sep_mr_cnt; i++) {
		ret = fi_close(&cxit_sep_mr[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad MR close[%d] = %d\n",
			  i, ret);
		free(cxit_sep_mr_buf[i]);
	}
	for (i = 0; i < cxit_sep_tx_cnt; i++) {
		ret = fi_close(&cxit_sep_tx[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad TX close[%d] = %d\n",
			  i, ret);

		ret = fi_close(&cxit_sep_tx_cq[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad TXCQ close[%d] = %d\n",
			  i, ret);
	}
	for (i = 0; i < cxit_sep_rx_cnt; i++) {
		ret = fi_close(&cxit_sep_rx[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad RX close[%d] = %d\n",
			  i, ret);

		ret = fi_close(&cxit_sep_rx_cq[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad RXCQ close[%d] = %d\n",
			  i, ret);

		free(cxit_sep_rx_buf[i]);
	}

	cxit_destroy_sep();
	cxit_destroy_av();
}

static int _cmpdata(uint8_t *rx, uint8_t *tx, int len)
{
	int i;

	for (i = 0; i < len; i++)
		if (rx[i] != (tx ? tx[i] : 0))
			break;
	return i;
}

/* Test basic SEP send/recv */
Test(sep, simple_msg_send, .timeout = 3)
{
	int i, ret;
	int txi, rxi, rxi2;
	uint8_t *tx_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;

	cxit_setup_sep(16, 16, 0, 64);

	/* We do one send at a time, reuse TX buffer */
	tx_buf = malloc(cxit_sep_buf_size);
	cr_assert(tx_buf);

	/* O(N^2) loop over combinations of TX and RX */
	for (txi = 0; txi < cxit_sep_tx_cnt; txi++) {
		/* Unique initialization of TX for each TX */
		for (i = 0; i < cxit_sep_buf_size; i++)
			tx_buf[i] = (txi << 4) | i;

		/* Send to each RX in turn */
		for (rxi = 0; rxi < cxit_sep_rx_cnt; rxi++) {
			ret = fi_tsend(cxit_sep_tx[txi], tx_buf,
				       cxit_sep_buf_size,
				       NULL, cxit_sep_rx_addr[rxi], 0, NULL);
			cr_assert(ret == FI_SUCCESS);

			/* Wait for async event indicating data received */
			ret = cxit_await_completion(cxit_sep_rx_cq[rxi],
						    &rx_cqe);
			cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

			/* Validate RX event fields */
			cr_assert(rx_cqe.op_context == NULL,
				  "RX CQE CTX mismatch[%d][%d]", txi, rxi);
			cr_assert(rx_cqe.flags == (FI_TAGGED | FI_RECV),
				  "RX CQE flags mismatch[%d][%d]", txi, rxi);
			cr_assert(rx_cqe.len == cxit_sep_buf_size,
				  "Invalid RX CQE length[%d][%d]", txi, rxi);
			cr_assert(rx_cqe.buf == 0,
				  "Invalid RX CQE addr[%d][%d]", txi, rxi);
			cr_assert(rx_cqe.data == 0,
				  "Invalid RX CQE data[%d][%d]", txi, rxi);
			cr_assert(rx_cqe.tag == 0,
				  "Invalid RX CQE tag[%d][%d]", txi, rxi);

			/* Wait for async event indicating data sent */
			ret = cxit_await_completion(cxit_sep_tx_cq[txi],
						    &tx_cqe);
			cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

			/* Validate TX event fields */
			cr_assert(tx_cqe.op_context == NULL,
				  "TX CQE CTX mismatch[%d][%d]", txi, rxi);
			cr_assert(tx_cqe.flags == (FI_TAGGED | FI_SEND),
				  "TX CQE flags mismatch[%d][%d]", txi, rxi);
			cr_assert(tx_cqe.len == 0,
				  "Invalid TX CQE length[%d][%d]", txi, rxi);
			cr_assert(tx_cqe.buf == 0,
				  "Invalid TX CQE addr[%d][%d]", txi, rxi);
			cr_assert(tx_cqe.data == 0,
				  "Invalid TX CQE data[%d][%d]", txi, rxi);
			cr_assert(tx_cqe.tag == 0,
				  "Invalid TX CQE tag[%d][%d]", txi, rxi);

			/* Test all of the receive buffers */
			for (rxi2 = 0; rxi2 < cxit_sep_rx_cnt; rxi2++) {
				ret = _cmpdata(cxit_sep_rx_buf[rxi2],
					       (rxi2 == rxi) ? tx_buf : NULL,
					       cxit_sep_buf_size);
				cr_assert_eq(ret, cxit_sep_buf_size,
					     "Byte compare[%d][%d|%d][%d] exp=%02x, saw=%02x",
					     txi, rxi, rxi2, ret,
					     (rxi2 == rxi) ? tx_buf[ret] : 0,
					     cxit_sep_rx_buf[rxi2][ret]);
			}

			/* Clear buffer  */
			memset(cxit_sep_rx_buf[rxi], 0, cxit_sep_buf_size);

			/* Post buffer again, except on the last TX pass. This
			 * is a workaround for the lack of fi_cancel()
			 * implmentation: if we post this now, there's no way to
			 * cleanly shut down.
			 * TODO: remove after fi_cancel() implemented
			 */
			if (txi == cxit_sep_tx_cnt-1)
				continue;

			ret = fi_trecv(cxit_sep_rx[rxi], cxit_sep_rx_buf[rxi],
				       cxit_sep_buf_size,
				       NULL, FI_ADDR_UNSPEC, 0, 0, NULL);
			cr_assert(ret == FI_SUCCESS,
				  "bad RX post[%d][%d] = %d\n",
				  txi, rxi, ret);
		}
	}

	free(tx_buf);

	cxit_teardown_sep();
}

Test(sep, simple_rma_write, .timeout = 10)
{
	struct fi_cq_tagged_entry cqe;
	uint8_t *tx_buf;
	int txi, mri;
	int i, ret;

	cxit_setup_sep(16, 0, 16, 64);

	/* We do one send at a time, reuse TX buffer */
	tx_buf = malloc(cxit_sep_buf_size);
	cr_assert(tx_buf);

	/* Iterate over TX contexts */
	for (txi = 0; txi < cxit_sep_tx_cnt; txi++) {
		/* Unique initialization of TX for each TX */
		for (i = 0; i < cxit_sep_buf_size; i++)
			tx_buf[i] = (txi << 4) | i;

		ret = fi_write(cxit_sep_tx[txi], tx_buf, cxit_sep_buf_size,
			       NULL, cxit_sep_rx_addr[0], 0, txi, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code  = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		/* Test all of the remote memory buffers */
		for (mri = 0; mri < cxit_sep_mr_cnt; mri++) {
			ret = _cmpdata(cxit_sep_mr_buf[mri],
				       (mri == txi) ? tx_buf : NULL,
				       cxit_sep_buf_size);
			cr_assert_eq(ret, cxit_sep_buf_size,
				     "Byte compare[%d|%d][%d] exp=%02x, saw=%02x",
				     txi, mri, ret,
				     (mri == txi) ? tx_buf[ret] : 0,
				     cxit_sep_mr_buf[mri][ret]);
		}

		/* Clear the remote buffer */
		memset(cxit_sep_mr_buf[txi], 0, cxit_sep_buf_size);
	}

	cxit_teardown_sep();
}

Test(sep, simple_rma_read, .timeout = 10)
{
	struct fi_cq_tagged_entry cqe;
	uint8_t *tx_buf;
	int txi;
	int i, ret;

	cxit_setup_sep(16, 0, 16, 64);

	/* We do one read at a time, reuse TX buffer */
	tx_buf = malloc(cxit_sep_buf_size);
	cr_assert(tx_buf);

	/* Iterate over TX contexts */
	for (txi = 0; txi < cxit_sep_tx_cnt; txi++) {
		/* Initialize the remote memory */
		for (i = 0; i < cxit_sep_buf_size; i++)
			cxit_sep_mr_buf[txi][i] = (txi << 4) | i;
		memset(tx_buf, 0, cxit_sep_buf_size);

		ret = fi_read(cxit_sep_tx[txi], tx_buf, cxit_sep_buf_size,
			      NULL, cxit_sep_rx_addr[0], 0, txi, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code  = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

		/* Test the read value */
		ret = _cmpdata(cxit_sep_mr_buf[txi], tx_buf,
			       cxit_sep_buf_size);
		cr_assert_eq(ret, cxit_sep_buf_size,
			     "Byte compare[%d][%d] exp=%02x, saw=%02x",
			     txi, ret, tx_buf[ret], cxit_sep_mr_buf[txi][ret]);

		/* Clear the remote buffer */
		memset(cxit_sep_mr_buf[txi], 0, cxit_sep_buf_size);
	}

	cxit_teardown_sep();
}

Test(sep, simple_amo, .timeout = 10)
{
	struct fi_cq_tagged_entry cqe;
	uint64_t operand1;
	uint64_t exp_remote;
	uint64_t *rma;
	int txi;
	int ret;

	cxit_setup_sep(16, 0, 16, 64);

	/* Iterate over TX contexts */
	for (txi = 0; txi < cxit_sep_tx_cnt; txi++) {
		exp_remote = 0;
		rma = (uint64_t *)cxit_sep_mr_buf[txi];

		operand1 = 1;
		exp_remote += operand1;
		ret = fi_atomic(cxit_sep_tx[txi], &operand1, 1, NULL,
				cxit_sep_rx_addr[0], 0, txi,
				FI_UINT64, FI_SUM, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code  = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
		cr_assert_eq(*rma, exp_remote,
			     "Remote Result[%2d] = %016lx, expected = %016lx",
			     txi, *rma, exp_remote);

		operand1 = 3;
		exp_remote += operand1;
		ret = fi_atomic(cxit_sep_tx[txi], &operand1, 1, NULL,
				cxit_sep_rx_addr[0], 0, txi,
				FI_UINT64, FI_SUM, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
		cr_assert_eq(*rma, exp_remote,
			     "Remote Result[%2d] = %016lx, expected = %016lx",
			     txi, *rma, exp_remote);

		operand1 = 9;
		exp_remote += operand1;
		ret = fi_atomic(cxit_sep_tx[txi], &operand1, 1, NULL,
				cxit_sep_rx_addr[0], 0, txi,
				FI_UINT64, FI_SUM, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
		cr_assert_eq(*rma, exp_remote,
			     "Remote Result[%2d] = %016lx, expected = %016lx",
			     txi, *rma, exp_remote);
	}

	cxit_teardown_sep();
}

Test(sep, simple_amo_fetch, .timeout = 10)
{
	struct fi_cq_tagged_entry cqe;
	uint64_t operand1;
	uint64_t exp_remote = 0;
	uint64_t exp_result = 0;
	uint64_t *rma;
	uint64_t *loc;
	int txi;
	int ret;

	cxit_setup_sep(16, 0, 16, 64);

	loc = CALLOC(cxit_sep_buf_size);
	cr_assert_not_null(loc);

	/* Iterate over TX contexts */
	for (txi = 0; txi < cxit_sep_tx_cnt; txi++) {
		exp_remote = 0;
		exp_result = 0;
		rma = (uint64_t *)cxit_sep_mr_buf[txi];

		operand1 = 1;
		*loc = -1;
		exp_result = exp_remote;
		exp_remote += operand1;
		ret = fi_fetch_atomic(cxit_sep_tx[txi], &operand1, 1, NULL,
				      loc, 0,
				      cxit_sep_rx_addr[0], 0, txi,
				      FI_UINT64, FI_SUM, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
		cr_assert_eq(*rma, exp_remote,
			     "Remote Result[%2d] = %016lx, expected = %016lx",
			     txi, *rma, exp_remote);
		cr_assert_eq(*loc, exp_result,
			     "Fetch  Result[%2d] = %016lx, expected = %016lx",
			     txi, *loc, exp_result);

		operand1 = 3;
		*loc = -1;
		exp_result = exp_remote;
		exp_remote += operand1;
		ret = fi_fetch_atomic(cxit_sep_tx[txi], &operand1, 1, NULL,
				      loc, 0,
				      cxit_sep_rx_addr[0], 0, txi,
				      FI_UINT64, FI_SUM, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
		cr_assert_eq(*rma, exp_remote,
			     "Remote Result[%2d] = %016lx, expected = %016lx",
			     txi, *rma, exp_remote);
		cr_assert_eq(*loc, exp_result,
			     "Fetch  Result[%2d] = %016lx, expected = %016lx",
			     txi, *loc, exp_result);

		operand1 = 9;
		*loc = -1;
		exp_result = exp_remote;
		exp_remote += operand1;
		ret = fi_fetch_atomic(cxit_sep_tx[txi], &operand1, 1, NULL,
				      loc, 0,
				      cxit_sep_rx_addr[0], 0, txi,
				      FI_UINT64, FI_SUM, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
		cr_assert_eq(*rma, exp_remote,
			     "Remote Result[%2d] = %016lx, expected = %016lx",
			     txi, *rma, exp_remote);
		cr_assert_eq(*loc, exp_result,
			     "Fetch  Result[%2d] = %016lx, expected = %016lx",
			     txi, *loc, exp_result);
	}

	free(loc);

	cxit_teardown_sep();
}

Test(sep, simple_amo_swap, .timeout = 10)
{
	struct fi_cq_tagged_entry cqe;
	uint64_t operand1;
	uint64_t compare;
	uint64_t exp_remote = 0;
	uint64_t exp_result = 0;
	uint64_t *rma;
	uint64_t *loc;
	int txi;
	int ret;

	cxit_setup_sep(16, 0, 16, 64);

	loc = CALLOC(cxit_sep_buf_size);
	cr_assert_not_null(loc);

	for (txi = 0; txi < cxit_sep_tx_cnt; txi++) {
		exp_remote = 0;
		exp_result = 0;
		rma = (uint64_t *)cxit_sep_mr_buf[txi];

		*rma = 0;	/* remote == 0 */
		operand1 = 1;	/* change remote to 1 */
		compare = 2;	/* if remote != 2 (true) */
		*loc = -1;	/* initialize result */
		exp_remote = 1;	/* expect remote == 1 */
		exp_result = 0;	/* expect result == 0 */
		ret = fi_compare_atomic(cxit_sep_tx[txi],
					&operand1, 1, 0,
					&compare, 0,
					loc, 0,
					0, 0, txi,
					FI_UINT64, FI_CSWAP_NE, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
		cr_assert_eq(*rma, exp_remote,
			     "Remote Result[%2d] = %016lx, expected = %016lx",
			     txi, *rma, exp_remote);
		cr_assert_eq(*loc, exp_result,
			     "Fetch  Result[%2d] = %016lx, expected = %016lx",
			     txi, *loc, exp_result);

		*rma = 2;	/* remote == 2 */
		operand1 = 1;	/* change remote to 1 */
		compare = 2;	/* if remote != 2 (false) */
		*loc = -1;	/* initialize result */
		exp_remote = 2;	/* expect remote == 2 (no op) */
		exp_result = 2;	/* expect result == 2 (does return value) */
		ret = fi_compare_atomic(cxit_sep_tx[txi],
					&operand1, 1, 0,
					&compare, 0,
					loc, 0,
					0, 0, txi,
					FI_UINT64, FI_CSWAP_NE, NULL);
		cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
		ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
		cr_assert_eq(*rma, exp_remote,
			     "Remote Result[%2d] = %016lx, expected = %016lx",
			     txi, *rma, exp_remote);
		cr_assert_eq(*loc, exp_result,
			     "Fetch  Result[%2d] = %016lx, expected = %016lx",
			     txi, *loc, exp_result);
	}

	free(loc);

	cxit_teardown_sep();
}

/* Reproducer for memory problem. This uses flushed printf() to ensure that
 * information is displayed in the event of a crash.
 */
// TODO: cleanup and enable when memory problem is fixed
Test(sep, advancing_amo_fetch, .timeout = 10, .disabled = false)
{
	struct fi_cq_tagged_entry cqe;
	uint64_t operand1;
	uint64_t exp_remote = 0;
	uint64_t exp_result = 0;
	uint64_t *rma;
	uint64_t *loc;
	int buflen;
	int mri;
	int txi;
	int ret;
	int err;

#if ENABLE_DEBUG
	cr_log_info("ENABLE_DEBUG SET");
#else
	cr_log_info("ENABLE_DEBUG UNSET");
#endif
	cxit_setup_sep(16, 0, 16, 64);
// 47 increment fails
// 45 increment fails
	for (buflen = 64; buflen < 512; buflen += 1) {

		loc = CALLOC(buflen);
		cr_assert_not_null(loc);
		//printf("loc[%2d] = %p\n", 0, loc);
		//fflush(stdout);

		//printf("unmap\n"); fflush(stdout);
		for (mri = 0; mri < cxit_sep_mr_cnt; mri++) {
			ret = fi_close(&cxit_sep_mr[mri]->fid);
			cr_assert(ret == FI_SUCCESS, "bad MR close[%d] = %d\n",
				  mri, ret);
			free(cxit_sep_mr_buf[mri]);
		}

		/* Create MRs */
		//printf("map buflen = %d\n", buflen);
		//fflush(stdout);
		for (mri = 0; mri < cxit_sep_mr_cnt; mri++) {
			cxit_sep_mr_buf[mri] = CALLOC(buflen);
			cr_assert_not_null(cxit_sep_mr_buf[mri]);
			rma = (uint64_t *)cxit_sep_mr_buf[mri];
			*rma = 1;

			ret = fi_mr_reg(cxit_domain, cxit_sep_mr_buf[mri],
					buflen,
					FI_REMOTE_READ|FI_REMOTE_WRITE,
					0, mri, 0, &cxit_sep_mr[mri], NULL);
			cr_assert_eq(ret, FI_SUCCESS,
				     "fi_mr_reg[%d]=%d", mri, ret);

			ret = fi_mr_bind(cxit_sep_mr[mri], &cxit_sep->fid, 0);
			cr_assert_eq(ret, FI_SUCCESS,
				     "fi_mr_bind[%d]=%d", mri, ret);

			ret = fi_mr_enable(cxit_sep_mr[mri]);
			cr_assert_eq(ret, FI_SUCCESS,
				     "fi_mr_enable[%d]=%d", mri, ret);
			//printf("buf[%2d] = %p\n", mri, cxit_sep_mr_buf[mri]);
			//fflush(stdout);
		}

		/* Iterate over TX contexts */
		err = 0;
		for (txi = 0; txi < cxit_sep_tx_cnt; txi++) {
			rma = (uint64_t *)cxit_sep_mr_buf[txi];
			//printf("testing memory[%d] %p\n", txi, rma);
			//fflush(stdout);

			exp_remote = 1;
			operand1 = 1;
			*loc = -1;
			exp_result = exp_remote;
			exp_remote += operand1;
			ret = fi_fetch_atomic(cxit_sep_tx[txi], &operand1,
					      1, NULL, loc, 0,
					      cxit_sep_rx_addr[0], 0, txi,
					      FI_UINT64, FI_SUM, NULL);
			cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
			ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
			cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
			if (*rma != exp_remote) {
				printf(
				       "buflen=%d, txi=%2d, op#%d, remote exp=%ld, saw=%ld, rma=%p loc=%p\n",
				       buflen, txi, 1, exp_remote, *rma,
				       rma, loc);
				fflush(stdout);
				//assert(0);
			}
			if (*loc != exp_result) {
				printf(
				       "buflen=%d, txi=%2d, op#%d, local  exp=%ld, saw=%ld, rma=%p loc=%p\n",
				       buflen, txi, 1, exp_result, *loc,
				       rma, loc);
				fflush(stdout);
				//assert(0);
			}
#if 0
			cr_expect_eq(*rma, exp_remote,
				     "Remote Result[%d] = %ld, expected = %ld, err %d",
				     txi, *rma, exp_remote, ++err);
			cr_expect_eq(*loc, exp_result,
				     "Fetch Result[%d] = %016lx, expected = %016lx, err %d",
				     txi, *loc, exp_result, ++err);
#endif

			operand1 = 3;
			*loc = -1;
			exp_result = exp_remote;
			exp_remote += operand1;
			ret = fi_fetch_atomic(cxit_sep_tx[txi], &operand1,
					      1, NULL, loc, 0,
					      cxit_sep_rx_addr[0], 0, txi,
					      FI_UINT64, FI_SUM, NULL);
			cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
			ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
			cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
			if (*rma != exp_remote) {
				printf(
				       "buflen=%d, txi=%2d, op#%d, remote exp=%ld, saw=%ld, rma=%p loc=%p\n",
				       buflen, txi, 2, exp_remote, *rma,
				       rma, loc);
				fflush(stdout);
				//assert(0);
			}
			if (*loc != exp_result) {
				printf(
				       "buflen=%d, txi=%2d, op#%d, local  exp=%ld, saw=%ld, rma=%p loc=%p\n",
				       buflen, txi, 2, exp_result, *loc,
				       rma, loc);
				fflush(stdout);
				//assert(0);
			}
#if 0
			cr_expect_eq(*rma, exp_remote,
				     "Remote Result[%d] = %ld, expected = %ld, err %d",
				     txi, *rma, exp_remote, ++err);
			cr_expect_eq(*loc, exp_result,
				     "Fetch Result[%d] = %016lx, expected = %016lx, err %d",
				     txi, *loc, exp_result, ++err);
#endif

			operand1 = 9;
			*loc = -1;
			exp_result = exp_remote;
			exp_remote += operand1;
			ret = fi_fetch_atomic(cxit_sep_tx[txi], &operand1,
					      1, NULL, loc, 0,
					      cxit_sep_rx_addr[0], 0, txi,
					      FI_UINT64, FI_SUM, NULL);
			cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
			ret = cxit_await_completion(cxit_sep_tx_cq[txi], &cqe);
			cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);
			if (*rma != exp_remote) {
				printf(
				       "buflen=%d, txi=%2d, op#%d, remote exp=%ld, saw=%ld, rma=%p loc=%p\n",
				       buflen, txi, 3, exp_remote, *rma,
				       rma, loc);
				fflush(stdout);
				//assert(0);
			}
			if (*loc != exp_result) {
				printf(
				       "buflen=%d, txi=%2d, op#%d, local  exp=%ld, saw=%ld, rma=%p loc=%p\n",
				       buflen, txi, 3, exp_result, *loc,
				       rma, loc);
				fflush(stdout);
				//assert(0);
			}
#if 0
			cr_expect_eq(*rma, exp_remote,
				     "Remote Result[%d] = %ld, expected = %ld, err %d",
				     txi, *rma, exp_remote, ++err);
			cr_expect_eq(*loc, exp_result,
				     "Fetch Result[%d] = %016lx, expected = %016lx, err %d",
				     txi, *loc, exp_result, ++err);
#endif
		}
		cr_assert_eq(err, 0);

		free(loc);
	}

	cxit_teardown_sep();
}

