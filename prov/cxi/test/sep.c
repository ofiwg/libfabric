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

TestSuite(sep, .init = cxit_setup_ep, .fini = cxit_teardown_ep);

/* Test basic SEP creation */
Test(sep, simple)
{
	cxit_create_sep();
	cxit_dump_attr(NULL);
	cxit_destroy_sep();
}

/* Test invalid (NULL argument) syntax */
Test(sep, null_args)
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

struct fid_ep *tx_ep[CXIP_EP_MAX_TX_CNT] = {};
struct fid_ep *rx_ep[CXIP_EP_MAX_RX_CNT] = {};
struct fid_cq *tx_cq[CXIP_EP_MAX_TX_CNT] = {};
struct fid_cq *rx_cq[CXIP_EP_MAX_RX_CNT] = {};
fi_addr_t rx_addr[CXIP_EP_MAX_RX_CNT] = {};
int num_tx_ctx = 2;
int num_rx_ctx = 2;

void cxit_setup_sep_msg(void)
{
	struct cxip_av *av;
	fi_addr_t dest_addr;
	int ret;
	int i, j;

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

	cxit_av_attr.rx_ctx_bits = CXIP_EP_MAX_CTX_BITS;
	cxit_create_av();	// cxit_av

	av = container_of(cxit_av, struct cxip_av, av_fid);
	cr_assert_eq(av->rx_ctx_bits, CXIP_EP_MAX_CTX_BITS,
		     "av->rx_ctx_bits = %d\n", av->rx_ctx_bits);

	/* Bind the AV */
	ret = fi_ep_bind(cxit_sep, &cxit_av->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "bad retval = %d\n", ret);

	ret = fi_av_insert(cxit_av, cxit_fi->src_addr, 1, &dest_addr, 0, NULL);
	cr_assert_eq(ret, 1, "bad retval = %d\n", ret);

	/* Create TX contexts */
	for (i = 0; i < num_tx_ctx; i++) {
		ret = fi_tx_context(cxit_sep, i, NULL, &tx_ep[i],
				    (void *)&tx_ep[i]);
		cr_assert_eq(ret, FI_SUCCESS, "bad TX[%d] = %d\n",
			     i, ret);
		cr_assert_not_null(tx_ep[i]);

		for (j = 0; j < i-1; j++)
			cr_assert_neq(tx_ep[j], tx_ep[i]);
	}

	/* Create RX contexts */
	for (i = 0; i < num_rx_ctx; i++) {
		ret = fi_rx_context(cxit_sep, i, NULL, &rx_ep[i],
				    (void *)&rx_ep[i]);
		cr_assert_eq(ret, FI_SUCCESS, "bad RX[%d] = %d\n",
			     i, ret);
		cr_assert_not_null(rx_ep[i]);

		for (j = 0; j < i-1; j++)
			cr_assert_neq(rx_ep[j], rx_ep[i]);

		rx_addr[i] = fi_rx_addr(dest_addr, i, av->rx_ctx_bits);
	}

	/* Create CQ objects, one per TX and RX */
	cxit_tx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cxit_rx_cq_attr.format = FI_CQ_FORMAT_TAGGED;
	for (i = 0; i < num_tx_ctx; i++) {
		ret = fi_cq_open(cxit_domain, &cxit_tx_cq_attr, &tx_cq[i],
				 NULL);
		cr_assert_eq(ret, FI_SUCCESS, "bad TXCQ[%d] = %d\n",
			     i, ret);
	}
	for (i = 0; i < num_rx_ctx; i++) {
		ret = fi_cq_open(cxit_domain, &cxit_rx_cq_attr, &rx_cq[i],
				 NULL);
		cr_assert_eq(ret, FI_SUCCESS, "bad RXCQ[%d] = %d\n",
			     i, ret);
	}

	/* Bind contexts to CQs */
	for (i = 0; i < num_tx_ctx; i++) {
		ret = fi_scalable_ep_bind(tx_ep[i], &tx_cq[i]->fid,
					  FI_SEND);
		cr_assert_eq(ret, FI_SUCCESS, "bad TX bind[%d] = %d\n",
			     i, ret);
	}
	for (i = 0; i < num_rx_ctx; i++) {
		ret = fi_scalable_ep_bind(rx_ep[i], &rx_cq[i]->fid,
					  FI_RECV);
		cr_assert_eq(ret, FI_SUCCESS, "bad RX bind[%d] = %d\n",
			     i, ret);
	}

	/* Enable the SEP */
	ret = fi_enable(cxit_sep);
	cr_assert(ret == FI_SUCCESS);

	/* Enable the contexts */
	for (i = 0; i < num_tx_ctx; i++) {
		ret = fi_enable(tx_ep[i]);
		cr_assert(ret == FI_SUCCESS, "bad TX enable[%d] = %d\n",
			  i, ret);
	}
	for (i = 0; i < num_rx_ctx; i++) {
		ret = fi_enable(rx_ep[i]);
		cr_assert(ret == FI_SUCCESS, "bad RX enable[%d] = %d\n",
			  i, ret);
	}
}

void cxit_teardown_sep_msg(void)
{
	int ret;
	int i;

	for (i = 0; i < num_tx_ctx; i++) {
		ret = fi_close(&tx_ep[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad TX close[%d] = %d\n",
			  i, ret);
		ret = fi_close(&tx_cq[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad TXCQ close[%d] = %d\n",
			  i, ret);
	}
	for (i = 0; i < num_rx_ctx; i++) {
		ret = fi_close(&rx_ep[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad RX close[%d] = %d\n",
			  i, ret);
		ret = fi_close(&rx_cq[i]->fid);
		cr_assert(ret == FI_SUCCESS, "bad RXCQ close[%d] = %d\n",
			  i, ret);
	}

	cxit_destroy_sep();
	cxit_destroy_av();
}

/* Test basic SEP send/recv */
Test(sep, ping, .timeout = 3)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	int txi = 1;
	int rxi = 1;

	cxit_setup_sep_msg();

	recv_buf = calloc(recv_len, 1);
	cr_assert(recv_buf);

	send_buf = malloc(send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	ret = fi_trecv(rx_ep[rxi], recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Send 64 bytes to FI address 0 (self) */
	ret = fi_tsend(tx_ep[txi], send_buf, send_len, NULL, rx_addr[rxi], 0,
		       NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_read(rx_cq[rxi], &rx_cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_TAGGED | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");

	/* Wait for async event indicating data has been sent */
	do {
		ret = fi_cq_read(tx_cq[txi], &tx_cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_TAGGED | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);

	cxit_teardown_sep_msg();
}


