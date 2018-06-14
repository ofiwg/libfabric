/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include "cxi_prov.h"
#include "cxi_test_common.h"

TestSuite(ep, .init = cxit_setup_ep, .fini = cxit_teardown_ep);

/* Test basic EP creation */
Test(ep, simple)
{
	cxit_create_ep();

	cxit_destroy_ep();
}

/* Test NULL parameter passed with EP creation */
Test(ep, ep_null_info)
{
	int ret;

	ret = fi_endpoint(cxit_domain, NULL, &cxit_ep, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "Failure with NULL info. %d", ret);
}

/* Test NULL parameter passed with EP creation */
Test(ep, ep_null_ep)
{
	int ret;

	ret = fi_endpoint(cxit_domain, cxit_fi, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "Failure with NULL ep. %d", ret);
}

struct ep_test_params {
	void *context;
	enum fi_ep_type type;
	int retval;
};

static struct ep_test_params ep_sep_params[] = {
	{.type = FI_EP_RDM,
		.retval = FI_SUCCESS},
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
		.context = (void *)0xabcdef,
		.retval = FI_SUCCESS},
};

ParameterizedTestParameters(ep, fi_ep_types)
{
	size_t param_sz;

	param_sz = ARRAY_SIZE(ep_sep_params);
	return cr_make_param_array(struct ep_test_params, ep_sep_params,
				   param_sz);
}

ParameterizedTest(struct ep_test_params *param, ep, fi_ep_types)
{
	int ret;
	struct cxi_ep *cep;

	cxit_fi->ep_attr->type = param->type;
	cxit_ep = NULL;
	ret = fi_endpoint(cxit_domain, cxit_fi, &cxit_ep, param->context);
	cr_assert_eq(ret, param->retval,
		     "fi_endpoint() error for type %d. %d != %d",
		     param->type, ret, param->retval);

	if (ret != FI_SUCCESS)
		return;

	cr_assert_not_null(cxit_ep);
	cr_expect_eq(cxit_ep->fid.fclass, FI_CLASS_EP);
	cr_expect_eq(cxit_ep->fid.context, param->context);
	cep = container_of(cxit_ep, struct cxi_ep, ep);
	cr_expect_not_null(cep->attr);

	cxit_destroy_ep();
}

ParameterizedTestParameters(ep, fi_sep_types)
{
	size_t param_sz;

	param_sz = ARRAY_SIZE(ep_sep_params);
	return cr_make_param_array(struct ep_test_params, ep_sep_params,
				   param_sz);
}

ParameterizedTest(struct ep_test_params *param, ep, fi_sep_types)
{
	int ret;
	struct cxi_ep *cep;

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
	cep = container_of(cxit_ep, struct cxi_ep, ep);
	cr_expect_not_null(cep->attr);

	cxit_destroy_ep();
}

/* Test Passive EP creation is not supported */
Test(ep, passive_ep)
{
	int ret;
	struct fid_pep *pep = NULL;

	ret = fi_passive_ep(cxit_fabric, cxit_fi, &pep, NULL);
	cr_assert_eq(ret, -FI_ENOSYS, "Failure with fi_passive_ep. %d", ret);
	cr_assert_null(pep);
}

Test(ep, ep_bind_null_bind_obj)
{
	int ret;

	cxit_create_ep();

	ret = fi_ep_bind(cxit_ep, NULL, 0);
	cr_assert_eq(ret, -FI_EINVAL);

	cxit_destroy_ep();
}

Test(ep, sep_bind_null_bind_obj)
{
	int ret;

	cxit_create_sep();

	ret = fi_scalable_ep_bind(cxit_sep, NULL, 0);
	cr_assert_eq(ret, -FI_EINVAL);

	cxit_destroy_sep();
}

Test(ep, ep_bind_invalid_fclass)
{
	int ret;

	cxit_create_ep();
	cxit_create_av();

	/* try to bind an unsupported class type */
	cxit_ep->fid.fclass = FI_CLASS_PEP;
	ret = fi_ep_bind(cxit_ep, &cxit_av->fid, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	cxit_ep->fid.fclass = FI_CLASS_EP;

	cxit_destroy_av();
	cxit_destroy_ep();
}

Test(ep, ep_bind_av)
{
	struct cxi_ep *ep;
	struct cxi_av *av;

	cxit_create_ep();
	cxit_create_av();

	cxit_bind_av();

	av = container_of(cxit_av, struct cxi_av, av_fid.fid);
	ep = container_of(cxit_ep, struct cxi_ep, ep.fid);

	cr_assert_not_null(ep->attr);
	cr_assert_eq(ep->attr->av, av);

	cxit_destroy_ep();
	cxit_destroy_av();
}

Test(ep, ep_bind_eq)
{
	int ret;
	struct fi_eq_attr *attr = NULL;
	struct fid_eq *eq = NULL;
	void *context = NULL;

	ret = fi_eq_open(cxit_fabric, attr, &eq, context);
	cr_assert_eq(ret, -FI_ENOSYS,
		     "TODO Add test for EQs binding to the endpoint when implemented");
}

Test(ep, ep_bind_mr)
{
	int ret;

	/*
	 * At the time of implementing this test MRs were not supported by the
	 * CXI provider. Fake attempting to register a MR with a EP using an AV
	 */
	cxit_create_ep();
	cxit_create_av();

	cxit_av->fid.fclass = FI_CLASS_MR;
	ret = fi_ep_bind(cxit_ep, &cxit_av->fid, 0);
	cr_assert_eq(ret, -FI_EINVAL, "Bind (fake) MR to EP. %d", ret);
	cxit_av->fid.fclass = FI_CLASS_AV;

	cxit_destroy_ep();
	cxit_destroy_av();
}

Test(ep, ep_bind_cq)
{
	struct cxi_ep *ep;
	struct cxi_cq *rx_cq, *tx_cq;
	struct cxi_tx_ctx *tx_ctx = NULL;
	struct cxi_rx_ctx *rx_ctx = NULL;

	cxit_create_ep();
	cxit_create_cqs();
	cr_assert_not_null(cxit_tx_cq);
	cr_assert_not_null(cxit_rx_cq);

	cxit_bind_cqs();

	rx_cq = container_of(cxit_rx_cq, struct cxi_cq, cq_fid.fid);
	tx_cq = container_of(cxit_tx_cq, struct cxi_cq, cq_fid.fid);
	ep = container_of(cxit_ep, struct cxi_ep, ep.fid);

	cr_assert_not_null(ep->attr);

	for (size_t i = 0; i < ep->attr->ep_attr.tx_ctx_cnt; i++) {
		tx_ctx = ep->attr->tx_array[i];

		if (!tx_ctx)
			continue;

		cr_assert_eq(tx_ctx->fid.ctx.fid.fclass, FI_CLASS_TX_CTX);
		cr_assert_eq(tx_ctx->comp.send_cq, tx_cq);
		break;
	}
	cr_assert_not_null(tx_ctx);

	for (size_t i = 0; i < ep->attr->ep_attr.rx_ctx_cnt; i++) {
		rx_ctx = ep->attr->rx_array[i];

		if (!rx_ctx)
			continue;

		cr_assert_eq(rx_ctx->ctx.fid.fclass, FI_CLASS_RX_CTX);
		cr_assert_eq(rx_ctx->comp.recv_cq, rx_cq);
		break;
	}
	cr_assert_not_null(rx_ctx);

	cxit_destroy_ep();
	cxit_destroy_cqs();
}

Test(ep, ep_bind_cntr)
{
	int ret;
	struct fi_cntr_attr *attr = NULL;
	struct fid_cntr *cntr = NULL;
	void *context = NULL;

	ret = fi_cntr_open(cxit_domain, attr, &cntr, context);
	cr_assert_eq(ret, -FI_ENOSYS,
		     "TODO Add test for CNTRs binding to the endpoint when implemented");
}

Test(ep, ep_bind_stx_ctx)
{
	int ret;
	struct fi_tx_attr *attr = NULL;
	void *context = NULL;

	ret = fi_stx_context(cxit_domain, attr, NULL, context);
	cr_assert_eq(ret, -FI_ENOSYS,
		     "TODO Add test for STX CTXs binding to the endpoint when implemented");
}

Test(ep, ep_bind_srx_ctx)
{
	int ret;
	struct fi_rx_attr *attr = NULL;
	void *context = NULL;

	ret = fi_srx_context(cxit_domain, attr, NULL, context);
	cr_assert_eq(ret, -FI_ENOSYS,
		     "TODO Add test for SRX CTXs binding to the endpoint when implemented");
}

Test(ep, ep_bind_unhandled)
{
	int ret;

	cxit_create_ep();
	cxit_create_av();

	/* Emulate a different type of object type */
	cxit_av->fid.fclass = -1;
	ret = fi_ep_bind(cxit_ep, &cxit_av->fid, 0);
	cr_assert_eq(ret, -FI_EINVAL, "fi_ep_bind unhandled object. %d", ret);
	cxit_av->fid.fclass = FI_CLASS_AV;

	cxit_destroy_ep();
	cxit_destroy_av();
}

Test(ep, cancel_ep)
{
	int ret;

	cxit_create_ep();

	ret = fi_cancel(&cxit_ep->fid, NULL);
	cr_assert_eq(ret, FI_SUCCESS);

	cxit_destroy_ep();
}

Test(ep, cancel_unhandled)
{
	int ret;

	cxit_create_ep();

	/* Emulate a different type of object type */
	cxit_ep->fid.fclass = FI_CLASS_PEP;
	ret = fi_cancel(&cxit_ep->fid, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	cxit_ep->fid.fclass = FI_CLASS_EP;

	cxit_destroy_ep();
}
