/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

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

Test(ep, control_unhandled_obj)
{
	int ret;

	cxit_create_ep();

	/* Emulate a different type of object type */
	cxit_ep->fid.fclass = FI_CLASS_PEP;
	ret = fi_control(&cxit_ep->fid, -1, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	cxit_ep->fid.fclass = FI_CLASS_EP;

	cxit_destroy_ep();
}

Test(ep, control_unhandled_cmd)
{
	int ret;

	cxit_create_ep();

	ret = fi_control(&cxit_ep->fid, -1, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	cxit_destroy_ep();
}

Test(ep, control_null_fid_alias)
{
	int ret;
	struct fi_alias alias = {0};

	cxit_create_ep();

	/* A null alias.fid causes -FI_EINVAL */
	ret = fi_control(&cxit_ep->fid, FI_ALIAS, &alias);
	cr_assert_eq(ret, -FI_EINVAL, "fi_control FI_ALIAS. %d", ret);

	cxit_destroy_ep();
}

Test(ep, control_empty_alias)
{
	int ret;
	struct fi_alias alias = {0};
	struct fid *alias_fid;

	cxit_create_ep();

	/* Empty alias.flags causes -FI_EINVAL */
	alias.fid = &alias_fid;
	ret = fi_control(&cxit_ep->fid, FI_ALIAS, &alias);
	cr_assert_eq(ret, -FI_EINVAL, "fi_control FI_ALIAS. %d", ret);

	cxit_destroy_ep();
}

Test(ep, control_bad_flags_alias)
{
	int ret;
	struct fi_alias alias = {0};

	cxit_create_ep();

	/* Both Tx and Rx flags causes -FI_EINVAL */
	alias.flags = FI_TRANSMIT | FI_RECV;
	ret = fi_control(&cxit_ep->fid, FI_ALIAS, &alias);
	cr_assert_eq(ret, -FI_EINVAL, "fi_control FI_ALIAS. %d", ret);

	cxit_destroy_ep();
}

Test(ep, control_tx_flags_alias)
{
	int ret;
	struct fi_alias alias = {0};
	struct fid *alias_fid = NULL;
	struct cxi_ep *cxi_ep, *alias_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxi_ep, ep.fid);

	alias.fid = &alias_fid;
	alias.flags = FI_TRANSMIT;
	ret = fi_control(&cxit_ep->fid, FI_ALIAS, &alias);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_ALIAS. %d", ret);
	cr_assert_not_null(alias_fid);

	/* verify alias vs cxit_ep */
	alias_ep = container_of(alias_fid, struct cxi_ep, ep.fid);
	cr_assert_eq(alias_ep->attr, cxi_ep->attr, "EP Attr");
	cr_assert_eq(alias_ep->is_alias, 1, "EP is_alias");
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->attr->ref), 1, "EP refs 1");

	/* close alias */
	ret = fi_close(alias_fid);
	cr_assert(ret == FI_SUCCESS, "fi_close endpoint");
	alias_fid = NULL;
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->attr->ref), 0, "EP refs 0");

	cxit_destroy_ep();
}

Test(ep, control_rx_flags_alias)
{
	int ret;
	struct fi_alias alias = {0};
	struct fid *alias_fid = NULL;
	struct cxi_ep *cxi_ep, *alias_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxi_ep, ep.fid);

	alias.fid = &alias_fid;
	alias.flags = FI_RECV;
	ret = fi_control(&cxit_ep->fid, FI_ALIAS, &alias);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_ALIAS. %d", ret);
	cr_assert_not_null(alias_fid);

	alias_ep = container_of(alias_fid, struct cxi_ep, ep.fid);
	cr_assert_eq(alias_ep->attr, cxi_ep->attr, "EP Attr");
	cr_assert_eq(alias_ep->is_alias, 1, "EP is_alias");
	cr_assert_not_null(cxi_ep->attr, "EP attr NULL");
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->attr->ref), 1, "EP refs 1");

	/* close alias */
	ret = fi_close(alias_fid);
	cr_assert(ret == FI_SUCCESS, "fi_close endpoint");
	alias_fid = NULL;
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->attr->ref), 0, "EP refs 0");

	cxit_destroy_ep();
}

Test(ep, control_getopsflag_both_tx_rx)
{
	int ret;
	uint64_t flags = FI_TRANSMIT | FI_RECV;

	cxit_create_ep();

	ret = fi_control(&cxit_ep->fid, FI_GETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, -FI_EINVAL, "fi_control FI_GETOPSFLAG TX/RX. %d",
		     ret);

	cxit_destroy_ep();
}

Test(ep, control_getopsflag_no_flags)
{
	int ret;
	uint64_t flags = FI_TRANSMIT | FI_RECV;

	cxit_create_ep();

	ret = fi_control(&cxit_ep->fid, FI_GETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, -FI_EINVAL, "fi_control FI_GETOPSFLAG 0. %d", ret);

	cxit_destroy_ep();
}

Test(ep, control_getopsflag_tx)
{
	int ret;
	uint64_t flags = FI_TRANSMIT;
	struct cxi_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxi_ep, ep.fid);

	ret = fi_control(&cxit_ep->fid, FI_GETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_GETOPSFLAG TX. %d", ret);
	cr_assert_eq(cxi_ep->tx_attr.op_flags, flags,
		     "fi_control FI_GETOPSFLAG Flag mismatch. %" PRIx64 " != %"
		     PRIx64 " ", cxi_ep->tx_attr.op_flags, flags);

	cxit_destroy_ep();
}

Test(ep, control_getopsflag_rx)
{
	int ret;
	uint64_t flags = FI_RECV;
	struct cxi_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxi_ep, ep.fid);

	ret = fi_control(&cxit_ep->fid, FI_GETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_GETOPSFLAG RX. %d", ret);
	cr_assert_eq(cxi_ep->rx_attr.op_flags, flags,
		     "fi_control FI_GETOPSFLAG Flag mismatch. %" PRIx64 " != %"
		     PRIx64 " ", cxi_ep->rx_attr.op_flags, flags);

	cxit_destroy_ep();
}

Test(ep, control_setopsflag_both_tx_rx)
{
	int ret;
	uint64_t flags = FI_TRANSMIT | FI_RECV;

	cxit_create_ep();

	ret = fi_control(&cxit_ep->fid, FI_SETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, -FI_EINVAL, "fi_control FI_SETOPSFLAG TX/RX. %d",
		     ret);

	cxit_destroy_ep();
}

Test(ep, control_setopsflag_no_flags)
{
	int ret;
	uint64_t flags = FI_TRANSMIT | FI_RECV;

	cxit_create_ep();

	ret = fi_control(&cxit_ep->fid, FI_SETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, -FI_EINVAL, "fi_control FI_SETOPSFLAG 0. %d", ret);

	cxit_destroy_ep();
}

Test(ep, control_setopsflag_tx)
{
	int ret;
	uint64_t flags = (FI_TRANSMIT | FI_MSG | FI_TRIGGER |
			  FI_DELIVERY_COMPLETE);
	uint64_t tx_flags;
	struct cxi_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxi_ep, ep.fid);

	ret = fi_control(&cxit_ep->fid, FI_SETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_SETOPSFLAG TX. %d", ret);
	flags &= ~FI_TRANSMIT;
	tx_flags = cxi_ep->tx_attr.op_flags;
	cr_assert_eq(tx_flags, flags,
		     "fi_control FI_SETOPSFLAG TX Flag mismatch. %" PRIx64
		     " != %" PRIx64, tx_flags, flags);

	cxit_destroy_ep();
}

Test(ep, control_setopsflag_tx_complete)
{
	int ret;
	uint64_t flags = FI_TRANSMIT | FI_MSG | FI_TRIGGER | FI_AFFINITY;
	uint64_t tx_flags;
	struct cxi_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxi_ep, ep.fid);

	ret = fi_control(&cxit_ep->fid, FI_SETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_SETOPSFLAG TX. %d", ret);
	flags &= ~FI_TRANSMIT;
	flags |= FI_TRANSMIT_COMPLETE;
	tx_flags = cxi_ep->tx_attr.op_flags;
	cr_assert_eq(tx_flags, flags,
		     "fi_control FI_SETOPSFLAG TXcomp Flag mismatch. %" PRIx64
		     " != %" PRIx64, tx_flags, flags);

	cxit_destroy_ep();
}

Test(ep, control_setopsflag_rx)
{
	int ret;
	uint64_t flags = FI_RECV | FI_TAGGED | FI_NUMERICHOST | FI_EVENT;
	uint64_t rx_flags;
	struct cxi_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxi_ep, ep.fid);

	ret = fi_control(&cxit_ep->fid, FI_SETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_SETOPSFLAG RX. %d", ret);
	flags &= ~FI_RECV;
	rx_flags = cxi_ep->rx_attr.op_flags;
	cr_assert_eq(rx_flags, flags,
		     "fi_control FI_SETOPSFLAG RX Flag mismatch. %" PRIx64
		     " != %" PRIx64, rx_flags, flags);

	cxit_destroy_ep();
}

Test(ep, control_enable)
{
	int ret;
	struct fi_alias alias = {0};
	struct cxi_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxi_ep, ep.fid);

	ret = fi_control(&cxit_ep->fid, FI_ENABLE, &alias);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_ENABLE. %d", ret);
	cr_assert_not_null(cxi_ep->attr, "EP attr NULL");
	cr_assert_eq(cxi_ep->attr->is_enabled, 1, "EP not enabled");

	cxit_destroy_ep();
}

struct ep_ctrl_null_params {
	int command;
	int retval;
};

ParameterizedTestParameters(ep, ctrl_null_arg)
{
	size_t param_sz;

	static struct ep_ctrl_null_params ep_null_params[] = {
		{.command = FI_EP_RDM,
		 .retval = -FI_EINVAL},
		{.command = FI_EP_UNSPEC,
		 .retval = -FI_EINVAL},
		{.command = FI_SETOPSFLAG,
		 .retval = -FI_EINVAL},
		{.command = FI_ENABLE,
		 .retval = FI_SUCCESS},
	};

	param_sz = ARRAY_SIZE(ep_null_params);
	return cr_make_param_array(struct ep_ctrl_null_params, ep_null_params,
				   param_sz);
}

ParameterizedTest(struct ep_ctrl_null_params *param, ep, ctrl_null_arg)
{
	int ret;

	cxit_create_ep();

	ret = fi_control(&cxit_ep->fid, param->command, NULL);
	cr_assert_eq(ret, param->retval, "fi_control type %d. %d != %d",
		     param->command, ret, param->retval);

	cxit_destroy_ep();
}
