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
	struct cxip_ep *cep;

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
	cep = container_of(cxit_ep, struct cxip_ep, ep);
	cr_expect_not_null(cep->ep_obj);

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
	struct cxip_ep *ep;
	struct cxip_av *av;

	cxit_create_ep();
	cxit_create_av();

	cxit_bind_av();

	av = container_of(cxit_av, struct cxip_av, av_fid.fid);
	ep = container_of(cxit_ep, struct cxip_ep, ep.fid);

	cr_assert_not_null(ep->ep_obj);
	cr_assert_eq(ep->ep_obj->av, av);

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
	struct cxip_ep *ep;
	struct cxip_cq *rx_cq, *tx_cq;
	struct cxip_tx_ctx *tx_ctx = NULL;
	struct cxip_rx_ctx *rx_ctx = NULL;

	cxit_create_ep();
	cxit_create_cqs();
	cr_assert_not_null(cxit_tx_cq);
	cr_assert_not_null(cxit_rx_cq);

	cxit_bind_cqs();

	rx_cq = container_of(cxit_rx_cq, struct cxip_cq, cq_fid.fid);
	tx_cq = container_of(cxit_tx_cq, struct cxip_cq, cq_fid.fid);
	ep = container_of(cxit_ep, struct cxip_ep, ep.fid);

	cr_assert_not_null(ep->ep_obj);

	for (size_t i = 0; i < ep->ep_obj->ep_attr.tx_ctx_cnt; i++) {
		tx_ctx = ep->ep_obj->tx_array[i];

		if (!tx_ctx)
			continue;

		cr_assert_eq(tx_ctx->fid.ctx.fid.fclass, FI_CLASS_TX_CTX);
		cr_assert_eq(tx_ctx->comp.send_cq, tx_cq);
		break;
	}
	cr_assert_not_null(tx_ctx);

	for (size_t i = 0; i < ep->ep_obj->ep_attr.rx_ctx_cnt; i++) {
		rx_ctx = ep->ep_obj->rx_array[i];

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
	struct cxip_ep *cxi_ep, *alias_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

	alias.fid = &alias_fid;
	alias.flags = FI_TRANSMIT;
	ret = fi_control(&cxit_ep->fid, FI_ALIAS, &alias);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_ALIAS. %d", ret);
	cr_assert_not_null(alias_fid);

	/* verify alias vs cxit_ep */
	alias_ep = container_of(alias_fid, struct cxip_ep, ep.fid);
	cr_assert_eq(alias_ep->ep_obj, cxi_ep->ep_obj, "EP Attr");
	cr_assert_eq(alias_ep->is_alias, 1, "EP is_alias");
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->ep_obj->ref), 1, "EP refs 1");

	/* close alias */
	ret = fi_close(alias_fid);
	cr_assert(ret == FI_SUCCESS, "fi_close endpoint");
	alias_fid = NULL;
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->ep_obj->ref), 0, "EP refs 0");

	cxit_destroy_ep();
}

Test(ep, control_rx_flags_alias)
{
	int ret;
	struct fi_alias alias = {0};
	struct fid *alias_fid = NULL;
	struct cxip_ep *cxi_ep, *alias_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

	alias.fid = &alias_fid;
	alias.flags = FI_RECV;
	ret = fi_control(&cxit_ep->fid, FI_ALIAS, &alias);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_ALIAS. %d", ret);
	cr_assert_not_null(alias_fid);

	alias_ep = container_of(alias_fid, struct cxip_ep, ep.fid);
	cr_assert_eq(alias_ep->ep_obj, cxi_ep->ep_obj, "EP Attr");
	cr_assert_eq(alias_ep->is_alias, 1, "EP is_alias");
	cr_assert_not_null(cxi_ep->ep_obj, "EP attr NULL");
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->ep_obj->ref), 1, "EP refs 1");

	/* close alias */
	ret = fi_close(alias_fid);
	cr_assert(ret == FI_SUCCESS, "fi_close endpoint");
	alias_fid = NULL;
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->ep_obj->ref), 0, "EP refs 0");

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
	struct cxip_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

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
	struct cxip_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

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
	struct cxip_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

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
	struct cxip_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

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
	struct cxip_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

	ret = fi_control(&cxit_ep->fid, FI_SETOPSFLAG, (void *)&flags);
	cr_assert_eq(ret, FI_SUCCESS, "fi_control FI_SETOPSFLAG RX. %d", ret);
	flags &= ~FI_RECV;
	rx_flags = cxi_ep->rx_attr.op_flags;
	cr_assert_eq(rx_flags, flags,
		     "fi_control FI_SETOPSFLAG RX Flag mismatch. %" PRIx64
		     " != %" PRIx64, rx_flags, flags);

	cxit_destroy_ep();
}

Test(ep, control_enable_nocq)
{
	int ret;

	cxit_create_ep();

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, -FI_ENOCQ, "fi_enable. %d", ret);

	cxit_destroy_ep();
}

Test(ep, control_enable_noav)
{
	int ret;

	cxit_create_ep();
	cxit_create_cqs();
	cxit_bind_cqs();

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, -FI_EINVAL, "fi_enable. %d", ret);

	cxit_destroy_ep();
	cxit_destroy_cqs();
}

Test(ep, control_enable)
{
	int ret;

	cxit_create_ep();
	cxit_create_cqs();
	cxit_bind_cqs();
	cxit_create_av();
	cxit_bind_av();

	ret = fi_enable(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable. %d", ret);

	cxit_destroy_ep();
	cxit_destroy_av();
	cxit_destroy_cqs();
}

struct ep_ctrl_null_params {
	int command;
	int retval;
};

ParameterizedTestParameters(ep, ctrl_null_arg)
{
	size_t param_sz;

	static struct ep_ctrl_null_params ep_null_params[] = {
		{.command = -1,
		 .retval = -FI_EINVAL},
		{.command = FI_SETOPSFLAG,
		 .retval = -FI_EINVAL},
		{.command = FI_ENABLE,
		 .retval = -FI_ENOCQ},
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

struct ep_getopt_args {
	int level;
	int optname;
	size_t *optval;
	size_t *optlen;
	int retval;
};

static size_t optvalue;
static size_t optlength;
static struct ep_getopt_args ep_null_params[] = {
	{.level = -1,
	 .retval = -FI_ENOPROTOOPT},
	{.level = FI_OPT_ENDPOINT,
	 .optname = FI_OPT_CM_DATA_SIZE,
	 .retval = -FI_ENOPROTOOPT},
	{.level = FI_OPT_ENDPOINT,
	 .optname = -1,
	 .retval = -FI_ENOPROTOOPT},
	{.level = FI_OPT_ENDPOINT,
	 .optname = FI_OPT_MIN_MULTI_RECV,
	 .optval = NULL,
	 .optlen = NULL,
	 .retval = -FI_EINVAL},
	{.level = FI_OPT_ENDPOINT,
	 .optname = FI_OPT_MIN_MULTI_RECV,
	 .optval = &optvalue,
	 .optlen = NULL,
	 .retval = -FI_EINVAL},
	{.level = FI_OPT_ENDPOINT,
	 .optname = FI_OPT_MIN_MULTI_RECV,
	 .optval = &optvalue,
	 .optlen = &optlength,
	 .retval = FI_SUCCESS},
};

ParameterizedTestParameters(ep, getopt_args)
{
	size_t param_sz;

	param_sz = ARRAY_SIZE(ep_null_params);
	return cr_make_param_array(struct ep_getopt_args, ep_null_params,
				   param_sz);
}

ParameterizedTest(struct ep_getopt_args *param, ep, getopt_args)
{
	int ret;
	struct cxip_ep *cxi_ep;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

	ret = fi_getopt(&cxit_ep->fid, param->level, param->optname,
			(void *)param->optval, param->optlen);
	cr_assert_eq(ret, param->retval,
		     "fi_getopt lvl %d name %d val %p len %p. %d != %d",
		     param->level, param->optname, param->optval,
		     param->optlen, ret, param->retval);

	if (ret == FI_SUCCESS) {
		cr_assert_not_null(cxi_ep->ep_obj);
		cr_assert_eq(*param->optval, cxi_ep->ep_obj->min_multi_recv,
			     "fi_getopt val mismatch. %zd != %zd",
			     *param->optval, cxi_ep->ep_obj->min_multi_recv);
		cr_assert_eq(*param->optlen, sizeof(size_t),
			     "fi_getopt len mismatch. %zd != %zd",
			     *param->optlen, sizeof(size_t));
	}

	cxit_destroy_ep();
}

struct ep_setopt_args {
	int level;
	int optname;
	size_t optval;
	size_t optlen;
	int retval;
};

ParameterizedTestParameters(ep, setopt_args)
{
	size_t param_sz;

	static struct ep_setopt_args ep_null_params[] = {
		{.level = -1,
		.retval = -FI_ENOPROTOOPT},
		{.level = FI_OPT_ENDPOINT,
		.optname = FI_OPT_CM_DATA_SIZE,
		.retval = -FI_ENOPROTOOPT},
		{.level = FI_OPT_ENDPOINT,
		.optname = -1,
		.retval = -FI_ENOPROTOOPT},
		{.level = FI_OPT_ENDPOINT,
		.optname = FI_OPT_MIN_MULTI_RECV,
		.optval = 0,
		.retval = -FI_EINVAL},
		{.level = FI_OPT_ENDPOINT,
		.optname = FI_OPT_MIN_MULTI_RECV,
		.optval = 26,
		.retval = FI_SUCCESS},
		{.level = FI_OPT_ENDPOINT,
		.optname = FI_OPT_MIN_MULTI_RECV,
		.optval = 90001,
		.retval = FI_SUCCESS},
	};

	param_sz = ARRAY_SIZE(ep_null_params);
	return cr_make_param_array(struct ep_setopt_args, ep_null_params,
				   param_sz);
}

ParameterizedTest(struct ep_setopt_args *param, ep, setopt_args)
{
	int ret;
	struct cxip_ep *cxi_ep;
	void *val = NULL;

	if (param->optval != 0)
		val = &param->optval;

	cxit_create_ep();

	cxi_ep = container_of(&cxit_ep->fid, struct cxip_ep, ep.fid);

	ret = fi_setopt(&cxit_ep->fid, param->level, param->optname,
			val, param->optlen);
	cr_assert_eq(ret, param->retval,
		     "fi_setopt lvl %d name %d val %zd. %d != %d",
		     param->level, param->optname, param->optval,
		     ret, param->retval);

	if (ret == FI_SUCCESS) {
		cr_assert_not_null(cxi_ep->ep_obj);
		cr_assert_eq(param->optval, cxi_ep->ep_obj->min_multi_recv,
			     "fi_setopt val mismatch. %zd != %zd",
			     param->optval, cxi_ep->ep_obj->min_multi_recv);
	}

	cxit_destroy_ep();
}

Test(ep, rx_ctx_ep)
{
	int ret;

	cxit_create_ep();

	/* RX context doesn't work with anything but scalable eps */
	ret = fi_rx_context(cxit_ep, 0, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_rx_context bad ep. %d", ret);

	cxit_destroy_ep();
}

Test(ep, rx_ctx_sep_idx)
{
	int ret;

	cxit_create_sep();

	/* Induce index out of range error */
	ret = fi_rx_context(cxit_sep, INT_MAX, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_rx_context bad idx. %d", ret);

	cxit_destroy_sep();
}

Test(ep, rx_ctx_sep_null_rx)
{
	int ret;

	cxit_create_sep();

	/* Null rx_ep value error */
	ret = fi_rx_context(cxit_sep, 0, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_rx_context bad rx. %d", ret);

	cxit_destroy_sep();
}

Test(ep, rx_ctx_sep)
{
	int ret;
	struct cxip_ep *cxi_ep;
	struct cxip_rx_ctx *rx_ctx;
	struct fid_ep *rx_ep = NULL;
	void *context = &ret;
	struct fi_rx_attr *attr = NULL;

	cxit_create_sep();

	cxi_ep = container_of(cxit_sep, struct cxip_ep, ep.fid);

	ret = fi_rx_context(cxit_sep, 0, attr, &rx_ep, context);
	cr_assert_eq(ret, FI_SUCCESS, "fi_rx_context bad rx. %d", ret);
	cr_assert_not_null(rx_ep);

	/* Validate RX Ctx */
	rx_ctx = container_of(rx_ep, struct cxip_rx_ctx, ctx);
	cr_assert_eq(rx_ctx->ep_obj, cxi_ep->ep_obj);
	cr_assert_eq(rx_ctx->domain, cxi_ep->ep_obj->domain);
	cr_assert_eq(rx_ctx->av, cxi_ep->ep_obj->av);
	cr_assert_eq(rx_ctx->min_multi_recv, cxi_ep->ep_obj->min_multi_recv);
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->ep_obj->num_rx_ctx), 1);
	cr_assert_eq(rx_ctx->ctx.fid.fclass, FI_CLASS_RX_CTX);
	cr_assert_eq(rx_ctx->ctx.fid.context, context);

	ret = fi_close(&rx_ep->fid);
	cr_assert_eq(ret, FI_SUCCESS, "fi_close rx_ep. %d", ret);

	cxit_destroy_sep();
}

Test(ep, tx_ctx_ep)
{
	int ret;

	cxit_create_ep();

	/* RX context doesn't work with anything but scalable eps */
	ret = fi_tx_context(cxit_ep, 0, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_tx_context bad ep. %d", ret);

	cxit_destroy_ep();
}

Test(ep, tx_ctx_sep_idx)
{
	int ret;

	cxit_create_sep();

	/* Induce index out of range error */
	ret = fi_tx_context(cxit_sep, INT_MAX, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_tx_context bad idx. %d", ret);

	cxit_destroy_sep();
}

Test(ep, tx_ctx_sep_null_tx)
{
	int ret;

	cxit_create_sep();

	/* Null tx_ep value error */
	ret = fi_tx_context(cxit_sep, 0, NULL, NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_tx_context bad tx. %d", ret);

	cxit_destroy_sep();
}

Test(ep, tx_ctx_sep)
{
	int ret;
	struct cxip_ep *cxi_ep;
	struct cxip_tx_ctx *tx_ctx;
	struct fid_ep *tx_ep = NULL;
	void *context = &ret;
	struct fi_tx_attr *attr = NULL;

	cxit_create_sep();

	cxi_ep = container_of(cxit_sep, struct cxip_ep, ep.fid);

	ret = fi_tx_context(cxit_sep, 0, attr, &tx_ep, context);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tx_context bad tx. %d", ret);
	cr_assert_not_null(tx_ep);

	/* Validate RX Ctx */
	tx_ctx = container_of(tx_ep, struct cxip_tx_ctx, fid.ctx);
	cr_assert_eq(tx_ctx->ep_obj, cxi_ep->ep_obj);
	cr_assert_eq(tx_ctx->domain, cxi_ep->ep_obj->domain);
	cr_assert_eq(tx_ctx->av, cxi_ep->ep_obj->av);
	cr_assert_eq(ofi_atomic_get32(&cxi_ep->ep_obj->num_tx_ctx), 1);
	cr_assert_eq(tx_ctx->fid.ctx.fid.fclass, FI_CLASS_TX_CTX);
	cr_assert_eq(tx_ctx->fclass, FI_CLASS_TX_CTX);
	cr_assert_eq(tx_ctx->fid.ctx.fid.context, context);

	ret = fi_close(&tx_ep->fid);
	cr_assert_eq(ret, FI_SUCCESS, "fi_close tx_ep. %d", ret);

	cxit_destroy_sep();
}

Test(ep, stx_ctx_null_stx)
{
	int ret;
	struct fi_tx_attr *attr = NULL;
	void *context = NULL;

	ret = fi_stx_context(cxit_domain, attr, NULL, context);
	/* TODO Fix when fi_stx_context is implemented, should be -FI_EINVAL */
	cr_assert_eq(ret, -FI_ENOSYS, "fi_stx_context null stx. %d", ret);
}

Test(ep, stx_ctx)
{
	int ret;
	struct fi_tx_attr *attr = NULL;
	struct fid_stx *stx;
	void *context = &ret;
	struct cxip_domain *dom;
	struct cxip_tx_ctx *tx_ctx;
	int refs;

	dom = container_of(cxit_domain, struct cxip_domain, dom_fid);
	refs = ofi_atomic_get32(&dom->ref);

	ret = fi_stx_context(cxit_domain, attr, &stx, context);

	/* TODO Fix when fi_stx_context is implemented, should be FI_SUCCESS */
	cr_assert_eq(ret, -FI_ENOSYS, "fi_stx_context failed. %d", ret);
	if (ret == -FI_ENOSYS)
		return;

	tx_ctx = container_of(stx, struct cxip_tx_ctx, fid.stx);

	/* Validate stx */
	cr_assert_eq(tx_ctx->domain, dom);
	cr_assert_eq(ofi_atomic_inc32(&dom->ref), refs + 1);
	cr_assert_eq(tx_ctx->fid.ctx.fid.fclass, FI_CLASS_TX_CTX);
	cr_assert_eq(tx_ctx->fclass, FI_CLASS_TX_CTX);
	cr_assert_eq(tx_ctx->fid.ctx.fid.context, context);

	ret = fi_close(&stx->fid);
	cr_assert_eq(ret, FI_SUCCESS, "fi_close stx_ep. %d", ret);
}

Test(ep, srx_ctx_null_srx)
{
	int ret;
	struct fi_rx_attr *attr = NULL;
	void *context = NULL;

	ret = fi_srx_context(cxit_domain, attr, NULL, context);
	/* TODO Fix when fi_srx_context is implemented, should be -FI_EINVAL */
	cr_assert_eq(ret, -FI_ENOSYS, "fi_srx_context null srx. %d", ret);
}

Test(ep, srx_ctx)
{
	int ret;
	struct fi_rx_attr *attr = NULL;
	struct fid_ep *srx;
	void *context = &ret;
	struct cxip_domain *dom;
	struct cxip_rx_ctx *rx_ctx;
	int refs;

	dom = container_of(cxit_domain, struct cxip_domain, dom_fid);
	refs = ofi_atomic_get32(&dom->ref);

	ret = fi_srx_context(cxit_domain, attr, &srx, context);
	/* TODO Fix when fi_srx_context is implemented, should be FI_SUCCESS */
	cr_assert_eq(ret, -FI_ENOSYS, "fi_stx_context failed. %d", ret);
	if (ret == -FI_ENOSYS)
		return;

	rx_ctx = container_of(srx, struct cxip_rx_ctx, ctx);

	/* Validate stx */
	cr_assert_eq(rx_ctx->domain, dom);
	cr_assert_eq(ofi_atomic_inc32(&dom->ref), refs + 1);
	cr_assert_eq(rx_ctx->ctx.fid.fclass, FI_CLASS_SRX_CTX);
	cr_assert_eq(rx_ctx->ctx.fid.fclass, FI_CLASS_RX_CTX);
	cr_assert_eq(rx_ctx->ctx.fid.context, context);
	cr_assert_eq(rx_ctx->enabled, 1);
	cr_assert_eq(rx_ctx->min_multi_recv, CXIP_EP_MIN_MULTI_RECV);

	ret = fi_close(&srx->fid);
	cr_assert_eq(ret, FI_SUCCESS, "fi_close srx_ep. %d", ret);
}
