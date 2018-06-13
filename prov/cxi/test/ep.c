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
	cr_assert(cxit_ep != NULL);

	cxit_destroy_ep();
}

/* Test NULL parameter passed with EP creation */
Test(ep, ep_null_info)
{
	int ret;

	ret = fi_endpoint(cxit_domain, NULL, &cxit_ep, NULL);
	cr_assert(ret == -FI_EINVAL, "Failure with NULL info. %d", ret);
}

/* Test NULL parameter passed with EP creation */
Test(ep, ep_null_ep)
{
	int ret;

	ret = fi_endpoint(cxit_domain, cxit_fi, NULL, NULL);
	cr_assert(ret == -FI_EINVAL, "Failure with NULL ep. %d", ret);
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
	cr_assert(ret == -FI_ENOSYS, "Failure with fi_passive_ep. %d", ret);
	cr_assert_null(pep);
}

