/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2022 Hewlett Packard Enterprise Development LP
 */
#include <stdio.h>
#include <stdlib.h>
#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

void *memdup(const void *src, size_t n)
{
	void *dest;

	dest = malloc(n);
	if (dest == NULL)
		return NULL;

	return memcpy(dest, src, n);
}

TestSuite(auth_key, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test fi_getinfo() verification of hints argument. */
Test(auth_key, missing_auth_key_domain_attr_hints)
{
	struct cxi_auth_key auth_key = {
		.svc_id = CXI_DEFAULT_SVC_ID,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints, "strdup failed");

	hints->domain_attr->auth_key_size = sizeof(auth_key);
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
}

/* Test fi_getinfo() verification of hints argument. */
Test(auth_key, invalid_auth_key_size_domain_attr_hints)
{
	struct cxi_auth_key auth_key = {
		.svc_id = CXI_DEFAULT_SVC_ID,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints, "strdup failed");

	hints->domain_attr->auth_key_size = 1;
	hints->domain_attr->auth_key = memdup(&auth_key, 1);
	cr_assert_not_null(hints->domain_attr->auth_key, "memdup failed");

	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
}

/* Test fi_getinfo() verification of hints argument. */
Test(auth_key, missing_auth_key_size_domain_attr_hints)
{
	struct cxi_auth_key auth_key = {
		.svc_id = CXI_DEFAULT_SVC_ID,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	hints->domain_attr->auth_key = memdup(&auth_key, 1);
	cr_assert_not_null(hints->domain_attr->auth_key, "memdup failed");

	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
}

/* Test fi_getinfo() verification of hints argument. */
Test(auth_key, missing_auth_key_ep_attr_hints)
{
	struct cxi_auth_key auth_key = {
		.svc_id = CXI_DEFAULT_SVC_ID,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints, "strdup failed");

	hints->ep_attr->auth_key_size = sizeof(auth_key);
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
}

/* Test fi_getinfo() verification of hints argument. */
Test(auth_key, invalid_auth_key_size_ep_attr_hints)
{
	struct cxi_auth_key auth_key = {
		.svc_id = CXI_DEFAULT_SVC_ID,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	hints->ep_attr->auth_key_size = 1;
	hints->ep_attr->auth_key = memdup(&auth_key, 1);
	cr_assert_not_null(hints->ep_attr->auth_key, "memdup failed");

	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
}

/* Test fi_getinfo() verification of hints argument. */
Test(auth_key, missing_auth_key_size_ep_attr_hints)
{
	struct cxi_auth_key auth_key = {
		.svc_id = CXI_DEFAULT_SVC_ID,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	hints->ep_attr->auth_key = memdup(&auth_key, 1);
	cr_assert_not_null(hints->ep_attr->auth_key, "memdup failed");

	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
}

/* Verify fi_getinfo() correctly echos back a valid auth_key hint using the
 * default svc_id.
 */
Test(auth_key, valid_default_domain_auth_key_hint)
{
	struct cxi_auth_key auth_key = {
		.svc_id = CXI_DEFAULT_SVC_ID,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	hints->domain_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	cr_assert_not_null(hints->domain_attr->auth_key, "memdup failed");

	hints->domain_attr->auth_key_size = sizeof(auth_key);
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	cr_assert_not_null(info->domain_attr->auth_key, "NULL domain auth_key");
	cr_assert_eq(hints->domain_attr->auth_key_size,
		     info->domain_attr->auth_key_size,
		     "fi_getinfo returned auth_key_size does not match hints");

	ret = memcmp(hints->domain_attr->auth_key, info->domain_attr->auth_key,
		     hints->domain_attr->auth_key_size);
	cr_assert_eq(ret, 0, "fi_getinfo returned auth_key does not match hints");

	fi_freeinfo(info);
	fi_freeinfo(hints);
}

/* Verify fi_getinfo() correctly echos back a valid auth_key hint using the
 * default svc_id.
 */
Test(auth_key, valid_default_ep_auth_key_hint)
{
	struct cxi_auth_key auth_key = {
		.svc_id = CXI_DEFAULT_SVC_ID,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	hints->ep_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	cr_assert_not_null(hints->ep_attr->auth_key, "memdup failed");

	hints->ep_attr->auth_key_size = sizeof(auth_key);
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	cr_assert_not_null(info->ep_attr->auth_key, "NULL ep auth_key");
	cr_assert_eq(hints->ep_attr->auth_key_size,
		     info->ep_attr->auth_key_size,
		     "fi_getinfo returned auth_key_size does not match hints");

	ret = memcmp(hints->ep_attr->auth_key, info->ep_attr->auth_key,
		     hints->ep_attr->auth_key_size);
	cr_assert_eq(ret, 0, "fi_getinfo returned auth_key does not match hints");

	/* Since hints domain auth_key is NULL, CXI provider should echo the
	 * hints ep auth_key into info domain auth_key. This is the behavior
	 * some MPICH versions expect.
	 */
	cr_assert_not_null(info->domain_attr->auth_key, "NULL domain auth_key");
	cr_assert_eq(hints->ep_attr->auth_key_size,
		     info->domain_attr->auth_key_size,
		     "fi_getinfo returned auth_key_size does not match hints");

	ret = memcmp(hints->ep_attr->auth_key, info->domain_attr->auth_key,
		     hints->ep_attr->auth_key_size);
	cr_assert_eq(ret, 0, "fi_getinfo returned auth_key does not match hints");

	fi_freeinfo(info);
	fi_freeinfo(hints);
}

/* Verify fi_getinfo() rejects a svc_id which has not been allocated thus making
 * the auth_key invalid.
 */
Test(auth_key, invalid_user_defined_domain_svc_id_hint)
{
	struct cxi_auth_key auth_key = {
		.svc_id = 0xffff,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	hints->domain_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	cr_assert_not_null(hints->domain_attr->auth_key, "memdup failed");

	hints->domain_attr->auth_key_size = sizeof(auth_key);
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
}

/* Verify fi_getinfo() rejects a svc_id which has not been allocated thus making
 * the auth_key invalid.
 */
Test(auth_key, invalid_user_defined_ep_svc_id_hint)
{
	struct cxi_auth_key auth_key = {
		.svc_id = 0xffff,
		.vni = 1234,
	};
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	hints->ep_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	cr_assert_not_null(hints->ep_attr->auth_key, "memdup failed");

	hints->ep_attr->auth_key_size = sizeof(auth_key);
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
}

/* Verify fi_domain() rejects an invalid auth_key. */
Test(auth_key, invalid_user_defined_domain_svc_id)
{
	struct cxi_auth_key auth_key = {
		.svc_id = 0xffff,
		.vni = 1234,
	};
	int ret;
	struct fi_info *info;
	struct fid_fabric *fab;
	struct fid_domain *dom;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, NULL, &info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	ret = fi_fabric(info->fabric_attr, &fab, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_fabric failed: %d", ret);

	/* Override auth_key with bad auth_key. */
	if (info->domain_attr->auth_key)
		free(info->domain_attr->auth_key);
	info->domain_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	info->domain_attr->auth_key_size = sizeof(auth_key);

	ret = fi_domain(fab, info, &dom, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_domain failed: %d", ret);

	fi_close(&fab->fid);
	fi_freeinfo(info);
}

/* Verify fi_endpoint() rejects an invalid auth_key. */
Test(auth_key, invalid_user_defined_ep_svc_id)
{
	struct cxi_auth_key auth_key = {
		.svc_id = 0xffff,
		.vni = 1234,
	};
	int ret;
	struct fi_info *info;
	struct fid_fabric *fab;
	struct fid_domain *dom;
	struct fid_ep *ep;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, NULL, &info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	ret = fi_fabric(info->fabric_attr, &fab, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_fabric failed: %d", ret);

	ret = fi_domain(fab, info, &dom, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_domain failed: %d", ret);

	/* Override auth_key with bad auth_key. */
	if (info->domain_attr->auth_key)
		free(info->domain_attr->auth_key);
	info->domain_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	info->domain_attr->auth_key_size = sizeof(auth_key);

	ret = fi_endpoint(dom, info, &ep, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_endpoint failed: %d", ret);

	fi_close(&dom->fid);
	fi_close(&fab->fid);
	fi_freeinfo(info);
}

/* Valid service ID but invalid VNI for the service ID. */
Test(auth_key, valid_user_defined_svc_id_invalid_vni_hints)
{
	int ret;
	struct cxil_dev *dev;
	struct cxi_svc_fail_info fail_info = {};
	struct cxi_svc_desc svc_desc = {};
	uint16_t valid_vni = 0x120;
	struct fi_info *hints;
	struct fi_info *info;
	struct cxi_auth_key auth_key = {
		.vni = 0x123,
	};

	/* Need to allocate a service to be used by libfabric. */
	ret = cxil_open_device(0, &dev);
	cr_assert_eq(ret, 0, "cxil_open_device failed: %d", ret);

	svc_desc.restricted_vnis = 1;
	svc_desc.enable = 1;
	svc_desc.num_vld_vnis = 1;
	svc_desc.vnis[0] = valid_vni;

	ret = cxil_alloc_svc(dev, &svc_desc, &fail_info);
	cr_assert_gt(ret, 0, "cxil_alloc_svc failed: %d", ret);
	svc_desc.svc_id = ret;

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	auth_key.svc_id = svc_desc.svc_id;
	hints->ep_attr->auth_key_size = sizeof(auth_key);
	hints->ep_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	cr_assert_not_null(hints->ep_attr->auth_key, "memdup failed");

	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA, "fi_getinfo failed: %d", ret);

	fi_freeinfo(hints);
	ret = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(ret, 0, "cxil_destroy_svc failed: %d", ret);
	cxil_close_device(dev);
}

/* Valid service ID but invalid VNI for the service ID. */
Test(auth_key, valid_user_defined_svc_id_invalid_vni_dom_attr)
{
	int ret;
	struct cxil_dev *dev;
	struct cxi_svc_fail_info fail_info = {};
	struct cxi_svc_desc svc_desc = {};
	uint16_t valid_vni = 0x120;
	struct fi_info *info;
	struct cxi_auth_key auth_key = {
		.vni = 0x123,
	};
	struct fid_fabric *fab;
	struct fid_domain *dom;

	/* Need to allocate a service to be used by libfabric. */
	ret = cxil_open_device(0, &dev);
	cr_assert_eq(ret, 0, "cxil_open_device failed: %d", ret);

	svc_desc.restricted_vnis = 1;
	svc_desc.enable = 1;
	svc_desc.num_vld_vnis = 1;
	svc_desc.vnis[0] = valid_vni;

	ret = cxil_alloc_svc(dev, &svc_desc, &fail_info);
	cr_assert_gt(ret, 0, "cxil_alloc_svc failed: %d", ret);
	svc_desc.svc_id = ret;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, NULL, &info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	ret = fi_fabric(info->fabric_attr, &fab, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_fabric failed: %d", ret);

	/* Override auth_key with bad auth_key. */
	auth_key.svc_id = svc_desc.svc_id;

	if (info->domain_attr->auth_key)
		free(info->domain_attr->auth_key);
	info->domain_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	info->domain_attr->auth_key_size = sizeof(auth_key);

	ret = fi_domain(fab, info, &dom, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_domain failed: %d", ret);

	fi_close(&fab->fid);
	fi_freeinfo(info);
	ret = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(ret, 0, "cxil_destroy_svc failed: %d", ret);
	cxil_close_device(dev);
}

/* Valid service ID but invalid VNI for the service ID. */
Test(auth_key, valid_user_defined_svc_id_invalid_vni_ep_attr)
{
	int ret;
	struct cxil_dev *dev;
	struct cxi_svc_fail_info fail_info = {};
	struct cxi_svc_desc svc_desc = {};
	uint16_t valid_vni = 0x120;
	struct fi_info *info;
	struct cxi_auth_key auth_key = {
		.vni = 0x123,
	};
	struct fid_fabric *fab;
	struct fid_domain *dom;
	struct fid_ep *ep;

	/* Need to allocate a service to be used by libfabric. */
	ret = cxil_open_device(0, &dev);
	cr_assert_eq(ret, 0, "cxil_open_device failed: %d", ret);

	svc_desc.restricted_vnis = 1;
	svc_desc.enable = 1;
	svc_desc.num_vld_vnis = 1;
	svc_desc.vnis[0] = valid_vni;

	ret = cxil_alloc_svc(dev, &svc_desc, &fail_info);
	cr_assert_gt(ret, 0, "cxil_alloc_svc failed: %d", ret);
	svc_desc.svc_id = ret;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 NULL, FI_SOURCE, NULL, &info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	ret = fi_fabric(info->fabric_attr, &fab, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_fabric failed: %d", ret);

	ret = fi_domain(fab, info, &dom, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_domain failed: %d", ret);

	/* Override auth_key with bad auth_key. */
	auth_key.svc_id = svc_desc.svc_id;

	if (info->domain_attr->auth_key)
		free(info->domain_attr->auth_key);
	info->domain_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	info->domain_attr->auth_key_size = sizeof(auth_key);

	ret = fi_endpoint(dom, info, &ep, NULL);
	cr_assert_eq(ret, -FI_EINVAL, "fi_endpoint failed: %d", ret);

	fi_close(&dom->fid);
	fi_close(&fab->fid);
	fi_freeinfo(info);
	ret = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(ret, 0, "cxil_destroy_svc failed: %d", ret);
	cxil_close_device(dev);
}

static void alloc_endpoint(struct fi_info *info, struct fid_fabric **fab,
			   struct fid_domain **dom, struct fid_av **av,
			   struct fid_cq **cq, struct fid_ep **ep)
{
	int ret;
	struct fi_cq_attr cq_attr = {
		.format = FI_CQ_FORMAT_TAGGED,
	};
	struct fi_av_attr av_attr = {};

	ret = fi_fabric(info->fabric_attr, fab, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_fabric failed: %d", ret);

	ret = fi_domain(*fab, info, dom, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_domain failed: %d", ret);

	ret = fi_cq_open(*dom, &cq_attr, cq, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_cq_open failed: %d", ret);

	ret = fi_av_open(*dom, &av_attr, av, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_av_open failed: %d", ret);

	ret = fi_endpoint(*dom, info, ep, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_endpoint failed: %d", ret);

	ret = fi_ep_bind(*ep, &(*av)->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_ep_bind failed: %d", ret);

	ret = fi_ep_bind(*ep, &(*cq)->fid, FI_TRANSMIT | FI_RECV);
	cr_assert_eq(ret, FI_SUCCESS, "fi_ep_bind failed: %d", ret);

	ret = fi_enable(*ep);
	cr_assert_eq(ret, FI_SUCCESS, "fi_enable failed: %d", ret);
}

Test(auth_key, valid_user_defined_svc_id_valid_vni_verify_vni_enforcement)
{
	int ret;
	struct cxil_dev *dev;
	struct cxi_svc_fail_info fail_info = {};
	struct cxi_svc_desc svc_desc = {};
	struct fi_info *hints;
	struct fi_info *default_info;
	struct fi_info *user_info;
	struct cxi_auth_key auth_key = {};
	uint16_t valid_vni = 0x1234;
	struct fid_fabric *default_fab;
	struct fid_domain *default_dom;
	struct fid_av *default_av;
	struct fid_cq *default_cq;
	struct fid_ep *default_ep;
	struct fid_fabric *user_fab;
	struct fid_domain *user_dom;
	struct fid_av *user_av;
	struct fid_cq *user_cq;
	struct fid_ep *user_ep;
	char buf[256];
	fi_addr_t target_default_ep;
	struct fi_cq_tagged_entry event;
	struct fi_cq_err_entry error;

	/* Need to allocate a service to be used by libfabric. */
	ret = cxil_open_device(0, &dev);
	cr_assert_eq(ret, 0, "cxil_open_device failed: %d", ret);

	svc_desc.restricted_vnis = 1;
	svc_desc.enable = 1;
	svc_desc.num_vld_vnis = 1;
	svc_desc.vnis[0] = valid_vni;

	ret = cxil_alloc_svc(dev, &svc_desc, &fail_info);
	cr_assert_gt(ret, 0, "cxil_alloc_svc failed: %d", ret);
	svc_desc.svc_id = ret;

	/* Allocate infos for RDMA test. Default_info users the provider
	 * assigned default auth_key where user_info uses the user defined
	 * auth_key.
	 */
	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 "0", FI_SOURCE, NULL, &default_info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	hints = fi_allocinfo();
	cr_assert_not_null(hints, "fi_allocinfo failed");

	hints->domain_attr->mr_mode = FI_MR_ENDPOINT;

	hints->fabric_attr->prov_name = strdup("cxi");
	cr_assert_not_null(hints->fabric_attr->prov_name, "strdup failed");

	auth_key.svc_id = svc_desc.svc_id;
	auth_key.vni = valid_vni;
	hints->domain_attr->auth_key_size = sizeof(auth_key);
	hints->domain_attr->auth_key = memdup(&auth_key, sizeof(auth_key));
	cr_assert_not_null(hints->domain_attr->auth_key, "memdup failed");

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), "cxi0",
			 "255", FI_SOURCE, hints, &user_info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	/* Allocate endpoints using different service IDs and VNIs. */
	alloc_endpoint(default_info, &default_fab, &default_dom, &default_av,
		       &default_cq, &default_ep);
	alloc_endpoint(user_info, &user_fab, &user_dom, &user_av,
		       &user_cq, &user_ep);

	/* Insert the default EP address into the user AVs. */
	ret = fi_av_insert(user_av, default_info->src_addr, 1,
			   &target_default_ep, 0, NULL);
	cr_assert_eq(ret, 1, "fi_av_insert failed: %d", ret);

	/* These two endpoints should not be able to talk due to operating in
	 * different VNIs. This should result in an I/O error at the initiator.
	 */
	ret = fi_recv(default_ep, buf, sizeof(buf), NULL, FI_ADDR_UNSPEC, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed: %d", ret);

	ret = fi_send(user_ep, buf, sizeof(buf), NULL, target_default_ep, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_send failed: %d", ret);

	do {
		ret = fi_cq_read(user_cq, &event, 1);
	} while (ret == -FI_EAGAIN);

	cr_assert_eq(ret, -FI_EAVAIL, "fi_cq_read failed: %d", ret);

	ret = fi_cq_readerr(user_cq, &error, 0);
	cr_assert_eq(ret, 1, "fi_cq_readerr failed: %d", ret);

	/* Single these tests are loopback on the same NIC, RC_PTLTE_NOT_FOUND
	 * is returned instead of RC_VNI_NOT_FOUND since the VNI is valid.
	 * Non-loopback should returned RC_VNI_NOT_FOUND.
	 */
	cr_assert_eq(error.prov_errno, C_RC_PTLTE_NOT_FOUND,
		     "Bad error.prov_errno: got=%d expected=%d",
		     error.prov_errno, C_RC_PTLTE_NOT_FOUND);

	fi_close(&user_ep->fid);
	fi_close(&user_cq->fid);
	fi_close(&user_av->fid);
	fi_close(&user_dom->fid);
	fi_close(&user_fab->fid);
	fi_close(&default_ep->fid);
	fi_close(&default_cq->fid);
	fi_close(&default_av->fid);
	fi_close(&default_dom->fid);
	fi_close(&default_fab->fid);
	fi_freeinfo(user_info);
	fi_freeinfo(hints);
	fi_freeinfo(default_info);
	ret = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(ret, 0, "cxil_destroy_svc failed: %d", ret);
	cxil_close_device(dev);
}
