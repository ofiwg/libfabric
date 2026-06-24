/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2014-2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "config.h"

#include <netdb.h>
#include <inttypes.h>

#include <infiniband/efadv.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include "ofi_prov.h"
#include <ofi_util.h>

#include "efa_cq.h"
#include "efa_domain.h"
#include "efa_fabric_util.h"
#include "rdm/efa_rdm_domain.h"
#include "rdm/efa_rdm_fabric.h"
#include "efa_prov_info.h"

static int efa_fabric_close(fid_t fid)
{
	struct efa_fabric *efa_fabric;
	int ret;

	efa_fabric = container_of(fid, struct efa_fabric, util_fabric.fabric_fid.fid);
	ret = efa_fabric_destruct_base(efa_fabric);
	if (ret)
		return ret;

	free(efa_fabric);
	return 0;
}

static int efa_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	struct efa_cq *efa_cq;
	int ret, i;

	for (i = 0; i < count; i++) {
		if (fids[i]->fclass == FI_CLASS_CQ) {
			efa_cq = container_of(fids[i], struct efa_cq, util_cq.cq_fid.fid);
			ret = efa_cq_trywait(efa_cq);
		} else {
			ret = efa_non_cq_trywait(fids[i]);
		}
		if (ret)
			return ret;
	}
	return FI_SUCCESS;
}

/*
 * Feature strings advertised by the efa-direct fabric.
 *
 * Features are runtime-discoverable flags whose answer may differ
 * between efa-direct and efa (efa-rdm) because the two fabrics
 * exercise different code paths. A string absent from this list
 * returns false.
 *
 * Consumers query by literal string, so renaming is a breaking change;
 * prefer adding new strings over editing existing ones.
 */
static const char * const efa_features_direct[] = {
	"mixed_hmem_iov",
};

static bool efa_fabric_feature_query_direct(const char *feature)
{
	return efa_feature_in(efa_features_direct,
			      ARRAY_SIZE(efa_features_direct), feature);
}

static struct fi_efa_feature_ops efa_feature_ops_direct = {
	.query = efa_fabric_feature_query_direct
};

static int efa_fabric_ops_open(struct fid *fid, const char *ops_name,
			       uint64_t flags, void **ops, void *context)
{
	if (strcmp(ops_name, FI_EFA_FEATURE_OPS) == 0) {
		*ops = &efa_feature_ops_direct;
		return FI_SUCCESS;
	}

	EFA_WARN(FI_LOG_FABRIC, "Unknown ops name: %s\n", ops_name);
	return -FI_EINVAL;
}

static struct fi_ops efa_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = efa_fabric_ops_open
};

static struct fi_ops_fabric efa_ops_fabric_direct = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = efa_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = efa_trywait
};

static int efa_fabric_open_base(struct fi_fabric_attr *attr,
				struct fid_fabric **fabric_fid, void *context)
{
	struct efa_fabric *efa_fabric;
	int ret;

	efa_fabric = calloc(1, sizeof(*efa_fabric));
	if (!efa_fabric)
		return -FI_ENOMEM;

	ret = efa_fabric_init_base(efa_fabric, attr, context);
	if (ret)
		goto err_free_fabric;

	*fabric_fid = &efa_fabric->util_fabric.fabric_fid;
	(*fabric_fid)->fid.fclass = FI_CLASS_FABRIC;
	(*fabric_fid)->fid.ops = &efa_fi_ops;
	(*fabric_fid)->ops = &efa_ops_fabric_direct;
	(*fabric_fid)->api_version = attr->api_version;

	return 0;

err_free_fabric:
	free(efa_fabric);
	return ret;
}

int efa_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric_fid,
	       void *context)
{
	if (attr && attr->name &&
	    strcasecmp(attr->name, EFA_DIRECT_FABRIC_NAME) == 0)
		return efa_fabric_open_base(attr, fabric_fid, context);
	return efa_rdm_fabric_open(attr, fabric_fid, context);
}
