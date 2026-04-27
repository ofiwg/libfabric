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
#include "efa_prov_info.h"
#ifdef EFA_PERF_ENABLED
const char *efa_perf_counters_str[] = {
	EFA_PERF_FOREACH(OFI_STR)
};
#endif


static int efa_fabric_close(fid_t fid)
{
	struct efa_fabric *efa_fabric;
	int ret;

	efa_fabric = container_of(fid, struct efa_fabric, util_fabric.fabric_fid.fid);
	ret = ofi_fabric_close(&efa_fabric->util_fabric);
	if (ret) {
		EFA_WARN(FI_LOG_FABRIC,
			"Unable to close fabric: %s\n",
			fi_strerror(-ret));
		return ret;
	}

	if (efa_fabric->shm_fabric) {
		ret = fi_close(&efa_fabric->shm_fabric->fid);
		if (ret) {
			EFA_WARN(FI_LOG_FABRIC,
				"Unable to close fabric: %s\n",
				fi_strerror(-ret));
			return ret;
		}
	}

#ifdef EFA_PERF_ENABLED
	ofi_perfset_log(&efa_fabric->perf_set, efa_perf_counters_str);
	ofi_perfset_close(&efa_fabric->perf_set);
#endif
	free(efa_fabric);

	return 0;
}

static int efa_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	struct efa_cq *efa_cq;
	struct util_wait *wait;
	int ret, i;

	for (i = 0; i < count; i++) {
		if (fids[i]->fclass == FI_CLASS_CQ) {
			/* Use EFA-specific CQ trywait */
			efa_cq = container_of(fids[i], struct efa_cq, util_cq.cq_fid.fid);
			if (container_of(efa_cq->util_cq.domain, struct efa_domain, util_domain)->info_type == EFA_INFO_RDM) {
				if (!efa_cq->util_cq.wait)
					return -FI_EINVAL;
				ret = efa_cq->util_cq.wait->wait_try(efa_cq->util_cq.wait);
			} else {
				ret = efa_cq_trywait(efa_cq);
			}
			if (ret)
				return ret;
		} else {
			/* Use generic util trywait logic for non-CQ types */
			switch (fids[i]->fclass) {
			case FI_CLASS_EQ:
				wait = container_of(fids[i], struct util_eq, eq_fid.fid)->wait;
				break;
			case FI_CLASS_CNTR:
				wait = container_of(fids[i], struct util_cntr, cntr_fid.fid)->wait;
				break;
			case FI_CLASS_WAIT:
				wait = container_of(fids[i], struct util_wait, wait_fid.fid);
				break;
			default:
				return -FI_EINVAL;
			}

			ret = wait->wait_try(wait);
			if (ret)
				return ret;
		}
	}
	return FI_SUCCESS;
}

/*
 * Feature strings advertised by this build, per fabric.
 *
 * Features are runtime-discoverable flags whose answer may differ
 * between efa-direct and efa (efa-proto) because the two fabrics
 * exercise different code paths. efa_features_direct contains strings
 * valid for the efa-direct fabric; efa_features_proto contains
 * strings valid for the efa fabric (RDM and DGRAM). A string absent
 * from the matching list returns false.
 *
 * Consumers query by literal string, so renaming is a breaking change;
 * prefer adding new strings over editing existing ones.
 */
static const char * const efa_features_direct[] = {
	"mixed_hmem_iov",
};

static const char * const efa_features_proto[] = {
	/*
	 * "mixed_hmem_iov" is not advertised on the efa (RDM) fabric:
	 * efa_rdm_pke_copy_payload_to_ope() still dispatches copy
	 * methods based on desc[0] only, so a multi-iov request mixing
	 * host and HMEM descriptors can still misbehave even after the
	 * send-path descriptor scans are in place.
	 *
	 * The NULL placeholder keeps this a valid initializer under
	 * pre-C23 compilers (notably MSVC), which reject an empty {}.
	 * efa_feature_in() skips NULL entries.
	 */
	NULL,
};

static bool efa_feature_in(const char * const *list, size_t n,
			   const char *feature)
{
	size_t i;

	if (!feature)
		return false;

	for (i = 0; i < n; i++)
		if (list[i] && strcmp(list[i], feature) == 0)
			return true;

	return false;
}

static bool efa_fabric_feature_query_direct(const char *feature)
{
	return efa_feature_in(efa_features_direct,
			      ARRAY_SIZE(efa_features_direct), feature);
}

static bool efa_fabric_feature_query_proto(const char *feature)
{
	return efa_feature_in(efa_features_proto,
			      ARRAY_SIZE(efa_features_proto), feature);
}

static struct fi_efa_feature_ops efa_feature_ops_direct = {
	.query = efa_fabric_feature_query_direct,
};

static struct fi_efa_feature_ops efa_feature_ops_proto = {
	.query = efa_fabric_feature_query_proto,
};

static int efa_fabric_ops_open(struct fid *fid, const char *ops_name,
			       uint64_t flags, void **ops, void *context)
{
	struct efa_fabric *efa_fabric;

	if (strcmp(ops_name, FI_EFA_FEATURE_OPS) == 0) {
		efa_fabric = container_of(fid, struct efa_fabric,
					  util_fabric.fabric_fid.fid);
		if (strcasecmp(efa_fabric->util_fabric.name,
			       EFA_DIRECT_FABRIC_NAME) == 0)
			*ops = &efa_feature_ops_direct;
		else
			*ops = &efa_feature_ops_proto;
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
	.ops_open = efa_fabric_ops_open,
};

static struct fi_ops_fabric efa_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = efa_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = efa_trywait
};

int efa_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric_fid,
	       void *context)
{
	const struct fi_info *info;
	struct efa_fabric *efa_fabric;
	int ret = 0;

	efa_fabric = calloc(1, sizeof(*efa_fabric));
	if (!efa_fabric)
		return -FI_ENOMEM;

	for (info = efa_util_prov.info; info; info = info->next) {
		ret = ofi_fabric_init(&efa_prov, info->fabric_attr, attr,
				      &efa_fabric->util_fabric, context);
		if (ret != -FI_ENODATA)
			break;
	}

	if (ret)
		goto err_free_fabric;

#ifdef EFA_PERF_ENABLED
	ret = ofi_perfset_create(&efa_prov, &efa_fabric->perf_set,
				 efa_perf_size, perf_domain, perf_cntr,
				 perf_flags);

	if (ret)
		EFA_WARN(FI_LOG_FABRIC,
			"Error initializing EFA perfset: %s\n",
			fi_strerror(-ret));
#endif


	*fabric_fid = &efa_fabric->util_fabric.fabric_fid;
	(*fabric_fid)->fid.fclass = FI_CLASS_FABRIC;
	(*fabric_fid)->fid.ops = &efa_fi_ops;
	(*fabric_fid)->ops = &efa_ops_fabric;
	(*fabric_fid)->api_version = attr->api_version;

	return 0;

err_free_fabric:
	free(efa_fabric);

	return ret;
}
