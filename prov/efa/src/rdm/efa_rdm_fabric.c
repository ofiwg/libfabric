/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2014-2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "config.h"

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <ofi_util.h>

#include "efa.h"
#include "efa_cq.h"
#include "efa_domain.h"
#include "efa_fabric_util.h"
#include "rdm/efa_rdm_domain.h"
#include "rdm/efa_rdm_fabric.h"

#ifdef EFA_PERF_ENABLED
static const char *efa_perf_counters_str[] = {
	EFA_PERF_FOREACH(OFI_STR)
};
#endif

static int efa_rdm_fabric_close(fid_t fid)
{
	struct efa_rdm_fabric *rdm_fabric;
	int ret;

	rdm_fabric = container_of(fid, struct efa_rdm_fabric,
				  efa_fabric.util_fabric.fabric_fid.fid);

	ret = efa_fabric_destruct_base(&rdm_fabric->efa_fabric);
	if (ret)
		return ret;

	if (rdm_fabric->shm_fabric) {
		ret = fi_close(&rdm_fabric->shm_fabric->fid);
		if (ret) {
			EFA_WARN(FI_LOG_FABRIC,
				"Unable to close fabric: %s\n",
				fi_strerror(-ret));
			return ret;
		}
	}

#ifdef EFA_PERF_ENABLED
	ofi_perfset_log(&rdm_fabric->perf_set, efa_perf_counters_str);
	ofi_perfset_close(&rdm_fabric->perf_set);
#endif
	free(rdm_fabric);

	return 0;
}

static const char * const efa_rdm_features[] = {
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

static bool efa_rdm_fabric_feature_query(const char *feature)
{
	return efa_feature_in(efa_rdm_features,
			      ARRAY_SIZE(efa_rdm_features), feature);
}

static struct fi_efa_feature_ops efa_rdm_feature_ops = {
	.query = efa_rdm_fabric_feature_query
};

static int efa_rdm_fabric_ops_open(struct fid *fid, const char *ops_name,
				   uint64_t flags, void **ops, void *context)
{
	if (strcmp(ops_name, FI_EFA_FEATURE_OPS) == 0) {
		*ops = &efa_rdm_feature_ops;
		return FI_SUCCESS;
	}

	EFA_WARN(FI_LOG_FABRIC, "Unknown ops name: %s\n", ops_name);
	return -FI_EINVAL;
}

static struct fi_ops efa_rdm_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_rdm_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = efa_rdm_fabric_ops_open
};

static int efa_rdm_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	struct efa_cq *efa_cq;
	int ret, i;

	for (i = 0; i < count; i++) {
		if (fids[i]->fclass == FI_CLASS_CQ) {
			efa_cq = container_of(fids[i], struct efa_cq, util_cq.cq_fid.fid);
			if (!efa_cq->util_cq.wait)
				return -FI_EINVAL;
			ret = efa_cq->util_cq.wait->wait_try(efa_cq->util_cq.wait);
		} else {
			ret = efa_non_cq_trywait(fids[i]);
		}
		if (ret)
			return ret;
	}
	return FI_SUCCESS;
}

static struct fi_ops_fabric efa_rdm_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = efa_rdm_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = efa_rdm_trywait
};

int efa_rdm_fabric_open(struct fi_fabric_attr *attr,
			struct fid_fabric **fabric_fid, void *context)
{
	struct efa_rdm_fabric *rdm_fabric;
	int ret;

	rdm_fabric = calloc(1, sizeof(*rdm_fabric));
	if (!rdm_fabric)
		return -FI_ENOMEM;

	ret = efa_fabric_init_base(&rdm_fabric->efa_fabric, attr, context);
	if (ret)
		goto err_free_fabric;

#ifdef EFA_PERF_ENABLED
	ret = ofi_perfset_create(&efa_prov, &rdm_fabric->perf_set,
				 efa_perf_size, perf_domain, perf_cntr,
				 perf_flags);
	if (ret)
		EFA_WARN(FI_LOG_FABRIC,
			"Error initializing EFA perfset: %s\n",
			fi_strerror(-ret));
#endif

	*fabric_fid = &rdm_fabric->efa_fabric.util_fabric.fabric_fid;
	(*fabric_fid)->fid.fclass = FI_CLASS_FABRIC;
	(*fabric_fid)->fid.ops = &efa_rdm_fi_ops;
	(*fabric_fid)->ops = &efa_rdm_ops_fabric;
	(*fabric_fid)->api_version = attr->api_version;

	return 0;

err_free_fabric:
	free(rdm_fabric);
	return ret;
}
