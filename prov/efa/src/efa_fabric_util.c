/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <ofi_util.h>

#include "config.h"
#include "efa.h"
#include "efa_fabric_util.h"

int efa_fabric_destruct_base(struct efa_fabric *efa_fabric)
{
	int ret;

	ret = ofi_fabric_close(&efa_fabric->util_fabric);
	if (ret)
		EFA_WARN(FI_LOG_FABRIC,
			 "Unable to close fabric: %s\n",
			 fi_strerror(-ret));
	return ret;
}

int efa_non_cq_trywait(struct fid *fid)
{
	struct util_wait *wait;

	switch (fid->fclass) {
	case FI_CLASS_EQ:
		wait = container_of(fid, struct util_eq, eq_fid.fid)->wait;
		break;
	case FI_CLASS_CNTR:
		wait = container_of(fid, struct util_cntr, cntr_fid.fid)->wait;
		break;
	case FI_CLASS_WAIT:
		wait = container_of(fid, struct util_wait, wait_fid.fid);
		break;
	default:
		return -FI_EINVAL;
	}

	return wait->wait_try(wait);
}

bool efa_feature_in(const char * const *list, size_t n, const char *feature)
{
	size_t i;

	if (!feature)
		return false;

	for (i = 0; i < n; i++)
		if (list[i] && strcmp(list[i], feature) == 0)
			return true;

	return false;
}

/*
 * Walk efa_util_prov.info, calling ofi_fabric_init for each entry until
 * something other than -FI_ENODATA is returned. The first matching info
 * configures @p efa_fabric->util_fabric. Shared by efa_fabric_open_base
 * and efa_rdm_fabric_open.
 */
int efa_fabric_init_base(struct efa_fabric *efa_fabric,
			 struct fi_fabric_attr *attr,
			 void *context)
{
	const struct fi_info *info;
	int ret = -FI_ENODATA;

	for (info = efa_util_prov.info; info; info = info->next) {
		ret = ofi_fabric_init(&efa_prov, info->fabric_attr, attr,
				      &efa_fabric->util_fabric, context);
		if (ret != -FI_ENODATA)
			break;
	}

	return ret;
}
