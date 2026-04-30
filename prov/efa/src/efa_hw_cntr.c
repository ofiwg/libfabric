/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "ofi_util.h"
#include "efa.h"
#include "efa_cq.h"
#include "efa_hw_cntr.h"

#if HAVE_EFADV_CREATE_COMP_CNTR
struct fi_ops efa_hw_cntr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_hw_cntr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

struct fi_ops_cntr efa_hw_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.read = efa_hw_cntr_read,
	.readerr = efa_hw_cntr_readerr,
	.add = efa_hw_cntr_add,
	.adderr = efa_hw_cntr_adderr,
	.set = efa_hw_cntr_set,
	.seterr = efa_hw_cntr_seterr,
	.wait = efa_hw_cntr_wait,
};

static int efa_hw_cntr_check_attr(struct efa_domain *efa_domain,
				  const struct fi_cntr_attr *attr,
				  struct ibv_comp_cntr_init_attr *cc_attr)
{
	if (efa_domain->info->domain_attr->max_cntr_value >
	    efa_domain->device->comp_count_max_value) {
		EFA_WARN(FI_LOG_CNTR,
			 "Domain max_cntr_value (%lu) exceeds completion counter limit (%lu).\n",
			 efa_domain->info->domain_attr->max_cntr_value,
			 efa_domain->device->comp_count_max_value);
		return -FI_EOPNOTSUPP;
	}

	if (efa_domain->info->domain_attr->max_err_cntr_value >
	    efa_domain->device->err_count_max_value) {
		EFA_WARN(FI_LOG_CNTR,
			 "Domain max_err_cntr_value (%lu) exceeds error counter limit (%lu).\n",
			 efa_domain->info->domain_attr->max_err_cntr_value,
			 efa_domain->device->err_count_max_value);
		return -FI_EOPNOTSUPP;
	}

	if (!attr) {
		cc_attr->type = IBV_COMP_CNTR_TYPE_WRS;
		return FI_SUCCESS;
	}

	if (attr->wait_obj != FI_WAIT_NONE &&
	    attr->wait_obj != FI_WAIT_UNSPEC) {
		EFA_WARN(FI_LOG_CNTR,
			 "Only FI_WAIT_NONE and FI_WAIT_UNSPEC are supported "
			 "for hardware counters\n");
		return -FI_EOPNOTSUPP;
	}

	if (attr->flags) {
		EFA_WARN(FI_LOG_CNTR,
			 "Unsupported flags for hardware counter.\n");
		return -FI_EOPNOTSUPP;
	}

	switch (attr->events) {
	case FI_CNTR_EVENTS_COMP:
		cc_attr->type = IBV_COMP_CNTR_TYPE_WRS;
		break;
	case FI_CNTR_EVENTS_BYTES:
		cc_attr->type = IBV_COMP_CNTR_TYPE_BYTES;
		break;
	default:
		EFA_WARN(FI_LOG_CNTR,
			 "Unsupported events type %d for hardware counter.\n",
			 attr->events);
		return -FI_EOPNOTSUPP;
	}

	return FI_SUCCESS;
}

int efa_hw_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		     struct efa_cntr *cntr, struct fid_cntr **cntr_fid,
		     void *context,
		     struct efadv_comp_cntr_init_attr *efa_cc_attr)
{
	struct efa_domain *efa_domain;
	struct ibv_comp_cntr_init_attr cc_attr = {0};
	int ret;

	if (!efa_env.use_hw_cntr)
		return -FI_EOPNOTSUPP;

	efa_domain = container_of(domain, struct efa_domain, util_domain.domain_fid);

	ret = efa_hw_cntr_check_attr(efa_domain, attr, &cc_attr);
	if (ret)
		return ret;

	cntr->ibv_comp_cntr = efadv_create_comp_cntr(efa_domain->device->ibv_ctx,
						     &cc_attr, efa_cc_attr,
						     sizeof(*efa_cc_attr));
	if (!cntr->ibv_comp_cntr) {
		EFA_WARN_ERRNO(FI_LOG_CNTR, "efadv_create_comp_cntr failed.", errno);
		return -errno;
	}

	ret = efa_cntr_construct(cntr, domain, attr, efa_cntr_progress, context);
	if (ret) {
		ibv_destroy_comp_cntr(cntr->ibv_comp_cntr);
		return ret;
	}

	cntr->util_cntr.cntr_fid.fid.ops = &efa_hw_cntr_fi_ops;
	cntr->util_cntr.cntr_fid.ops = &efa_hw_cntr_ops;
	*cntr_fid = &cntr->util_cntr.cntr_fid;
	cntr->wait_obj = attr ? attr->wait_obj : FI_WAIT_UNSPEC;

	return FI_SUCCESS;
}

int efa_hw_cntr_close(struct fid *fid)
{
	struct efa_cntr *cntr;
	int ret, retv;

	retv = 0;
	cntr = container_of(fid, struct efa_cntr, util_cntr.cntr_fid.fid);

	ret = ibv_destroy_comp_cntr(cntr->ibv_comp_cntr);
	if (ret) {
		EFA_WARN_ERRNO(FI_LOG_CNTR, "ibv_destroy_comp_cntr failed", ret);
		retv = -ret;
	}

	efa_cntr_destruct(cntr);
	free(cntr);
	return retv;
}

int efa_hw_cntr_add(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct efa_cntr *cntr;
	int ret;

	cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);
	assert(cntr->ibv_comp_cntr);
	ret = ibv_inc_comp_cntr(cntr->ibv_comp_cntr, value);
	if (ret) {
		EFA_WARN_ERRNO(FI_LOG_CNTR, "ibv_inc_comp_cntr failed", ret);
		return -ret;
	}
	return FI_SUCCESS;
}

int efa_hw_cntr_adderr(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct efa_cntr *cntr;
	int ret;

	cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);
	assert(cntr->ibv_comp_cntr);
	ret = ibv_inc_err_comp_cntr(cntr->ibv_comp_cntr, value);
	if (ret) {
		EFA_WARN_ERRNO(FI_LOG_CNTR, "ibv_inc_err_comp_cntr failed", ret);
		return -ret;
	}
	return FI_SUCCESS;
}

int efa_hw_cntr_set(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct efa_cntr *cntr;
	int ret;

	cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);
	assert(cntr->ibv_comp_cntr);
	ret = ibv_set_comp_cntr(cntr->ibv_comp_cntr, value);
	if (ret) {
		EFA_WARN_ERRNO(FI_LOG_CNTR, "ibv_set_comp_cntr failed", ret);
		return -ret;
	}
	return FI_SUCCESS;
}

int efa_hw_cntr_seterr(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct efa_cntr *cntr;
	int ret;

	cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);
	assert(cntr->ibv_comp_cntr);
	ret = ibv_set_err_comp_cntr(cntr->ibv_comp_cntr, value);
	if (ret) {
		EFA_WARN_ERRNO(FI_LOG_CNTR, "ibv_set_err_comp_cntr failed", ret);
		return -ret;
	}
	return FI_SUCCESS;
}

uint64_t efa_hw_cntr_read(struct fid_cntr *cntr_fid)
{
	struct efa_cntr *cntr;
	uint64_t value;
	int ret;

	cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);

	/* Progress CQ to complete WQE in SQ and RQ */
	(void) efa_cntr_read(cntr_fid);

	if (cntr->comp_use_device_mem) {
		EFA_WARN(FI_LOG_CNTR,
			 "fi_cntr_read not supported for counters in device memory\n");
		return 0;
	}

	assert(cntr->ibv_comp_cntr);
	ret = ibv_read_comp_cntr(cntr->ibv_comp_cntr, &value);
	if (ret) {
		EFA_WARN_ERRNO(FI_LOG_CNTR, "ibv_read_comp_cntr failed", ret);
		return 0;
	}
	return value;
}

uint64_t efa_hw_cntr_readerr(struct fid_cntr *cntr_fid)
{
	struct efa_cntr *cntr;
	uint64_t value;
	int ret;

	cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);

	/* Progress CQ to complete WQE in SQ and RQ */
	(void) efa_cntr_readerr(cntr_fid);

	if (cntr->err_use_device_mem) {
		EFA_WARN(FI_LOG_CNTR,
			 "fi_cntr_readerr not supported for error counters in device memory\n");
		return 0;
	}

	assert(cntr->ibv_comp_cntr);
	ret = ibv_read_err_comp_cntr(cntr->ibv_comp_cntr, &value);
	if (ret) {
		EFA_WARN_ERRNO(FI_LOG_CNTR, "ibv_read_err_comp_cntr failed", ret);
		return 0;
	}
	return value;
}

int efa_hw_cntr_wait(struct fid_cntr *cntr_fid, uint64_t threshold, int timeout)
{
	struct efa_cntr *cntr;
	uint64_t start, errcnt;
	int ret = -FI_ETIMEDOUT;
	int numtry = 5;
	int tryid = 0;
	int waitim = 1;
	static const int waitim_max = 1000; /* cap at 1ms */

	cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);
	if (cntr->comp_use_device_mem || cntr->err_use_device_mem) {
		EFA_WARN(FI_LOG_CNTR,
			 "fi_cntr_wait not supported for counters in device "
			 "memory\n");
		return -FI_EOPNOTSUPP;
	}

	if (cntr->wait_obj == FI_WAIT_NONE) {
		EFA_WARN(FI_LOG_CNTR,
			 "Invalid to call fi_cntr_wait with FI_WAIT_NONE\n");
		return -FI_EINVAL;
	}

	errcnt = efa_hw_cntr_readerr(cntr_fid);
	start = (timeout >= 0) ? ofi_gettime_ms() : 0;
	for (tryid = 0; tryid < numtry; ++tryid) {
		if (threshold <= efa_hw_cntr_read(cntr_fid)) {
			ret = FI_SUCCESS;
			break;
		}
		if (errcnt != efa_hw_cntr_readerr(cntr_fid)) {
			ret = -FI_EAVAIL;
			break;
		}
		if (timeout >= 0) {
			timeout -= (int) (ofi_gettime_ms() - start);
			if (timeout <= 0) {
				ret = -FI_ETIMEDOUT;
				break;
			}
		} else {
			tryid = 0;
		}
		usleep(waitim);
		if (waitim < waitim_max)
			waitim *= 2;
	}
	return ret;
}
#endif /* HAVE_EFADV_CREATE_COMP_CNTR */
