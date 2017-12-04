/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "psmx2.h"
#include "prov.h"
#include <glob.h>
#include <dlfcn.h>

static int psmx2_init_count = 0;
static int psmx2_lib_initialized = 0;
static pthread_mutex_t psmx2_lib_mutex;

struct psmx2_env psmx2_env = {
	.name_server	= 1,
	.tagged_rma	= 1,
	.uuid		= PSMX2_DEFAULT_UUID,
	.delay		= 1,
	.timeout	= 5,
	.prog_interval	= -1,
	.prog_affinity	= NULL,
	.sep		= 0,
	.max_trx_ctxt	= 1,
	.sep_trx_ctxt	= 0,
	.num_devunits	= 1,
	.inject_size	= PSMX2_INJECT_SIZE,
	.lock_level	= 2,
};

static void psmx2_init_env(void)
{
	if (getenv("OMPI_COMM_WORLD_RANK") || getenv("PMI_RANK"))
		psmx2_env.name_server = 0;

	fi_param_get_bool(&psmx2_prov, "name_server", &psmx2_env.name_server);
	fi_param_get_bool(&psmx2_prov, "tagged_rma", &psmx2_env.tagged_rma);
	fi_param_get_str(&psmx2_prov, "uuid", &psmx2_env.uuid);
	fi_param_get_int(&psmx2_prov, "delay", &psmx2_env.delay);
	fi_param_get_int(&psmx2_prov, "timeout", &psmx2_env.timeout);
	fi_param_get_int(&psmx2_prov, "prog_interval", &psmx2_env.prog_interval);
	fi_param_get_str(&psmx2_prov, "prog_affinity", &psmx2_env.prog_affinity);
	fi_param_get_int(&psmx2_prov, "inject_size", &psmx2_env.inject_size);
	fi_param_get_bool(&psmx2_prov, "lock_level", &psmx2_env.lock_level);
}

static int psmx2_check_sep_cap(void)
{
	if (!psmx2_sep_ok())
		return 0;

	psmx2_env.sep = 1;
	psmx2_env.max_trx_ctxt = PSMX2_MAX_TRX_CTXT;
	psmx2_env.sep_trx_ctxt = PSMX2_MAX_TRX_CTXT - 1;

	return 1;
}

static int psmx2_init_lib(int default_multi_ep)
{
	int major, minor;
	int ret = 0, err;

	if (psmx2_lib_initialized)
		return 0;

	pthread_mutex_lock(&psmx2_lib_mutex);

	if (psmx2_lib_initialized)
		goto out;

	if (default_multi_ep)
		setenv("PSM2_MULTI_EP", "1", 0);

	psm2_error_register_handler(NULL, PSM2_ERRHANDLER_NO_HANDLER);

	major = PSM2_VERNO_MAJOR;
	minor = PSM2_VERNO_MINOR;

	err = psm2_init(&major, &minor);
	if (err != PSM2_OK) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"psm2_init failed: %s\n", psm2_error_get_string(err));
		ret = err;
		goto out;
	}

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"PSM header version = (%d, %d)\n", PSM2_VERNO_MAJOR, PSM2_VERNO_MINOR);
	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"PSM library version = (%d, %d)\n", major, minor);

	if (psmx2_check_sep_cap())
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "Scalable EP enabled.\n");
	else
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "Scalable EP disabled.\n");

	psmx2_lib_initialized = 1;

out:
	pthread_mutex_unlock(&psmx2_lib_mutex);
	return ret;
}

#define PSMX2_SYSFS_PATH "/sys/class/infiniband/hfi1"
static int psmx2_read_sysfs_int(int unit, char *entry)
{
	char path[64];
	char buffer[32];
	int n, ret = 0;

	sprintf(path, "%s_%d", PSMX2_SYSFS_PATH, unit);
	n = fi_read_file(path, entry, buffer, 32);
	if (n > 0) {
		buffer[n] = 0;
		ret = strtol(buffer, NULL, 10);
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "%s/%s: %d\n", path, entry, ret);
	}

	return ret;
}

static void psmx2_update_sep_cap(void)
{
	int i;
	int nctxts = 0;
	int nfreectxts = 0;

	for (i = 0; i < psmx2_env.num_devunits; i++) {
		nctxts += psmx2_read_sysfs_int(i, "nctxts");
		nfreectxts += psmx2_read_sysfs_int(i, "nfreectxts");
	}

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"hfi1 contexts: total %d, free %d\n",
		nctxts, nfreectxts);

	if (nctxts > PSMX2_MAX_TRX_CTXT)
		nctxts = PSMX2_MAX_TRX_CTXT;

	if (nfreectxts > PSMX2_MAX_TRX_CTXT)
		nfreectxts = PSMX2_MAX_TRX_CTXT;

	psmx2_env.max_trx_ctxt = nctxts;
	psmx2_env.sep_trx_ctxt = nfreectxts;

	/*
	 * One context is reserved for regular endpoints. It is allocated
	 * when the domain is opened.
	 */
	if ((!psmx2_active_fabric || !psmx2_active_fabric->active_domain) &&
	    psmx2_env.sep_trx_ctxt)
		--psmx2_env.sep_trx_ctxt;

	FI_INFO(&psmx2_prov, FI_LOG_CORE, "SEP: %d Tx/Rx contexts allowed.\n",
		psmx2_env.sep_trx_ctxt);
}

static int psmx2_getinfo(uint32_t version, const char *node,
			 const char *service, uint64_t flags,
			 struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *psmx2_info;
	uint32_t cnt = 0;
	struct psmx2_ep_name *dest_addr = NULL;
	struct psmx2_ep_name *src_addr = NULL;
	int ep_type = FI_EP_RDM;
	int av_type = FI_AV_UNSPEC;
	uint64_t mode = FI_CONTEXT;
	enum fi_mr_mode mr_mode = FI_MR_SCALABLE;
	enum fi_threading threading = FI_THREAD_COMPLETION;
	enum fi_progress control_progress = FI_PROGRESS_MANUAL;
	enum fi_progress data_progress = FI_PROGRESS_MANUAL;
	uint64_t caps = PSMX2_CAPS;
	uint64_t max_tag_value = -1ULL;
	int err = -FI_ENODATA;
	int svc0, svc = PSMX2_ANY_SERVICE;
	glob_t glob_buf;
	int tx_ctx_cnt, rx_ctx_cnt;
	int default_multi_ep = 0;
	int addr_format = FI_ADDR_PSMX2;
	size_t len;
	void *addr;
	uint32_t fmt;

	FI_INFO(&psmx2_prov, FI_LOG_CORE,"\n");

	*info = NULL;

	if (FI_VERSION_GE(version, FI_VERSION(1,5)))
		mr_mode = 0;

	/*
	 * Try to turn on PSM2 multi-EP support if the application asks for
	 * more than one tx context per endpoint. This only works for the
	 * very first fi_getinfo() call.
	 */
	if (hints && hints->ep_attr && hints->ep_attr->tx_ctx_cnt > 1)
		default_multi_ep = 1;

	if (psmx2_init_lib(default_multi_ep))
		return -FI_ENODATA;

	/*
	 * psm2_ep_num_devunits() may wait for 15 seconds before return
	 * when /dev/hfi1_0 is not present. Check the existence of any hfi1
	 * device interface first to avoid this delay. Note that the devices
	 * don't necessarily appear consecutively so we need to check all
	 * possible device names before returning "no device found" error.
	 * This also means if "/dev/hfi1_0" doesn't exist but other devices
	 * exist, we are still going to see the delay; but that's a rare case.
	 */
	if ((glob("/dev/hfi1_[0-9]", 0, NULL, &glob_buf) != 0) &&
	    (glob("/dev/hfi1_[0-9][0-9]", GLOB_APPEND, NULL, &glob_buf) != 0)) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"no hfi1 device is found.\n");
		return -FI_ENODATA;
	}
	globfree(&glob_buf);

	if (psm2_ep_num_devunits(&cnt) || !cnt) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"no PSM2 device is found.\n");
		return -FI_ENODATA;
	}

	psmx2_env.num_devunits = cnt;
	psmx2_init_env();

	psmx2_update_sep_cap();
	tx_ctx_cnt = psmx2_env.sep_trx_ctxt;
	rx_ctx_cnt = psmx2_env.sep_trx_ctxt;

	if (node &&
	    !ofi_str_toaddr(node, &fmt, &addr, &len) &&
	    fmt == FI_ADDR_PSMX2) {
		if (flags & FI_SOURCE) {
			src_addr = addr;
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"'%s' is taken as src_addr: <unit=%d, port=%d, service=%d>\n",
				node, src_addr->unit, src_addr->port, src_addr->service);
		} else {
			dest_addr = addr;
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"'%s' is taken as dest_addr: <epid=%"PRIu64", vl=%d>\n",
				node, dest_addr->epid, dest_addr->vlane);
		}
		node = NULL;
	}

	if (!src_addr) {
		src_addr = calloc(1, sizeof(*src_addr));
		if (!src_addr) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"failed to allocate src addr.\n");
			return -FI_ENODATA;
		}
		src_addr->type = PSMX2_EP_SRC_ADDR;
		src_addr->epid = PSMX2_RESERVED_EPID;
		src_addr->unit = PSMX2_DEFAULT_UNIT;
		src_addr->port = PSMX2_DEFAULT_PORT;
		src_addr->service = PSMX2_ANY_SERVICE;

		if (flags & FI_SOURCE) {
			if (node)
				sscanf(node, "%*[^:]:%" SCNi8 ":%" SCNu8,
				       &src_addr->unit, &src_addr->port);
			if (service)
				sscanf(service, "%" SCNu32, &src_addr->service);
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"node '%s' service '%s' converted to <unit=%d, port=%d, service=%d>\n",
				node, service, src_addr->unit, src_addr->port, src_addr->service);
		}
	}

	if (!dest_addr && node && !(flags & FI_SOURCE)) {
		struct util_ns ns = (const struct util_ns){ 0 };
		struct util_ns_attr ns_attr = (const struct util_ns_attr){ 0 };
		psm2_uuid_t uuid;

		psmx2_get_uuid(uuid);
		ns_attr.ns_port = psmx2_uuid_to_port(uuid);
		ns_attr.name_len = sizeof(*dest_addr);
		ns_attr.service_len = sizeof(svc);
		ns_attr.service_cmp = psmx2_ns_service_cmp;
		ns_attr.is_service_wildcard = psmx2_ns_is_service_wildcard;
		err = ofi_ns_init(&ns_attr, &ns);
		if (err) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"ofi_ns_init returns %d\n", err);
			err = -FI_ENODATA;
			goto err_out;
		}

		if (service)
			svc = atoi(service);
		svc0 = svc;
		dest_addr = (struct psmx2_ep_name *)
			ofi_ns_resolve_name(&ns, node, &svc);
		if (dest_addr) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"'%s:%u' resolved to <epid=%"PRIu64", vl=%d>:%d\n",
				node, svc0, dest_addr->epid,
				dest_addr->vlane, svc);
		} else {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"failed to resolve '%s:%u'.\n", node, svc);
			err = -FI_ENODATA;
			goto err_out;
		}
	}

	if (hints) {
		switch (hints->addr_format) {
		case FI_FORMAT_UNSPEC:
		case FI_ADDR_PSMX2:
			break;
		case FI_ADDR_STR:
			addr_format = FI_ADDR_STR;
			break;
		default:
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"hints->addr_format=%d, supported=%d,%d,%d.\n",
				hints->addr_format, FI_FORMAT_UNSPEC, FI_ADDR_PSMX2, FI_ADDR_STR);
			goto err_out;
		}

		if (hints->ep_attr) {
			switch (hints->ep_attr->type) {
			case FI_EP_UNSPEC:
			case FI_EP_DGRAM:
			case FI_EP_RDM:
				break;
			default:
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->ep_attr->type=%d, supported=%d,%d,%d.\n",
					hints->ep_attr->type, FI_EP_UNSPEC,
					FI_EP_DGRAM, FI_EP_RDM);
				goto err_out;
			}

			switch (hints->ep_attr->protocol) {
			case FI_PROTO_UNSPEC:
			case FI_PROTO_PSMX2:
				break;
			default:
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->protocol=%d, supported=%d %d\n",
					hints->ep_attr->protocol,
					FI_PROTO_UNSPEC, FI_PROTO_PSMX2);
				goto err_out;
			}

			if (hints->ep_attr->tx_ctx_cnt > psmx2_env.sep_trx_ctxt &&
			    hints->ep_attr->tx_ctx_cnt != FI_SHARED_CONTEXT) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->ep_attr->tx_ctx_cnt=%"PRIu64", available=%d\n",
					hints->ep_attr->tx_ctx_cnt,
					psmx2_env.sep_trx_ctxt);
				goto err_out;
			}
			tx_ctx_cnt = hints->ep_attr->tx_ctx_cnt;

			if (hints->ep_attr->rx_ctx_cnt > psmx2_env.sep_trx_ctxt) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->ep_attr->rx_ctx_cnt=%"PRIu64", available=%d\n",
					hints->ep_attr->rx_ctx_cnt,
					psmx2_env.sep_trx_ctxt);
				goto err_out;
			}
			rx_ctx_cnt = hints->ep_attr->rx_ctx_cnt;
		}

		if ((hints->caps & PSMX2_CAPS) != hints->caps) {
			uint64_t psmx2_caps = PSMX2_CAPS;
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"hints->caps=%s,\n supported=%s\n",
				fi_tostr(&hints->caps, FI_TYPE_CAPS),
				fi_tostr(&psmx2_caps, FI_TYPE_CAPS));
			goto err_out;
		}

		if (hints->tx_attr) {
			if ((hints->tx_attr->op_flags & PSMX2_OP_FLAGS) !=
			    hints->tx_attr->op_flags) {
				uint64_t psmx2_op_flags = PSMX2_OP_FLAGS;
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->caps=%s,\n supported=%s\n",
					fi_tostr(&hints->tx_attr->op_flags, FI_TYPE_OP_FLAGS),
					fi_tostr(&psmx2_op_flags, FI_TYPE_OP_FLAGS));
				goto err_out;
			}
			if (hints->tx_attr->inject_size > PSMX2_INJECT_SIZE) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->inject_size=%"PRIu64
					", supported=%d.\n",
					hints->tx_attr->inject_size,
					PSMX2_INJECT_SIZE);
				goto err_out;
			}
		}

		if (hints->rx_attr &&
		    (hints->rx_attr->op_flags & PSMX2_OP_FLAGS) !=
		     hints->rx_attr->op_flags) {
			uint64_t psmx2_op_flags = PSMX2_OP_FLAGS;
			FI_INFO(&psmx2_prov, FI_LOG_CORE, "hints->rx->flags=%s,\n"
				" supported=%s.\n",
				fi_tostr(&hints->rx_attr->op_flags,
					 FI_TYPE_OP_FLAGS),
				fi_tostr(&psmx2_op_flags,
					 FI_TYPE_OP_FLAGS));
			goto err_out;
		}

		if ((hints->caps & FI_TAGGED) || (hints->caps & FI_MSG)) {
			if ((hints->mode & FI_CONTEXT) != FI_CONTEXT) {
				uint64_t psmx2_mode = FI_CONTEXT;
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->mode=%s,\n required=%s.\n",
					fi_tostr(&hints->mode, FI_TYPE_MODE),
					fi_tostr(&psmx2_mode, FI_TYPE_MODE));
				goto err_out;
			}
		} else {
			mode = 0;
		}

		if (hints->fabric_attr && hints->fabric_attr->name &&
		    strcmp(hints->fabric_attr->name, PSMX2_FABRIC_NAME)) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"hints->fabric_name=%s, supported=psm\n",
				hints->fabric_attr->name);
			goto err_out;
		}

		if (hints->domain_attr) {
			if (hints->domain_attr->name &&
			    strcmp(hints->domain_attr->name, PSMX2_DOMAIN_NAME)) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->domain_name=%s, supported=psm\n",
					hints->domain_attr->name);
				goto err_out;
			}

			switch (hints->domain_attr->av_type) {
			case FI_AV_UNSPEC:
			case FI_AV_MAP:
			case FI_AV_TABLE:
				av_type = hints->domain_attr->av_type;
				break;
			default:
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->domain_attr->av_type=%d, supported=%d %d %d\n",
					hints->domain_attr->av_type, FI_AV_UNSPEC, FI_AV_MAP,
					FI_AV_TABLE);
				goto err_out;
			}

			if (hints->domain_attr->mr_mode == FI_MR_BASIC) {
				mr_mode = FI_MR_BASIC;
			} else if (hints->domain_attr->mr_mode == FI_MR_SCALABLE) {
				mr_mode = FI_MR_SCALABLE;
			} else if (hints->domain_attr->mr_mode & (FI_MR_BASIC | FI_MR_SCALABLE)) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->domain_attr->mr_mode has FI_MR_BASIC or FI_MR_SCALABLE "
					"combined with other bits\n");
				goto err_out;
			}

			switch (hints->domain_attr->threading) {
			case FI_THREAD_UNSPEC:
				break;
			case FI_THREAD_FID:
			case FI_THREAD_ENDPOINT:
			case FI_THREAD_COMPLETION:
			case FI_THREAD_DOMAIN:
				threading = hints->domain_attr->threading;
				break;
			default:
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->domain_attr->threading=%d, supported=%d %d %d %d %d\n",
					hints->domain_attr->threading, FI_THREAD_UNSPEC,
					FI_THREAD_FID, FI_THREAD_ENDPOINT, FI_THREAD_COMPLETION,
					FI_THREAD_DOMAIN);
				goto err_out;
			}

			switch (hints->domain_attr->control_progress) {
			case FI_PROGRESS_UNSPEC:
				break;
			case FI_PROGRESS_MANUAL:
			case FI_PROGRESS_AUTO:
				control_progress = hints->domain_attr->control_progress;
				break;
			default:
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->domain_attr->control_progress=%d, supported=%d %d %d\n",
					hints->domain_attr->control_progress, FI_PROGRESS_UNSPEC,
					FI_PROGRESS_MANUAL, FI_PROGRESS_AUTO);
				goto err_out;
			}

			switch (hints->domain_attr->data_progress) {
			case FI_PROGRESS_UNSPEC:
				break;
			case FI_PROGRESS_MANUAL:
			case FI_PROGRESS_AUTO:
				data_progress = hints->domain_attr->data_progress;
				break;
			default:
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->domain_attr->data_progress=%d, supported=%d %d %d\n",
					hints->domain_attr->data_progress, FI_PROGRESS_UNSPEC,
					FI_PROGRESS_MANUAL, FI_PROGRESS_AUTO);
				goto err_out;
			}

			if (hints->domain_attr->caps & FI_SHARED_AV) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->domain_attr->caps=%lx, shared AV is unsupported\n",
					hints->domain_attr->caps);
			}
		}

		if (hints->ep_attr) {
			if (hints->ep_attr->max_msg_size > PSMX2_MAX_MSG_SIZE) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->ep_attr->max_msg_size=%"PRIu64","
					"supported=%llu.\n",
					hints->ep_attr->max_msg_size,
					PSMX2_MAX_MSG_SIZE);
				goto err_out;
			}
			max_tag_value = fi_tag_bits(hints->ep_attr->mem_tag_format);
		}

		if (hints->tx_attr) {
			if ((hints->tx_attr->msg_order & PSMX2_MSG_ORDER) !=
			    hints->tx_attr->msg_order) {
				uint64_t psmx2_msg_order = PSMX2_MSG_ORDER;
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->msg_order=%s,\n"
					"supported=%s.\n",
					fi_tostr(&hints->tx_attr->msg_order, FI_TYPE_MSG_ORDER),
					fi_tostr(&psmx2_msg_order, FI_TYPE_MSG_ORDER));
				goto err_out;
			}
			if ((hints->tx_attr->comp_order & PSMX2_COMP_ORDER) !=
			    hints->tx_attr->comp_order) {
				uint64_t psmx2_comp_order = PSMX2_COMP_ORDER;
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->msg_order=%s,\n"
					"supported=%s.\n",
					fi_tostr(&hints->tx_attr->comp_order, FI_TYPE_MSG_ORDER),
					fi_tostr(&psmx2_comp_order, FI_TYPE_MSG_ORDER));
				goto err_out;
			}
			if (hints->tx_attr->inject_size > PSMX2_INJECT_SIZE) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->inject_size=%ld,"
					"supported=%d.\n",
					hints->tx_attr->inject_size,
					PSMX2_INJECT_SIZE);
				goto err_out;
			}
			if (hints->tx_attr->iov_limit > PSMX2_IOV_MAX_COUNT) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->iov_limit=%"PRIu64
					", supported=%"PRIu64".\n",
					hints->tx_attr->iov_limit,
					PSMX2_IOV_MAX_COUNT);
				goto err_out;
			}
			if (hints->tx_attr->rma_iov_limit > 1) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->rma_iov_limit=%ld,"
					"supported=1.\n",
					hints->tx_attr->rma_iov_limit);
				goto err_out;
			}
		}

		if (hints->rx_attr) {
			if ((hints->rx_attr->msg_order & PSMX2_MSG_ORDER) !=
			    hints->rx_attr->msg_order) {
				uint64_t psmx2_msg_order = PSMX2_MSG_ORDER;
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->rx_attr->msg_order=%s,\n supported=%s.\n",
					fi_tostr(&hints->rx_attr->msg_order, FI_TYPE_MSG_ORDER),
					fi_tostr(&psmx2_msg_order, FI_TYPE_MSG_ORDER));
				goto err_out;
			}
			if ((hints->rx_attr->comp_order & PSMX2_COMP_ORDER) !=
			    hints->rx_attr->comp_order) {
				uint64_t psmx2_comp_order = PSMX2_COMP_ORDER;
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->rx_attr->comp_order=%s,\n supported=%s.\n",
					fi_tostr(&hints->rx_attr->comp_order, FI_TYPE_MSG_ORDER),
					fi_tostr(&psmx2_comp_order, FI_TYPE_MSG_ORDER));
				goto err_out;
			}
			if (hints->rx_attr->iov_limit > 1) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->rx_attr->iov_limit=%ld,"
					"supported=1.\n",
					hints->rx_attr->iov_limit);
				goto err_out;
			}
		}

		if (hints->caps)
			caps = hints->caps;

		/* TODO: check other fields of hints */
	}

	psmx2_info = fi_allocinfo();
	if (!psmx2_info) {
		err = -FI_ENOMEM;
		goto err_out;
	}

	psmx2_info->ep_attr->type = ep_type;
	psmx2_info->ep_attr->protocol = FI_PROTO_PSMX2;
	psmx2_info->ep_attr->protocol_version = PSM2_VERNO;
	psmx2_info->ep_attr->max_msg_size = PSMX2_MAX_MSG_SIZE;
	psmx2_info->ep_attr->mem_tag_format = fi_tag_format(max_tag_value);
	psmx2_info->ep_attr->tx_ctx_cnt = tx_ctx_cnt;
	psmx2_info->ep_attr->rx_ctx_cnt = rx_ctx_cnt;

	psmx2_info->domain_attr->threading = threading;
	psmx2_info->domain_attr->control_progress = control_progress;
	psmx2_info->domain_attr->data_progress = data_progress;
	psmx2_info->domain_attr->name = strdup(PSMX2_DOMAIN_NAME);
	psmx2_info->domain_attr->resource_mgmt = FI_RM_ENABLED;
	psmx2_info->domain_attr->av_type = av_type;
	psmx2_info->domain_attr->mr_mode = mr_mode;
	psmx2_info->domain_attr->mr_key_size = sizeof(uint64_t);
	psmx2_info->domain_attr->cq_data_size = 4;
	psmx2_info->domain_attr->cq_cnt = 65535;
	psmx2_info->domain_attr->ep_cnt = 65535;
	psmx2_info->domain_attr->tx_ctx_cnt = psmx2_env.sep_trx_ctxt;
	psmx2_info->domain_attr->rx_ctx_cnt = psmx2_env.sep_trx_ctxt;
	psmx2_info->domain_attr->max_ep_tx_ctx = psmx2_env.sep_trx_ctxt;
	psmx2_info->domain_attr->max_ep_rx_ctx = psmx2_env.sep_trx_ctxt;
	psmx2_info->domain_attr->max_ep_stx_ctx = 65535;
	psmx2_info->domain_attr->max_ep_srx_ctx = 0;
	psmx2_info->domain_attr->cntr_cnt = 65535;
	psmx2_info->domain_attr->mr_iov_limit = 65535;
	psmx2_info->domain_attr->caps = PSMX2_DOM_CAPS;
	psmx2_info->domain_attr->mode = 0;
	psmx2_info->domain_attr->mr_cnt = 65535;

	psmx2_info->next = NULL;
	psmx2_info->caps = caps;
	psmx2_info->mode = mode;
	psmx2_info->addr_format = addr_format;
	if (addr_format == FI_ADDR_STR) {
		psmx2_info->src_addr = psmx2_ep_name_to_string(src_addr, &len);
		psmx2_info->src_addrlen = len;
		free(src_addr);
		psmx2_info->dest_addr = psmx2_ep_name_to_string(dest_addr, &len);
		psmx2_info->dest_addrlen = len;
		free(dest_addr);
	} else {
		psmx2_info->src_addr = src_addr;
		psmx2_info->src_addrlen = sizeof(*src_addr);
		psmx2_info->dest_addr = dest_addr;
		psmx2_info->dest_addrlen = sizeof(*dest_addr);
	}
	psmx2_info->fabric_attr->name = strdup(PSMX2_FABRIC_NAME);
	psmx2_info->fabric_attr->prov_name = NULL;
	psmx2_info->fabric_attr->prov_version = PSMX2_VERSION;

	psmx2_info->tx_attr->caps = psmx2_info->caps;
	psmx2_info->tx_attr->mode = psmx2_info->mode;
	psmx2_info->tx_attr->op_flags = (hints && hints->tx_attr && hints->tx_attr->op_flags)
					? hints->tx_attr->op_flags : 0;
	psmx2_info->tx_attr->msg_order = PSMX2_MSG_ORDER;
	psmx2_info->tx_attr->comp_order = PSMX2_COMP_ORDER;
	psmx2_info->tx_attr->inject_size = PSMX2_INJECT_SIZE;
	psmx2_info->tx_attr->size = UINT64_MAX;
	psmx2_info->tx_attr->iov_limit = PSMX2_IOV_MAX_COUNT;
	psmx2_info->tx_attr->rma_iov_limit = 1;

	psmx2_info->rx_attr->caps = psmx2_info->caps;
	psmx2_info->rx_attr->mode = psmx2_info->mode;
	psmx2_info->rx_attr->op_flags = (hints && hints->rx_attr && hints->rx_attr->op_flags)
					? hints->rx_attr->op_flags : 0;
	psmx2_info->rx_attr->msg_order = PSMX2_MSG_ORDER;
	psmx2_info->rx_attr->comp_order = PSMX2_COMP_ORDER;
	psmx2_info->rx_attr->total_buffered_recv = ~(0ULL); /* that's how PSM handles it internally! */
	psmx2_info->rx_attr->size = UINT64_MAX;
	psmx2_info->rx_attr->iov_limit = 1;

	*info = psmx2_info;
	return 0;

err_out:
	free(dest_addr);
	free(src_addr);

	return err;
}

static void psmx2_fini(void)
{
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	if (! --psmx2_init_count && psmx2_lib_initialized) {
		/* This function is called from a library destructor, which is called
		 * automatically when exit() is called. The call to psm2_finalize()
		 * might cause deadlock if the applicaiton is terminated with Ctrl-C
		 * -- the application could be inside a PSM call, holding a lock that
		 * psm2_finalize() tries to acquire. This can be avoided by only
		 * calling psm2_finalize() when PSM is guaranteed to be unused.
		 */
		if (psmx2_active_fabric) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"psmx2_active_fabric != NULL, skip psm2_finalize\n");
		} else {
			psm2_finalize();
			psmx2_lib_initialized = 0;
		}
	}
}

struct fi_provider psmx2_prov = {
	.name = PSMX2_PROV_NAME,
	.version = PSMX2_VERSION,
	.fi_version = PSMX2_VERSION,
	.getinfo = psmx2_getinfo,
	.fabric = psmx2_fabric,
	.cleanup = psmx2_fini
};

PROVIDER_INI
{
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	fi_param_define(&psmx2_prov, "name_server", FI_PARAM_BOOL,
			"Whether to turn on the name server or not "
			"(default: yes)");

	fi_param_define(&psmx2_prov, "tagged_rma", FI_PARAM_BOOL,
			"Whether to use tagged messages for large size "
			"RMA or not (default: yes)");

	fi_param_define(&psmx2_prov, "uuid", FI_PARAM_STRING,
			"Unique Job ID required by the fabric");

	fi_param_define(&psmx2_prov, "delay", FI_PARAM_INT,
			"Delay (seconds) before finalization (for debugging)");

	fi_param_define(&psmx2_prov, "timeout", FI_PARAM_INT,
			"Timeout (seconds) for gracefully closing the PSM endpoint");

	fi_param_define(&psmx2_prov, "prog_interval", FI_PARAM_INT,
			"Interval (microseconds) between progress calls made in the "
			"progress thread (default: 1 if affinity is set, 1000 if not)");

	fi_param_define(&psmx2_prov, "prog_affinity", FI_PARAM_STRING,
			"When set, specify the set of CPU cores to set the progress "
			"thread affinity to. The format is "
			"<start>[:<end>[:<stride>]][,<start>[:<end>[:<stride>]]]*, "
			"where each triplet <start>:<end>:<stride> defines a block "
			"of core_ids. Both <start> and <end> can be either the core_id "
			"(when >=0) or core_id - num_cores (when <0). "
			"(default: affinity not set)");

	fi_param_define(&psmx2_prov, "inject_size", FI_PARAM_INT,
			"Maximum message size for fi_inject and fi_tinject (default: 64).");

	fi_param_define(&psmx2_prov, "lock_level", FI_PARAM_INT,
			"How internal locking is used. 0 means no locking. (default: 2).");

	pthread_mutex_init(&psmx2_lib_mutex, NULL);
	psmx2_init_count++;
	return (&psmx2_prov);
}

