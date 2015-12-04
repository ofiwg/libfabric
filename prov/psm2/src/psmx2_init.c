/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

static int psmx2_init_count = 0;

struct psmx2_env psmx2_env = {
	.name_server	= 1,
	.am_msg		= 0,
	.tagged_rma	= 1,
	.uuid		= PSMX2_DEFAULT_UUID,
	.delay		= 1,
	.timeout	= 5,
	.prog_interval	= -1,
	.prog_affinity	= NULL,
};

static void psmx2_init_env(void)
{
	if (getenv("OMPI_COMM_WORLD_RANK") || getenv("PMI_RANK"))
		psmx2_env.name_server = 0;

	fi_param_get_bool(&psmx2_prov, "name_server", &psmx2_env.name_server);
	fi_param_get_bool(&psmx2_prov, "am_msg", &psmx2_env.am_msg);
	fi_param_get_bool(&psmx2_prov, "tagged_rma", &psmx2_env.tagged_rma);
	fi_param_get_str(&psmx2_prov, "uuid", &psmx2_env.uuid);
	fi_param_get_int(&psmx2_prov, "delay", &psmx2_env.delay);
	fi_param_get_int(&psmx2_prov, "timeout", &psmx2_env.timeout);
	fi_param_get_int(&psmx2_prov, "prog_interval", &psmx2_env.prog_interval);
	fi_param_get_str(&psmx2_prov, "prog_affinity", &psmx2_env.prog_affinity);
}

static int psmx2_getinfo(uint32_t version, const char *node,
			 const char *service, uint64_t flags,
			 struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *psmx2_info;
	uint32_t cnt = 0;
	void *dest_addr = NULL;
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

	FI_INFO(&psmx2_prov, FI_LOG_CORE,"\n");

	*info = NULL;

	if (psm2_ep_num_devunits(&cnt) || !cnt) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"no PSM device is found.\n");
		return -FI_ENODATA;
	}

	psmx2_init_env();

	if (node && !(flags & FI_SOURCE)) {
		dest_addr = psmx2_resolve_name(node, 0);
		if (dest_addr)
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"node '%s' resolved to <epid=0x%llx, vl=%d>\n", node,
				((struct psmx2_ep_name *)dest_addr)->epid,
				((struct psmx2_ep_name *)dest_addr)->vlane);
		else
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"failed to resolve node '%s'.\n", node);
	}

	if (hints) {
		switch (hints->addr_format) {
		case FI_FORMAT_UNSPEC:
		case FI_ADDR_PSMX:
			break;
		default:
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"hints->addr_format=%d, supported=%d,%d.\n",
				hints->addr_format, FI_FORMAT_UNSPEC, FI_ADDR_PSMX);
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
			case FI_PROTO_PSMX:
				break;
			default:
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->protocol=%d, supported=%d %d\n",
					hints->ep_attr->protocol,
					FI_PROTO_UNSPEC, FI_PROTO_PSMX);
				goto err_out;
			}

			if (hints->ep_attr->tx_ctx_cnt > 1) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->ep_attr->tx_ctx_cnt=%d, supported=0,1\n",
					hints->ep_attr->tx_ctx_cnt);
				goto err_out;
			}

			if (hints->ep_attr->rx_ctx_cnt > 1) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->ep_attr->rx_ctx_cnt=%d, supported=0,1\n",
					hints->ep_attr->rx_ctx_cnt);
				goto err_out;
			}
		}

		if ((hints->caps & PSMX2_CAPS) != hints->caps) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"hints->caps=0x%llx, supported=0x%llx\n",
				hints->caps, PSMX2_CAPS);
			goto err_out;
		}

		if (hints->tx_attr) {
			if ((hints->tx_attr->op_flags & PSMX2_OP_FLAGS) !=
			    hints->tx_attr->op_flags) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx->flags=0x%llx, "
					"supported=0x%llx\n",
					hints->tx_attr->op_flags,
					PSMX2_OP_FLAGS);
				goto err_out;
			}
			if (hints->tx_attr->inject_size > PSMX2_INJECT_SIZE) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->inject_size=%ld,"
					"supported=%ld.\n",
					hints->tx_attr->inject_size,
					PSMX2_INJECT_SIZE);
				goto err_out;
			}
		}

		if (hints->rx_attr &&
		    (hints->rx_attr->op_flags & PSMX2_OP_FLAGS) !=
		     hints->rx_attr->op_flags) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"hints->rx->flags=0x%llx, supported=0x%llx\n",
				hints->rx_attr->op_flags, PSMX2_OP_FLAGS);
			goto err_out;
		}

		if ((hints->caps & FI_TAGGED) ||
		    ((hints->caps & FI_MSG) && !psmx2_env.am_msg)) {
			if ((hints->mode & FI_CONTEXT) != FI_CONTEXT) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->mode=0x%llx, required=0x%llx\n",
					hints->mode, FI_CONTEXT);
				goto err_out;
			}
		}
		else {
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

			switch (hints->domain_attr->mr_mode) {
			case FI_MR_UNSPEC:
				break;
			case FI_MR_BASIC:
			case FI_MR_SCALABLE:
				mr_mode = hints->domain_attr->mr_mode;
				break;
			default:
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->domain_attr->mr_mode=%d, supported=%d %d %d\n",
					hints->domain_attr->mr_mode, FI_MR_UNSPEC, FI_MR_BASIC,
					FI_MR_SCALABLE);
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
		}

		if (hints->ep_attr) {
			if (hints->ep_attr->max_msg_size > PSMX2_MAX_MSG_SIZE) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->ep_attr->max_msg_size=%ld,"
					"supported=%ld.\n",
					hints->ep_attr->max_msg_size,
					PSMX2_MAX_MSG_SIZE);
				goto err_out;
			}
			max_tag_value = fi_tag_bits(hints->ep_attr->mem_tag_format);
		}

		if (hints->tx_attr) {
			if ((hints->tx_attr->msg_order & PSMX2_MSG_ORDER) !=
			    hints->tx_attr->msg_order) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->msg_order=%lx,"
					"supported=%lx.\n",
					hints->tx_attr->msg_order,
					PSMX2_MSG_ORDER);
				goto err_out;
			}
			if ((hints->tx_attr->comp_order & PSMX2_COMP_ORDER) !=
			    hints->tx_attr->comp_order) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->msg_order=%lx,"
					"supported=%lx.\n",
					hints->tx_attr->comp_order,
					PSMX2_COMP_ORDER);
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
			if (hints->tx_attr->iov_limit > 1) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->tx_attr->iov_limit=%ld,"
					"supported=1.\n",
					hints->tx_attr->iov_limit);
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
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->rx_attr->msg_order=%lx,"
					"supported=%lx.\n",
					hints->rx_attr->msg_order,
					PSMX2_MSG_ORDER);
				goto err_out;
			}
			if ((hints->rx_attr->comp_order & PSMX2_COMP_ORDER) !=
			    hints->rx_attr->comp_order) {
				FI_INFO(&psmx2_prov, FI_LOG_CORE,
					"hints->rx_attr->msg_order=%lx,"
					"supported=%lx.\n",
					hints->rx_attr->comp_order,
					PSMX2_COMP_ORDER);
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
	psmx2_info->ep_attr->protocol = FI_PROTO_PSMX;
	psmx2_info->ep_attr->protocol_version = PSM2_VERNO;
	psmx2_info->ep_attr->max_msg_size = PSMX2_MAX_MSG_SIZE;
	psmx2_info->ep_attr->mem_tag_format = fi_tag_format(max_tag_value);
	psmx2_info->ep_attr->tx_ctx_cnt = 1;
	psmx2_info->ep_attr->rx_ctx_cnt = 1;

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
	psmx2_info->domain_attr->tx_ctx_cnt = 1;
	psmx2_info->domain_attr->rx_ctx_cnt = 1;
	psmx2_info->domain_attr->max_ep_tx_ctx = 65535;
	psmx2_info->domain_attr->max_ep_rx_ctx = 1;
	psmx2_info->domain_attr->max_ep_stx_ctx = 65535;
	psmx2_info->domain_attr->max_ep_srx_ctx = 0;

	psmx2_info->next = NULL;
	psmx2_info->caps = caps;
	psmx2_info->mode = mode;
	psmx2_info->addr_format = FI_ADDR_PSMX;
	psmx2_info->src_addrlen = 0;
	psmx2_info->dest_addrlen = sizeof(struct psmx2_ep_name);
	psmx2_info->src_addr = NULL;
	psmx2_info->dest_addr = dest_addr;
	psmx2_info->fabric_attr->name = strdup(PSMX2_FABRIC_NAME);
	psmx2_info->fabric_attr->prov_name = NULL;

	psmx2_info->tx_attr->caps = psmx2_info->caps;
	psmx2_info->tx_attr->mode = psmx2_info->mode;
	psmx2_info->tx_attr->op_flags = (hints && hints->tx_attr && hints->tx_attr->op_flags)
					? hints->tx_attr->op_flags : 0;
	psmx2_info->tx_attr->msg_order = PSMX2_MSG_ORDER;
	psmx2_info->tx_attr->comp_order = PSMX2_COMP_ORDER;
	psmx2_info->tx_attr->inject_size = PSMX2_INJECT_SIZE;
	psmx2_info->tx_attr->size = UINT64_MAX;
	psmx2_info->tx_attr->iov_limit = 1;
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
	if (dest_addr)
		free(dest_addr);

	return err;
}

static void psmx2_fini(void)
{
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	if (! --psmx2_init_count) {
		/* This function is called from a library destructor, which is called
		 * automatically when exit() is called. The call to psm2_finalize()
		 * might cause deadlock if the applicaiton is terminated with Ctrl-C
		 * -- the application could be inside a PSM call, holding a lock that
		 * psm2_finalize() tries to acquire. This can be avoided by only
		 * calling psm2_finalize() when PSM is guaranteed to be unused.
		 */
		if (psmx2_active_fabric)
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"psmx2_active_fabric != NULL, skip psm2_finalize\n");
		else
			psm2_finalize();
	}
}

struct fi_provider psmx2_prov = {
	.name = PSMX2_PROV_NAME,
	.version = FI_VERSION(0, 9),
	.fi_version = FI_VERSION(1, 1),
	.getinfo = psmx2_getinfo,
	.fabric = psmx2_fabric,
	.cleanup = psmx2_fini
};

PROVIDER_INI
{
	int major, minor;
	int err;

	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	fi_param_define(&psmx2_prov, "name_server", FI_PARAM_BOOL,
			"Whether to turn on the name server or not "
			"(default: yes)");

	fi_param_define(&psmx2_prov, "am_msg", FI_PARAM_BOOL,
			"Whether to use active message based messaging "
			"or not (default: no)");

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

        psm2_error_register_handler(NULL, PSM2_ERRHANDLER_NO_HANDLER);

	major = PSM2_VERNO_MAJOR;
	minor = PSM2_VERNO_MINOR;

        err = psm2_init(&major, &minor);
	if (err != PSM2_OK) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"psm2_init failed: %s\n", psm2_error_get_string(err));
		return NULL;
	}

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"PSM header version = (%d, %d)\n", PSM2_VERNO_MAJOR, PSM2_VERNO_MINOR);
	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"PSM library version = (%d, %d)\n", major, minor);

	psmx2_init_count++;
	return (&psmx2_prov);
}

