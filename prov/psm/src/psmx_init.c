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

#include "psmx.h"
#include "fi.h"
#include "prov.h"

volatile int psmx_init_count = 0;
struct psmx_fid_fabric *psmx_active_fabric = NULL;

struct psmx_env psmx_env = {
	.name_server	= 1,
	.am_msg		= 0,
	.tagged_rma	= 1,
	.uuid		= PSMX_DEFAULT_UUID,
	.delay		= 1,
	.timeout	= 5,
	.prog_interval	= -1,
	.prog_affinity	= NULL,
};

static void psmx_init_env(void)
{
	if (getenv("OMPI_COMM_WORLD_RANK") || getenv("PMI_RANK"))
		psmx_env.name_server = 0;

	fi_param_get_bool(&psmx_prov, "name_server", &psmx_env.name_server);
	fi_param_get_bool(&psmx_prov, "am_msg", &psmx_env.am_msg);
	fi_param_get_bool(&psmx_prov, "tagged_rma", &psmx_env.tagged_rma);
	fi_param_get_str(&psmx_prov, "uuid", &psmx_env.uuid);
	fi_param_get_int(&psmx_prov, "delay", &psmx_env.delay);
	fi_param_get_int(&psmx_prov, "timeout", &psmx_env.timeout);
	fi_param_get_int(&psmx_prov, "prog_interval", &psmx_env.prog_interval);
	fi_param_get_str(&psmx_prov, "prog_affinity", &psmx_env.prog_affinity);
}

static int psmx_reserve_tag_bits(int *caps, uint64_t *max_tag_value)
{
	uint64_t reserved_bits = 0;
	int ret_caps;
	int ask_caps = *caps;

	ret_caps = ask_caps ? ask_caps : PSMX_CAPS;

	if ((ret_caps & FI_MSG) && !psmx_env.am_msg) {
		if (*max_tag_value < PSMX_MSG_BIT) {
			reserved_bits |= PSMX_MSG_BIT;
		}
		else if (ask_caps) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"unable to reserve tag bit for FI_MSG support.\n"
				"ADVICE: please reduce the asked max_tag_value, "
				"or remove FI_MSG from the asked capabilities, "
				"or set FI_PSM_AM_MSG=1 to use an alternative (but "
				"less optimized) message queue implementation.\n");
			return -1;
		}
		else {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"unable to reserve tag bit for FI_MSG support. "
				"FI_MSG is removed from the capabilities.\n"
				"ADVICE: please reduce the asked max_tag_value, "
				"or set FI_PSM_AM_MSG=1 to use an alternative (but "
				"less optimized) message queue implementation.\n");
			ret_caps &= ~FI_MSG;
		}
	}

	if ((ret_caps & FI_RMA) && psmx_env.tagged_rma) {
		if (*max_tag_value < PSMX_RMA_BIT) {
			reserved_bits |= PSMX_RMA_BIT;
		}
		else if (ask_caps) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"unable to reserve tag bit for tagged RMA acceleration.\n"
				"ADVICE: please reduce the asked max_tag_value, or "
				"remove FI_RMA from the asked capabilities, or set "
				"FI_PSM_TAGGED_RMA=0 to disable RMA acceleration.\n");
			return -1;
		}
		else {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"unable to reserve tag bit for tagged RMA acceleration. "
				"FI_RMA is removed from the capabilities.\n"
				"ADVICE: please reduce the asked max_tag_value, or "
				"set FI_PSM_TAGGED_RMA=0 to disable RMA acceleration.\n");
			ret_caps &= ~FI_RMA;
		}
	}

	reserved_bits |= (reserved_bits << 1);

	*caps = ret_caps;
	*max_tag_value = ~reserved_bits;
	return 0;
}

static int psmx_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *psmx_info;
	uint32_t cnt = 0;
	void *dest_addr = NULL;
	int ep_type = FI_EP_RDM;
	int av_type = FI_AV_UNSPEC;
	uint64_t mode = FI_CONTEXT;
	enum fi_mr_mode mr_mode = FI_MR_SCALABLE;
	enum fi_threading threading = FI_THREAD_COMPLETION;
	enum fi_progress control_progress = FI_PROGRESS_MANUAL;
	enum fi_progress data_progress = FI_PROGRESS_MANUAL;
	int caps = 0;
	uint64_t max_tag_value = 0;
	int err = -FI_ENODATA;

	FI_INFO(&psmx_prov, FI_LOG_CORE,"\n");

	*info = NULL;

	if (psm_ep_num_devunits(&cnt) || !cnt) {
		FI_INFO(&psmx_prov, FI_LOG_CORE,
			"no PSM device is found.\n");
		return -FI_ENODATA;
	}

	psmx_init_env();

	if (node && !(flags & FI_SOURCE))
		dest_addr = psmx_resolve_name(node, 0);

	if (hints) {
		switch (hints->addr_format) {
		case FI_FORMAT_UNSPEC:
		case FI_ADDR_PSMX:
			break;
		default:
			FI_INFO(&psmx_prov, FI_LOG_CORE,
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
				FI_INFO(&psmx_prov, FI_LOG_CORE,
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
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->protocol=%d, supported=%d %d\n",
					hints->ep_attr->protocol,
					FI_PROTO_UNSPEC, FI_PROTO_PSMX);
				goto err_out;
			}

			if (hints->ep_attr->tx_ctx_cnt > 1) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->ep_attr->tx_ctx_cnt=%d, supported=0,1\n",
					hints->ep_attr->tx_ctx_cnt);
				goto err_out;
			}

			if (hints->ep_attr->rx_ctx_cnt > 1) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->ep_attr->rx_ctx_cnt=%d, supported=0,1\n",
					hints->ep_attr->rx_ctx_cnt);
				goto err_out;
			}
		}

		if ((hints->caps & PSMX_CAPS) != hints->caps &&
		    (hints->caps & PSMX_CAPS2) != hints->caps) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"hints->caps=0x%llx, supported=0x%llx,0x%llx\n",
				hints->caps, PSMX_CAPS, PSMX_CAPS2);
			goto err_out;
		}

		if (hints->tx_attr) {
			if ((hints->tx_attr->op_flags & PSMX_OP_FLAGS) !=
			    hints->tx_attr->op_flags) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->tx->flags=0x%llx, "
					"supported=0x%llx\n",
					hints->tx_attr->op_flags,
					PSMX_OP_FLAGS);
				goto err_out;
			}
			if (hints->tx_attr->inject_size > PSMX_INJECT_SIZE) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->tx_attr->inject_size=%ld,"
					"supported=%ld.\n",
					hints->tx_attr->inject_size,
					PSMX_INJECT_SIZE);
				goto err_out;
			}
		}

		if (hints->rx_attr &&
		    (hints->rx_attr->op_flags & PSMX_OP_FLAGS) !=
		     hints->rx_attr->op_flags) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"hints->rx->flags=0x%llx, supported=0x%llx\n",
				hints->rx_attr->op_flags, PSMX_OP_FLAGS);
			goto err_out;
		}

		if ((hints->caps & FI_TAGGED) ||
		    ((hints->caps & FI_MSG) && !psmx_env.am_msg)) {
			if ((hints->mode & FI_CONTEXT) != FI_CONTEXT) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->mode=0x%llx, required=0x%llx\n",
					hints->mode, FI_CONTEXT);
				goto err_out;
			}
		}
		else {
			mode = 0;
		}

		if (hints->fabric_attr && hints->fabric_attr->name &&
		    strcmp(hints->fabric_attr->name, PSMX_FABRIC_NAME)) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"hints->fabric_name=%s, supported=psm\n",
				hints->fabric_attr->name);
			goto err_out;
		}

		if (hints->domain_attr) {
			if (hints->domain_attr->name &&
			    strcmp(hints->domain_attr->name, PSMX_DOMAIN_NAME)) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
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
				FI_INFO(&psmx_prov, FI_LOG_CORE,
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
				FI_INFO(&psmx_prov, FI_LOG_CORE,
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
				FI_INFO(&psmx_prov, FI_LOG_CORE,
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
				FI_INFO(&psmx_prov, FI_LOG_CORE,
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
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->domain_attr->data_progress=%d, supported=%d %d %d\n",
					hints->domain_attr->data_progress, FI_PROGRESS_UNSPEC,
					FI_PROGRESS_MANUAL, FI_PROGRESS_AUTO);
				goto err_out;
			}
		}

		if (hints->ep_attr) {
			if (hints->ep_attr->max_msg_size > PSMX_MAX_MSG_SIZE) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->ep_attr->max_msg_size=%ld,"
					"supported=%ld.\n",
					hints->ep_attr->max_msg_size,
					PSMX_MAX_MSG_SIZE);
				goto err_out;
			}
			max_tag_value = fi_tag_bits(hints->ep_attr->mem_tag_format);
		}

		if (hints->tx_attr) {
			if ((hints->tx_attr->msg_order & PSMX_MSG_ORDER) !=
			    hints->tx_attr->msg_order) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->tx_attr->msg_order=%lx,"
					"supported=%lx.\n",
					hints->tx_attr->msg_order,
					PSMX_MSG_ORDER);
				goto err_out;
			}
			if ((hints->tx_attr->comp_order & PSMX_COMP_ORDER) !=
			    hints->tx_attr->comp_order) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->tx_attr->msg_order=%lx,"
					"supported=%lx.\n",
					hints->tx_attr->comp_order,
					PSMX_COMP_ORDER);
				goto err_out;
			}
			if (hints->tx_attr->inject_size > PSMX_INJECT_SIZE) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->tx_attr->inject_size=%ld,"
					"supported=%d.\n",
					hints->tx_attr->inject_size,
					PSMX_INJECT_SIZE);
				goto err_out;
			}
			if (hints->tx_attr->iov_limit > 1) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->tx_attr->iov_limit=%ld,"
					"supported=1.\n",
					hints->tx_attr->iov_limit);
				goto err_out;
			}
			if (hints->tx_attr->rma_iov_limit > 1) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->tx_attr->rma_iov_limit=%ld,"
					"supported=1.\n",
					hints->tx_attr->rma_iov_limit);
				goto err_out;
			}
		}

		if (hints->rx_attr) {
			if ((hints->rx_attr->msg_order & PSMX_MSG_ORDER) !=
			    hints->rx_attr->msg_order) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->rx_attr->msg_order=%lx,"
					"supported=%lx.\n",
					hints->rx_attr->msg_order,
					PSMX_MSG_ORDER);
				goto err_out;
			}
			if ((hints->rx_attr->comp_order & PSMX_COMP_ORDER) !=
			    hints->rx_attr->comp_order) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->rx_attr->msg_order=%lx,"
					"supported=%lx.\n",
					hints->rx_attr->comp_order,
					PSMX_COMP_ORDER);
				goto err_out;
			}
			if (hints->rx_attr->iov_limit > 1) {
				FI_INFO(&psmx_prov, FI_LOG_CORE,
					"hints->rx_attr->iov_limit=%ld,"
					"supported=1.\n",
					hints->rx_attr->iov_limit);
				goto err_out;
			}
		}

		caps = hints->caps;

		/* TODO: check other fields of hints */
	}

	if (psmx_reserve_tag_bits(&caps, &max_tag_value) < 0)
		goto err_out;

	psmx_info = fi_allocinfo();
	if (!psmx_info) {
		err = -FI_ENOMEM;
		goto err_out;
	}

	psmx_info->ep_attr->type = ep_type;
	psmx_info->ep_attr->protocol = FI_PROTO_PSMX;
	psmx_info->ep_attr->protocol_version = PSM_VERNO;
	psmx_info->ep_attr->max_msg_size = PSMX_MAX_MSG_SIZE;
	psmx_info->ep_attr->mem_tag_format = fi_tag_format(max_tag_value);
	psmx_info->ep_attr->tx_ctx_cnt = 1;
	psmx_info->ep_attr->rx_ctx_cnt = 1;

	psmx_info->domain_attr->threading = threading;
	psmx_info->domain_attr->control_progress = control_progress;
	psmx_info->domain_attr->data_progress = data_progress;
	psmx_info->domain_attr->name = strdup(PSMX_DOMAIN_NAME);
	psmx_info->domain_attr->resource_mgmt = FI_RM_ENABLED;
	psmx_info->domain_attr->av_type = av_type;
	psmx_info->domain_attr->mr_mode = mr_mode;
	psmx_info->domain_attr->mr_key_size = sizeof(uint64_t);
	psmx_info->domain_attr->cq_data_size = 4;
	psmx_info->domain_attr->cq_cnt = 65535;
	psmx_info->domain_attr->ep_cnt = 65535;
	psmx_info->domain_attr->tx_ctx_cnt = 1;
	psmx_info->domain_attr->rx_ctx_cnt = 1;
	psmx_info->domain_attr->max_ep_tx_ctx = 65535;
	psmx_info->domain_attr->max_ep_rx_ctx = 1;
	psmx_info->domain_attr->max_ep_stx_ctx = 65535;
	psmx_info->domain_attr->max_ep_srx_ctx = 0;

	psmx_info->next = NULL;
	psmx_info->caps = (hints && hints->caps) ? hints->caps : caps;
	psmx_info->mode = mode;
	psmx_info->addr_format = FI_ADDR_PSMX;
	psmx_info->src_addrlen = 0;
	psmx_info->dest_addrlen = sizeof(psm_epid_t);
	psmx_info->src_addr = NULL;
	psmx_info->dest_addr = dest_addr;
	psmx_info->fabric_attr->name = strdup(PSMX_FABRIC_NAME);
	psmx_info->fabric_attr->prov_name = NULL;

	psmx_info->tx_attr->caps = psmx_info->caps;
	psmx_info->tx_attr->mode = psmx_info->mode;
	psmx_info->tx_attr->op_flags = (hints && hints->tx_attr && hints->tx_attr->op_flags)
					? hints->tx_attr->op_flags : 0;
	psmx_info->tx_attr->msg_order = PSMX_MSG_ORDER;
	psmx_info->tx_attr->comp_order = PSMX_COMP_ORDER;
	psmx_info->tx_attr->inject_size = PSMX_INJECT_SIZE;
	psmx_info->tx_attr->size = UINT64_MAX;
	psmx_info->tx_attr->iov_limit = 1;
	psmx_info->tx_attr->rma_iov_limit = 1;

	psmx_info->rx_attr->caps = psmx_info->caps;
	psmx_info->rx_attr->mode = psmx_info->mode;
	psmx_info->rx_attr->op_flags = (hints && hints->rx_attr && hints->rx_attr->op_flags)
					? hints->rx_attr->op_flags : 0;
	psmx_info->rx_attr->msg_order = PSMX_MSG_ORDER;
	psmx_info->rx_attr->comp_order = PSMX_COMP_ORDER;
	psmx_info->rx_attr->total_buffered_recv = ~(0ULL); /* that's how PSM handles it internally! */
	psmx_info->rx_attr->size = UINT64_MAX;
	psmx_info->rx_attr->iov_limit = 1;

	*info = psmx_info;
	return 0;

err_out:
	if (dest_addr)
		free(dest_addr);

	return err;
}

void psmx_fabric_release(struct psmx_fid_fabric *fabric)
{
	void *exit_code;
	int ret;

	FI_INFO(&psmx_prov, FI_LOG_CORE, "refcnt=%d\n", fabric->refcnt);

	if (--fabric->refcnt)
		return;

	if (psmx_env.name_server &&
	    !pthread_equal(fabric->name_server_thread, pthread_self())) {
		ret = pthread_cancel(fabric->name_server_thread);
		if (ret) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"pthread_cancel returns %d\n", ret);
		}
		ret = pthread_join(fabric->name_server_thread, &exit_code);
		if (ret) {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"pthread_join returns %d\n", ret);
		}
		else {
			FI_INFO(&psmx_prov, FI_LOG_CORE,
				"name server thread exited with code %ld (%s)\n",
				(uintptr_t)exit_code,
				(exit_code == PTHREAD_CANCELED) ? "PTHREAD_CANCELED" : "?");
		}
	}
	if (fabric->active_domain) {
		FI_WARN(&psmx_prov, FI_LOG_CORE, "forced closing of active_domain\n");
		fi_close(&fabric->active_domain->domain.fid);
	}
	assert(fabric == psmx_active_fabric);
	psmx_active_fabric = NULL;
	free(fabric);
}

static int psmx_fabric_close(fid_t fid)
{
	struct psmx_fid_fabric *fabric;

	FI_INFO(&psmx_prov, FI_LOG_CORE, "\n");

	fabric = container_of(fid, struct psmx_fid_fabric, fabric.fid);

	psmx_fabric_release(fabric);

	return 0;
}

static struct fi_ops psmx_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_fabric_close,
};

static struct fi_ops_fabric psmx_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = psmx_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = psmx_eq_open,
	.wait_open = psmx_wait_open,
};

static int psmx_fabric(struct fi_fabric_attr *attr,
		       struct fid_fabric **fabric, void *context)
{
	struct psmx_fid_fabric *fabric_priv;
	int ret;

	FI_INFO(&psmx_prov, FI_LOG_CORE, "\n");

	if (strcmp(attr->name, PSMX_FABRIC_NAME))
		return -FI_ENODATA;

	if (psmx_active_fabric) {
		psmx_fabric_acquire(psmx_active_fabric);
		*fabric = &psmx_active_fabric->fabric;
		return 0;
	}

	fabric_priv = calloc(1, sizeof(*fabric_priv));
	if (!fabric_priv)
		return -FI_ENOMEM;

	fabric_priv->fabric.fid.fclass = FI_CLASS_FABRIC;
	fabric_priv->fabric.fid.context = context;
	fabric_priv->fabric.fid.ops = &psmx_fabric_fi_ops;
	fabric_priv->fabric.ops = &psmx_fabric_ops;

	psmx_get_uuid(fabric_priv->uuid);

	if (psmx_env.name_server) {
		ret = pthread_create(&fabric_priv->name_server_thread, NULL,
				     psmx_name_server, (void *)fabric_priv);
		if (ret) {
			FI_INFO(&psmx_prov, FI_LOG_CORE, "pthread_create returns %d\n", ret);
			/* use the main thread's ID as invalid value for the new thread */
			fabric_priv->name_server_thread = pthread_self();
		}
	}

	psmx_query_mpi();

	fabric_priv->refcnt = 1;
	*fabric = &fabric_priv->fabric;
	psmx_active_fabric = fabric_priv;

	return 0;
}

static void psmx_fini(void)
{
	FI_INFO(&psmx_prov, FI_LOG_CORE, "\n");

	if (! --psmx_init_count)
		psm_finalize();
}

struct fi_provider psmx_prov = {
	.name = PSMX_PROV_NAME,
	.version = FI_VERSION(0, 9),
	.fi_version = FI_VERSION(1, 1),
	.getinfo = psmx_getinfo,
	.fabric = psmx_fabric,
	.cleanup = psmx_fini
};

PROVIDER_INI
{
	int major, minor;
	int err;

	FI_INFO(&psmx_prov, FI_LOG_CORE, "\n");

	fi_param_define(&psmx_prov, "name_server", FI_PARAM_BOOL,
			"Whether to turn on the name server or not "
			"(default: yes)");

	fi_param_define(&psmx_prov, "am_msg", FI_PARAM_BOOL,
			"Whether to use active message based messaging "
			"or not (default: no)");

	fi_param_define(&psmx_prov, "tagged_rma", FI_PARAM_BOOL,
			"Whether to use tagged messages for large size "
			"RMA or not (default: yes)");

	fi_param_define(&psmx_prov, "uuid", FI_PARAM_STRING,
			"Unique Job ID required by the fabric");

	fi_param_define(&psmx_prov, "delay", FI_PARAM_INT,
			"Delay (seconds) before finalization (for debugging)");

	fi_param_define(&psmx_prov, "timeout", FI_PARAM_INT,
			"Timeout (seconds) for gracefully closing the PSM endpoint");

	fi_param_define(&psmx_prov, "prog_interval", FI_PARAM_INT,
			"Interval (microseconds) between progress calls made in the "
			"progress thread (default: 1 if affinity is set, 1000 if not)");

	fi_param_define(&psmx_prov, "prog_affinity", FI_PARAM_STRING,
			"When set, specify the set of CPU cores to set the progress "
			"thread affinity to. The format is "
			"<start>[:<end>[:<stride>]][,<start>[:<end>[:<stride>]]]*, "
			"where each triplet <start>:<end>:<stride> defines a block "
			"of core_ids. Both <start> and <end> can be either the core_id "
			"(when >=0) or core_id - num_cores (when <0). "
			"(default: affinity not set)");

        psm_error_register_handler(NULL, PSM_ERRHANDLER_NO_HANDLER);

	major = PSM_VERNO_MAJOR;
	minor = PSM_VERNO_MINOR;

        err = psm_init(&major, &minor);
	if (err != PSM_OK) {
		FI_WARN(&psmx_prov, FI_LOG_CORE,
			"psm_init failed: %s\n", psm_error_get_string(err));
		return NULL;
	}

	FI_INFO(&psmx_prov, FI_LOG_CORE,
		"PSM header version = (%d, %d)\n", PSM_VERNO_MAJOR, PSM_VERNO_MINOR);
	FI_INFO(&psmx_prov, FI_LOG_CORE,
		"PSM library version = (%d, %d)\n", major, minor);

	if (major != PSM_VERNO_MAJOR) {
		FI_WARN(&psmx_prov, FI_LOG_CORE,
			"PSM version mismatch: header %d.%d, library %d.%d.\n",
			PSM_VERNO_MAJOR, PSM_VERNO_MINOR, major, minor);
		return NULL;
	}

	psmx_init_count++;
	return (&psmx_prov);
}

