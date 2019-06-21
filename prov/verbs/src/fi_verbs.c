/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include "config.h"

#include <ofi_mem.h>

#include "fi_verbs.h"

static void fi_ibv_fini(void);

static const char *local_node = "localhost";

#define VERBS_DEFAULT_MIN_RNR_TIMER 12

struct fi_ibv_gl_data fi_ibv_gl_data = {
	.def_tx_size		= 384,
	.def_rx_size		= 384,
	.def_tx_iov_limit	= 4,
	.def_rx_iov_limit	= 4,
	.def_inline_size	= 256,
	.min_rnr_timer		= VERBS_DEFAULT_MIN_RNR_TIMER,
	/* Disable by default. Because this feature may corrupt
	 * data due to IBV_EXP_ACCESS_RELAXED flag. But usage
	 * this feature w/o this flag leads to poor bandwidth */
	.use_odp		= 0,
	.cqread_bunch_size	= 8,
	.iface			= NULL,
	.gid_idx		= 0,
	.dgram			= {
		.use_name_server	= 1,
		.name_server_port	= 5678,
	},

	.msg			= {
		/* Disabled by default. Use XRC transport for message
		 * endpoint only if it is explicitly requested */
		.prefer_xrc		= 0,
		.xrcd_filename		= "/tmp/verbs_xrcd",
	},
};

struct fi_ibv_dev_preset {
	int		max_inline_data;
	const char	*dev_name_prefix;
} verbs_dev_presets[] = {
	{
		.max_inline_data = 48,
		.dev_name_prefix = "i40iw",
	},
};

struct fi_provider fi_ibv_prov = {
	.name = VERBS_PROV_NAME,
	.version = VERBS_PROV_VERS,
	.fi_version = FI_VERSION(1, 8),
	.getinfo = fi_ibv_getinfo,
	.fabric = fi_ibv_fabric,
	.cleanup = fi_ibv_fini
};

struct util_prov fi_ibv_util_prov = {
	.prov = &fi_ibv_prov,
	.info = NULL,
	/* The support of the shared recieve contexts
	 * is dynamically calculated */
	.flags = 0,
};

int fi_ibv_sockaddr_len(struct sockaddr *addr)
{
	if (addr->sa_family == AF_IB)
		return sizeof(struct sockaddr_ib);
	else
		return ofi_sizeofaddr(addr);
}

int fi_ibv_get_rdma_rai(const char *node, const char *service, uint64_t flags,
		   const struct fi_info *hints, struct rdma_addrinfo **rai)
{
	struct rdma_addrinfo rai_hints, *_rai;
	struct rdma_addrinfo **rai_current;
	int ret = fi_ibv_fi_to_rai(hints, flags, &rai_hints);

	if (ret)
		goto out;

	if (!node && !rai_hints.ai_dst_addr) {
		if (!rai_hints.ai_src_addr && !service)
			node = local_node;
		rai_hints.ai_flags |= RAI_PASSIVE;
	}

	ret = rdma_getaddrinfo((char *) node, (char *) service,
				&rai_hints, &_rai);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_getaddrinfo", errno);
		if (errno) {
			ret = -errno;
		}
		goto out;
	}

	/*
	 * If caller requested rai, remove ib_rai entries added by IBACM to
	 * prevent wrong ib_connect_hdr from being sent in connect request.
	 */
	if (rai && hints && (hints->addr_format != FI_SOCKADDR_IB)) {
		for (rai_current = &_rai; *rai_current;) {
			struct rdma_addrinfo *rai_next;
			if ((*rai_current)->ai_family == AF_IB) {
				rai_next = (*rai_current)->ai_next;
				(*rai_current)->ai_next = NULL;
				rdma_freeaddrinfo(*rai_current);
				*rai_current = rai_next;
				continue;
			}
			rai_current = &(*rai_current)->ai_next;
		}
	}

	if (rai)
		*rai = _rai;

out:
	if (rai_hints.ai_src_addr)
		free(rai_hints.ai_src_addr);
	if (rai_hints.ai_dst_addr)
		free(rai_hints.ai_dst_addr);
	return ret;
}

int fi_ibv_get_rai_id(const char *node, const char *service, uint64_t flags,
		      const struct fi_info *hints, struct rdma_addrinfo **rai,
		      struct rdma_cm_id **id)
{
	int ret;

	// TODO create a similar function that won't require pruning ib_rai
	ret = fi_ibv_get_rdma_rai(node, service, flags, hints, rai);
	if (ret)
		return ret;

	ret = rdma_create_id(NULL, id, NULL, RDMA_PS_TCP);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_create_id", errno);
		ret = -errno;
		goto err1;
	}

	if ((*rai)->ai_flags & RAI_PASSIVE) {
		ret = rdma_bind_addr(*id, (*rai)->ai_src_addr);
		if (ret) {
			VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_bind_addr", errno);
			ret = -errno;
			goto err2;
		}
		return 0;
	}

	ret = rdma_resolve_addr(*id, (*rai)->ai_src_addr,
				(*rai)->ai_dst_addr, 2000);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_resolve_addr", errno);
		ret = -errno;
		goto err2;
	}
	return 0;
err2:
	if (rdma_destroy_id(*id))
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_destroy_id", errno);
err1:
	rdma_freeaddrinfo(*rai);
	return ret;
}

int fi_ibv_create_ep(const char *node, const char *service,
		     uint64_t flags, const struct fi_info *hints,
		     struct rdma_addrinfo **rai, struct rdma_cm_id **id)
{
	struct rdma_addrinfo *_rai = NULL;
	int ret;

	ret = fi_ibv_get_rdma_rai(node, service, flags, hints, &_rai);
	if (ret) {
		return ret;
	}

	ret = rdma_create_ep(id, _rai, NULL, NULL);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_create_ep", errno);
		ret = -errno;
		goto err1;
	}

	if (rai) {
		*rai = _rai;
	} else {
		rdma_freeaddrinfo(_rai);
	}

	return ret;
err1:
	rdma_freeaddrinfo(_rai);

	return ret;
}

void fi_ibv_destroy_ep(struct rdma_addrinfo *rai, struct rdma_cm_id **id)
{
	rdma_freeaddrinfo(rai);
	rdma_destroy_ep(*id);
}

static int fi_ibv_param_define(const char *param_name, const char *param_str,
			       enum fi_param_type type, void *param_default)
{
	char *param_help, param_default_str[256] = { 0 };
	char *begin_def_section = " (default: ", *end_def_section = ")";
	int ret = FI_SUCCESS;
	size_t len, param_default_sz = 0;
	size_t param_str_sz = strlen(param_str);
	size_t begin_def_section_sz = strlen(begin_def_section);
	size_t end_def_section_sz = strlen(end_def_section);

	if (param_default != NULL) {
		switch (type) {
		case FI_PARAM_STRING:
			if (*(char **)param_default != NULL) {
				param_default_sz =
					MIN(strlen(*(char **)param_default),
					    254);
				strncpy(param_default_str, *(char **)param_default,
					param_default_sz);
				param_default_str[param_default_sz + 1] = '\0';
			}
			break;
		case FI_PARAM_INT:
		case FI_PARAM_BOOL:
			snprintf(param_default_str, 256, "%d", *((int *)param_default));
			param_default_sz = strlen(param_default_str);
			break;
		case FI_PARAM_SIZE_T:
			snprintf(param_default_str, 256, "%zu", *((size_t *)param_default));
			param_default_sz = strlen(param_default_str);
			break;
		default:
			assert(0);
			ret = -FI_EINVAL;
			goto fn;
		}
	}

	len = param_str_sz + strlen(begin_def_section) +
		param_default_sz + end_def_section_sz + 1;
	param_help = calloc(1, len);
	if (!param_help) {
 		assert(0);
		ret = -FI_ENOMEM;
		goto fn;
	}

	strncat(param_help, param_str, param_str_sz + 1);
	strncat(param_help, begin_def_section, begin_def_section_sz + 1);
	strncat(param_help, param_default_str, param_default_sz + 1);
	strncat(param_help, end_def_section, end_def_section_sz + 1);

	param_help[len - 1] = '\0';

	fi_param_define(&fi_ibv_prov, param_name, type, param_help);

	free(param_help);
fn:
	return ret;
}

#if ENABLE_DEBUG
static int fi_ibv_dbg_query_qp_attr(struct ibv_qp *qp)
{
	struct ibv_qp_init_attr attr = { 0 };
	struct ibv_qp_attr qp_attr = { 0 };
	int ret;

	ret = ibv_query_qp(qp, &qp_attr, IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
			   IBV_QP_RNR_RETRY | IBV_QP_MIN_RNR_TIMER, &attr);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to query QP\n");
		return ret;
	}
	FI_DBG(&fi_ibv_prov, FI_LOG_EP_CTRL, "QP attributes: "
	       "min_rnr_timer"	": %" PRIu8 ", "
	       "timeout"	": %" PRIu8 ", "
	       "retry_cnt"	": %" PRIu8 ", "
	       "rnr_retry"	": %" PRIu8 "\n",
	       qp_attr.min_rnr_timer, qp_attr.timeout, qp_attr.retry_cnt,
	       qp_attr.rnr_retry);
	return 0;
}
#else
static int fi_ibv_dbg_query_qp_attr(struct ibv_qp *qp)
{
	return 0;
}
#endif

int fi_ibv_set_rnr_timer(struct ibv_qp *qp)
{
	struct ibv_qp_attr attr = { 0 };
	int ret;

	if (fi_ibv_gl_data.min_rnr_timer > 31) {
		VERBS_WARN(FI_LOG_EQ, "min_rnr_timer value out of valid range; "
			   "using default value of %d\n",
			   VERBS_DEFAULT_MIN_RNR_TIMER);
		attr.min_rnr_timer = VERBS_DEFAULT_MIN_RNR_TIMER;
	} else {
		attr.min_rnr_timer = fi_ibv_gl_data.min_rnr_timer;
	}

	/* XRC initiator QP do not have responder logic */
	if (qp->qp_type == IBV_QPT_XRC_SEND)
		return 0;

	ret = ibv_modify_qp(qp, &attr, IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		VERBS_WARN(FI_LOG_EQ, "Unable to modify QP attribute\n");
		return ret;
	}
	ret = fi_ibv_dbg_query_qp_attr(qp);
	if (ret)
		return ret;
	return 0;
}

int fi_ibv_find_max_inline(struct ibv_pd *pd, struct ibv_context *context,
			   enum ibv_qp_type qp_type)
{
	struct ibv_qp_init_attr qp_attr;
	struct ibv_qp *qp = NULL;
	struct ibv_cq *cq;
	int max_inline = 2;
	int rst = 0;
	const char *dev_name = ibv_get_device_name(context->device);
	uint8_t i;

	for (i = 0; i < count_of(verbs_dev_presets); i++) {
		if (!strncmp(dev_name, verbs_dev_presets[i].dev_name_prefix,
			     strlen(verbs_dev_presets[i].dev_name_prefix)))
			return verbs_dev_presets[i].max_inline_data;
	}

	cq = ibv_create_cq(context, 1, NULL, NULL, 0);
	assert(cq);

	memset(&qp_attr, 0, sizeof(qp_attr));
	qp_attr.send_cq = cq;
	qp_attr.qp_type = qp_type;
	qp_attr.cap.max_send_wr = 1;
	qp_attr.cap.max_send_sge = 1;
	if (!fi_ibv_is_xrc_send_qp(qp_type)) {
		qp_attr.recv_cq = cq;
		qp_attr.cap.max_recv_wr = 1;
		qp_attr.cap.max_recv_sge = 1;
	}
	qp_attr.sq_sig_all = 1;

	do {
		if (qp)
			ibv_destroy_qp(qp);
		qp_attr.cap.max_inline_data = max_inline;
		qp = ibv_create_qp(pd, &qp_attr);
		if (qp) {
			/*
			 * truescale returns max_inline_data 0
			 */
			if (qp_attr.cap.max_inline_data == 0)
				break;

			/*
			 * iWarp is able to create qp with unsupported
			 * max_inline, lets take first returned value.
			 */
			if (context->device->transport_type == IBV_TRANSPORT_IWARP) {
				max_inline = rst = qp_attr.cap.max_inline_data;
				break;
			}
			rst = max_inline;
		}
	} while (qp && (max_inline < INT_MAX / 2) && (max_inline *= 2));

	if (rst != 0) {
		int pos = rst, neg = max_inline;
		do {
			max_inline = pos + (neg - pos) / 2;
			if (qp)
				ibv_destroy_qp(qp);

			qp_attr.cap.max_inline_data = max_inline;
			qp = ibv_create_qp(pd, &qp_attr);
			if (qp)
				pos = max_inline;
			else
				neg = max_inline;

		} while (neg - pos > 2);

		rst = pos;
	}

	if (qp) {
		ibv_destroy_qp(qp);
	}

	if (cq) {
		ibv_destroy_cq(cq);
	}

	return rst;
}

static int fi_ibv_get_param_int(const char *param_name,
				const char *param_str,
				int *param_default)
{
	int param, ret;

	ret = fi_ibv_param_define(param_name, param_str,
				  FI_PARAM_INT,
				  param_default);
	if (ret)
		return ret;

	if (!fi_param_get_int(&fi_ibv_prov, param_name, &param))
		*param_default = param;

	return 0;
}

static int fi_ibv_get_param_bool(const char *param_name,
				 const char *param_str,
				 int *param_default)
{
	int param, ret;

	ret = fi_ibv_param_define(param_name, param_str,
				  FI_PARAM_BOOL,
				  param_default);
	if (ret)
		return ret;

	if (!fi_param_get_bool(&fi_ibv_prov, param_name, &param)) {
		*param_default = param;
		if ((*param_default != 1) && (*param_default != 0))
			return -FI_EINVAL;
	}

	return 0;
}

static int fi_ibv_get_param_str(const char *param_name,
				const char *param_str,
				char **param_default)
{
	char *param;
	int ret;

	ret = fi_ibv_param_define(param_name, param_str,
				  FI_PARAM_STRING,
				  param_default);
	if (ret)
		return ret;

	if (!fi_param_get_str(&fi_ibv_prov, param_name, &param))
		*param_default = param;

	return 0;
}

static int fi_ibv_read_params(void)
{
	/* Common parameters */
	if (fi_ibv_get_param_int("tx_size", "Default maximum tx context size",
				 &fi_ibv_gl_data.def_tx_size) ||
	    (fi_ibv_gl_data.def_tx_size < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of tx_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("rx_size", "Default maximum rx context size",
				 &fi_ibv_gl_data.def_rx_size) ||
	    (fi_ibv_gl_data.def_rx_size < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of rx_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("tx_iov_limit", "Default maximum tx iov_limit",
				 &fi_ibv_gl_data.def_tx_iov_limit) ||
	    (fi_ibv_gl_data.def_tx_iov_limit < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of tx_iov_limit\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("rx_iov_limit", "Default maximum rx iov_limit",
				 &fi_ibv_gl_data.def_rx_iov_limit) ||
	    (fi_ibv_gl_data.def_rx_iov_limit < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of rx_iov_limit\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("inline_size", "Default maximum inline size. "
				 "Actual inject size returned in fi_info may be "
				 "greater", &fi_ibv_gl_data.def_inline_size) ||
	    (fi_ibv_gl_data.def_inline_size < 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of inline_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("min_rnr_timer", "Set min_rnr_timer QP "
				 "attribute (0 - 31)",
				 &fi_ibv_gl_data.min_rnr_timer) ||
	    ((fi_ibv_gl_data.min_rnr_timer < 0) ||
	     (fi_ibv_gl_data.min_rnr_timer > 31))) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of min_rnr_timer\n");
		return -FI_EINVAL;
	}

	if (fi_ibv_get_param_bool("use_odp", "Enable on-demand paging experimental feature. "
				  "Currently this feature may corrupt data. "
				  "Use it on your own risk.",
				  &fi_ibv_gl_data.use_odp)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of use_odp\n");
		return -FI_EINVAL;
	}

	if (fi_ibv_get_param_bool("prefer_xrc", "Order XRC transport fi_infos"
				  "ahead of RC. Default orders RC first.",
				  &fi_ibv_gl_data.msg.prefer_xrc)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of prefer_xrc\n");
		return -FI_EINVAL;
	}

	if (fi_ibv_get_param_str("xrcd_filename", "A file to "
				 "associate with the XRC domain.",
				 &fi_ibv_gl_data.msg.xrcd_filename)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of xrcd_filename\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("cqread_bunch_size", "The number of entries to "
				 "be read from the verbs completion queue at a time",
				 &fi_ibv_gl_data.cqread_bunch_size) ||
	    (fi_ibv_gl_data.cqread_bunch_size <= 0)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of cqread_bunch_size\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_str("iface", "The prefix or the full name of the "
				 "network interface associated with the verbs device",
				 &fi_ibv_gl_data.iface)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of iface\n");
		return -FI_EINVAL;
	}

	/* DGRAM-specific parameters */
	if (getenv("OMPI_COMM_WORLD_RANK") || getenv("PMI_RANK"))
		fi_ibv_gl_data.dgram.use_name_server = 0;
	if (fi_ibv_get_param_bool("dgram_use_name_server", "The option that "
				  "enables/disables OFI Name Server thread that is used "
				  "to resolve IP-addresses to provider specific "
				  "addresses. If MPI is used, the NS is disabled "
				  "by default.", &fi_ibv_gl_data.dgram.use_name_server)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of dgram_use_name_server\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("dgram_name_server_port", "The port on which Name Server "
				 "thread listens incoming connections and requestes.",
				 &fi_ibv_gl_data.dgram.name_server_port) ||
	    (fi_ibv_gl_data.dgram.name_server_port < 0 ||
	     fi_ibv_gl_data.dgram.name_server_port > 65535)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of dgram_name_server_port\n");
		return -FI_EINVAL;
	}
	if (fi_ibv_get_param_int("gid_idx", "Set which gid index to use "
				 "attribute (0 - 255)",
				 &fi_ibv_gl_data.gid_idx) ||
	    (fi_ibv_gl_data.gid_idx < 0 ||
	     fi_ibv_gl_data.gid_idx > 255)) {
		VERBS_WARN(FI_LOG_CORE,
			   "Invalid value of gid index\n");
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

static void fi_ibv_fini(void)
{
#if HAVE_VERBS_DL
	ofi_monitor_cleanup();
	ofi_mem_fini();
#endif
	fi_freeinfo((void *)fi_ibv_util_prov.info);
	fi_ibv_util_prov.info = NULL;
}

VERBS_INI
{
#if HAVE_VERBS_DL
	ofi_mem_init();
	ofi_monitor_init();
#endif
	if (fi_ibv_read_params()|| fi_ibv_init_info(&fi_ibv_util_prov.info))
		return NULL;
	return &fi_ibv_prov;
}
