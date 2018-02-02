/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#ifndef _FI_MLX_H
#define _FI_MLX_H


#ifdef __cplusplus
extern "C" {
#endif

#include "config.h"
#include <ucp/api/ucp.h>
#include <ucm/api/ucm.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rdma/fabric.h>
#include <rdma/providers/fi_prov.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/providers/fi_log.h>
#include <ofi.h>
#include <ofi_lock.h>
#include <ofi_list.h>
#include "ofi_enosys.h"
#include <ofi_mem.h>
#include <ofi_atom.h>
#include <ofi_util.h>
#include <ofi_prov.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <ifaddrs.h>

#define FI_MLX_FABRIC_NAME "mlx"
#define FI_MLX_DEFAULT_INJECT_SIZE 1024
#define FI_MLX_DEFAULT_NS_PORT 12345
#define FI_MLX_DEF_CQ_SIZE (1024)
#define FI_MLX_DEF_MR_CNT (1 << 16)

#define FI_MLX_VERSION_MINOR 5
#define FI_MLX_VERSION_MAJOR 1
#define FI_MLX_VERSION (FI_VERSION(FI_MLX_VERSION_MAJOR, FI_MLX_VERSION_MINOR))

#define FI_MLX_RKEY_MAX_LEN (256)

#define FI_MLX_MAX_NAME_LEN (1024)

#define FI_MLX_CAPS (FI_SEND | FI_RECV | FI_TAGGED)
#define FI_MLX_MODE_REQUIRED (0ULL)
#define FI_MLX_MODE_SUPPORTED (FI_CONTEXT | FI_ASYNC_IOV)
#define FI_MLX_OP_FLAGS (FI_SEND | FI_RECV)
#define FI_MLX_ANY_SERVICE (0)
struct mlx_global_descriptor{
	ucp_config_t *config;
	int use_ns;
	int ns_port;
	struct util_ns name_serv;
	char *localhost;
};

struct mlx_fabric {
	struct util_fabric u_fabric;
};

struct mlx_domain {
	struct util_domain u_domain;
	ucp_context_h context;

	struct util_buf_pool *fast_path_pool;
	fastlock_t fpp_lock;
};


struct mlx_ep {
	struct util_ep ep;
	struct mlx_av *av; /*until AV is not implemented via utils*/
	ucp_worker_h worker;
	short service;
	void *addr;
	size_t addr_len;
};

struct mlx_av {
	struct fid_av av;
	struct mlx_domain *domain;
	struct mlx_ep *ep;
	struct util_eq *eq;
	int type;
	int async;
	size_t count;
	size_t addr_len;
};

typedef enum mlx_req_type {
	MLX_FI_REQ_UNINITIALIZED = 0,
	MLX_FI_REQ_REGULAR = 0xFD,
	MLX_FI_REQ_UNEXPECTED_ERR = 0xFE,
	MLX_FI_REQ_UNEXPECTED = 0xFF,
} mlx_req_type_t;

struct mlx_request {
	mlx_req_type_t type;

	union {
		struct fi_cq_tagged_entry tagged;
		struct fi_cq_err_entry error;
	} completion;

	struct util_cq* cq;
	struct mlx_ep* ep;
};

OFI_DECLARE_CIRQUE(struct fi_cq_tagged_entry, mlx_comp_cirq);

extern int mlx_errcode_translation_table[];
#define MLX_TRANSLATE_ERRCODE(X) mlx_errcode_translation_table[(-X)+1]
extern struct fi_provider mlx_prov;
extern struct mlx_global_descriptor mlx_descriptor;
extern struct util_prov mlx_util_prov;

extern struct fi_ops_cm mlx_cm_ops;
extern struct fi_ops_tagged mlx_tagged_ops;
extern struct fi_ops_mr mlx_mr_ops;
extern struct fi_fabric_attr mlx_fabric_attrs;

int mlx_fabric_open(
		struct fi_fabric_attr *attr,
		struct fid_fabric **fabric, 
		void *context);

int mlx_domain_open(
		struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **fid, void *context);

int mlx_ep_open(
		struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **fid, void *context);

int mlx_cq_open(
		struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq, void *context);

int mlx_av_open(
		struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context);

int mlx_ns_is_service_wildcard(void *svc);
int mlx_ns_service_cmp(void *svc1, void *svc2);
/* Callbacks */
void mlx_send_callback_no_compl( void *request, ucs_status_t status);
void mlx_send_callback( void *request, ucs_status_t status);
void mlx_recv_callback_no_compl(void *request, ucs_status_t status,
				ucp_tag_recv_info_t *info);
void mlx_recv_callback( void *request, ucs_status_t status,
			ucp_tag_recv_info_t *info);
#ifdef __cplusplus
}
#endif

#endif
