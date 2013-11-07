/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2005, 2006 Cisco Systems.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 * Copyright (c) 2013 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

#ifndef _FI_UVERBS_H_
#define _FI_UVERBS_H_


#include <linux/types.h>
#include <rdma/fabric.h>


#ifdef __cplusplus
extern "C" {
#endif


/*
 * This file must be kept in sync with the kernel's version of ib_user_verbs.h
 */

#define UVERBS_MIN_ABI_VERSION	6
#define UVERBS_MAX_ABI_VERSION	6

enum {
	UVERBS_CMD_GET_CONTEXT,
	UVERBS_CMD_QUERY_DEVICE,
	UVERBS_CMD_QUERY_PORT,
	UVERBS_CMD_ALLOC_PD,
	UVERBS_CMD_DEALLOC_PD,
	UVERBS_CMD_CREATE_AH,
	UVERBS_CMD_MODIFY_AH,	/* unused */
	UVERBS_CMD_QUERY_AH,	/* unused */
	UVERBS_CMD_DESTROY_AH,
	UVERBS_CMD_REG_MR,
	UVERBS_CMD_REG_SMR,	/* unused */
	UVERBS_CMD_REREG_MR,	/* unused */
	UVERBS_CMD_QUERY_MR,	/* unused */
	UVERBS_CMD_DEREG_MR,
	UVERBS_CMD_ALLOC_MW,	/* unused */
	UVERBS_CMD_BIND_MW,	/* unused */
	UVERBS_CMD_DEALLOC_MW,	/* unused */
	UVERBS_CMD_CREATE_COMP_CHANNEL,
	UVERBS_CMD_CREATE_CQ,
	UVERBS_CMD_RESIZE_CQ,
	UVERBS_CMD_DESTROY_CQ,
	UVERBS_CMD_POLL_CQ,
	UVERBS_CMD_PEEK_CQ,
	UVERBS_CMD_REQ_NOTIFY_CQ,
	UVERBS_CMD_CREATE_QP,
	UVERBS_CMD_QUERY_QP,
	UVERBS_CMD_MODIFY_QP,
	UVERBS_CMD_DESTROY_QP,
	UVERBS_CMD_POST_SEND,
	UVERBS_CMD_POST_RECV,
	UVERBS_CMD_ATTACH_MCAST,
	UVERBS_CMD_DETACH_MCAST,
	UVERBS_CMD_CREATE_SRQ,
	UVERBS_CMD_MODIFY_SRQ,
	UVERBS_CMD_QUERY_SRQ,
	UVERBS_CMD_DESTROY_SRQ,
	UVERBS_CMD_POST_SRQ_RECV,
	UVERBS_CMD_OPEN_XRCD,	/* TODO */
	UVERBS_CMD_CLOSE_XRCD,	/* TODO */
	UVERBS_CMD_CREATE_XSRQ,	/* TODO */
	UVERBS_CMD_OPEN_QP,	/* TODO */
};

/*
 * Make sure that all structs defined in this file remain laid out so
 * that they pack the same way on 32-bit and 64-bit architectures (to
 * avoid incompatibility between 32-bit userspace and 64-bit kernels).
 * Specifically:
 *  - Do not use pointer types -- pass pointers in __u64 instead.
 *  - Make sure that any structure larger than 4 bytes is padded to a
 *    multiple of 8 bytes.  Otherwise the structure size will be
 *    different between 32-bit and 64-bit architectures.
 */

struct ibv_kern_async_event {
	__u64 element;
	__u32 event_type;
	__u32 reserved;
};

struct ibv_comp_event {
	__u64 cq_handle;
};

/*
 * All commands from userspace should start with a __u32 command field
 * followed by __u16 in_words and out_words fields (which give the
 * length of the command block and response buffer if any in 32-bit
 * words).  The kernel driver will read these fields first and read
 * the rest of the command struct based on these value.
 */

struct ibv_query_params {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
};

struct ibv_query_params_resp {
	__u32 num_cq_events;
};

struct ibv_get_context {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 driver_data[0];
};

struct ibv_get_context_resp {
	__u32 async_fd;
	__u32 num_comp_vectors;
};

struct ibv_query_device {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 driver_data[0];
};

struct ibv_query_device_resp {
	__u64 fw_ver;
	__u64 node_guid;
	__u64 sys_image_guid;
	__u64 max_mr_size;
	__u64 page_size_cap;
	__u32 vendor_id;
	__u32 vendor_part_id;
	__u32 hw_ver;
	__u32 max_qp;
	__u32 max_qp_wr;
	__u32 device_cap_flags;
	__u32 max_sge;
	__u32 max_sge_rd;
	__u32 max_cq;
	__u32 max_cqe;
	__u32 max_mr;
	__u32 max_pd;
	__u32 max_qp_rd_atom;
	__u32 max_ee_rd_atom;
	__u32 max_res_rd_atom;
	__u32 max_qp_init_rd_atom;
	__u32 max_ee_init_rd_atom;
	__u32 atomic_cap;
	__u32 max_ee;
	__u32 max_rdd;
	__u32 max_mw;
	__u32 max_raw_ipv6_qp;
	__u32 max_raw_ethy_qp;
	__u32 max_mcast_grp;
	__u32 max_mcast_qp_attach;
	__u32 max_total_mcast_qp_attach;
	__u32 max_ah;
	__u32 max_fmr;
	__u32 max_map_per_fmr;
	__u32 max_srq;
	__u32 max_srq_wr;
	__u32 max_srq_sge;
	__u16 max_pkeys;
	__u8  local_ca_ack_delay;
	__u8  phys_port_cnt;
	__u8  reserved[4];
};

struct ibv_query_port {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u8  port_num;
	__u8  reserved[7];
	__u64 driver_data[0];
};

struct ibv_query_port_resp {
	__u32 port_cap_flags;
	__u32 max_msg_sz;
	__u32 bad_pkey_cntr;
	__u32 qkey_viol_cntr;
	__u32 gid_tbl_len;
	__u16 pkey_tbl_len;
	__u16 lid;
	__u16 sm_lid;
	__u8  state;
	__u8  max_mtu;
	__u8  active_mtu;
	__u8  lmc;
	__u8  max_vl_num;
	__u8  sm_sl;
	__u8  subnet_timeout;
	__u8  init_type_reply;
	__u8  active_width;
	__u8  active_speed;
	__u8  phys_state;
	__u8  link_layer;
	__u8  reserved[2];
};

struct ibv_alloc_pd {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 driver_data[0];
};

struct ibv_alloc_pd_resp {
	__u32 pd_handle;
};

struct ibv_dealloc_pd {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u32 pd_handle;
};

struct ibv_open_xrcd {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 fd;
	__u32 oflags;
	__u64 driver_data[0];
};

struct ibv_open_xrcd_resp {
	__u32 xrcd_handle;
};

struct ibv_close_xrcd {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u32 xrcd_handle;
};

struct ibv_reg_mr {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 start;
	__u64 length;
	__u64 hca_va;
	__u32 pd_handle;
	__u32 access_flags;
	__u64 driver_data[0];
};

struct ibv_reg_mr_resp {
	__u32 mr_handle;
	__u32 lkey;
	__u32 rkey;
};

struct ibv_dereg_mr {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u32 mr_handle;
};

struct ibv_create_comp_channel {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
};

struct ibv_create_comp_channel_resp {
	__u32 fd;
};

struct ibv_create_cq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 user_handle;
	__u32 cqe;
	__u32 comp_vector;
	__s32 comp_channel;
	__u32 reserved;
	__u64 driver_data[0];
};

struct ibv_create_cq_resp {
	__u32 cq_handle;
	__u32 cqe;
};

struct ibv_kern_wc {
	__u64  wr_id;
	__u32  status;
	__u32  opcode;
	__u32  vendor_err;
	__u32  byte_len;
	__u32  imm_data;
	__u32  qp_num;
	__u32  src_qp;
	__u32  wc_flags;
	__u16  pkey_index;
	__u16  slid;
	__u8   sl;
	__u8   dlid_path_bits;
	__u8   port_num;
	__u8   reserved;
};

struct ibv_poll_cq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 cq_handle;
	__u32 ne;
};

struct ibv_poll_cq_resp {
	__u32 count;
	__u32 reserved;
	struct ibv_kern_wc wc[0];
};

struct ibv_req_notify_cq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u32 cq_handle;
	__u32 solicited;
};

struct ibv_resize_cq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 cq_handle;
	__u32 cqe;
	__u64 driver_data[0];
};

struct ibv_resize_cq_resp {
	__u32 cqe;
	__u32 reserved;
	__u64 driver_data[0];
};

struct ibv_destroy_cq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 cq_handle;
	__u32 reserved;
};

struct ibv_destroy_cq_resp {
	__u32 comp_events_reported;
	__u32 async_events_reported;
};

struct ibv_kern_global_route {
	__u8  dgid[16];
	__u32 flow_label;
	__u8  sgid_index;
	__u8  hop_limit;
	__u8  traffic_class;
	__u8  reserved;
};

struct ibv_kern_ah_attr {
	struct ibv_kern_global_route grh;
	__u16 dlid;
	__u8  sl;
	__u8  src_path_bits;
	__u8  static_rate;
	__u8  is_global;
	__u8  port_num;
	__u8  reserved;
};

struct ibv_kern_qp_attr {
	__u32	qp_attr_mask;
	__u32	qp_state;
	__u32	cur_qp_state;
	__u32	path_mtu;
	__u32	path_mig_state;
	__u32	qkey;
	__u32	rq_psn;
	__u32	sq_psn;
	__u32	dest_qp_num;
	__u32	qp_access_flags;

	struct ibv_kern_ah_attr ah_attr;
	struct ibv_kern_ah_attr alt_ah_attr;

	/* ib_qp_cap */
	__u32	max_send_wr;
	__u32	max_recv_wr;
	__u32	max_send_sge;
	__u32	max_recv_sge;
	__u32	max_inline_data;

	__u16	pkey_index;
	__u16	alt_pkey_index;
	__u8	en_sqd_async_notify;
	__u8	sq_draining;
	__u8	max_rd_atomic;
	__u8	max_dest_rd_atomic;
	__u8	min_rnr_timer;
	__u8	port_num;
	__u8	timeout;
	__u8	retry_cnt;
	__u8	rnr_retry;
	__u8	alt_port_num;
	__u8	alt_timeout;
	__u8	reserved[5];
};

struct ibv_create_qp {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 user_handle;
	__u32 pd_handle;
	__u32 send_cq_handle;
	__u32 recv_cq_handle;
	__u32 srq_handle;
	__u32 max_send_wr;
	__u32 max_recv_wr;
	__u32 max_send_sge;
	__u32 max_recv_sge;
	__u32 max_inline_data;
	__u8  sq_sig_all;
	__u8  qp_type;
	__u8  is_srq;
	__u8  reserved;
	__u64 driver_data[0];
};

struct ibv_open_qp {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 user_handle;
	__u32 pd_handle;
	__u32 qpn;
	__u8  qp_type;
	__u8  reserved[7];
	__u64 driver_data[0];
};

/* also used for open response */
struct ibv_create_qp_resp {
	__u32 qp_handle;
	__u32 qpn;
	__u32 max_send_wr;
	__u32 max_recv_wr;
	__u32 max_send_sge;
	__u32 max_recv_sge;
	__u32 max_inline_data;
	__u32 reserved;
};

struct ibv_qp_dest {
	__u8  dgid[16];
	__u32 flow_label;
	__u16 dlid;
	__u16 reserved;
	__u8  sgid_index;
	__u8  hop_limit;
	__u8  traffic_class;
	__u8  sl;
	__u8  src_path_bits;
	__u8  static_rate;
	__u8  is_global;
	__u8  port_num;
};

struct ibv_query_qp {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 qp_handle;
	__u32 attr_mask;
	__u64 driver_data[0];
};

struct ibv_query_qp_resp {
	struct ibv_qp_dest dest;
	struct ibv_qp_dest alt_dest;
	__u32 max_send_wr;
	__u32 max_recv_wr;
	__u32 max_send_sge;
	__u32 max_recv_sge;
	__u32 max_inline_data;
	__u32 qkey;
	__u32 rq_psn;
	__u32 sq_psn;
	__u32 dest_qp_num;
	__u32 qp_access_flags;
	__u16 pkey_index;
	__u16 alt_pkey_index;
	__u8  qp_state;
	__u8  cur_qp_state;
	__u8  path_mtu;
	__u8  path_mig_state;
	__u8  sq_draining;
	__u8  max_rd_atomic;
	__u8  max_dest_rd_atomic;
	__u8  min_rnr_timer;
	__u8  port_num;
	__u8  timeout;
	__u8  retry_cnt;
	__u8  rnr_retry;
	__u8  alt_port_num;
	__u8  alt_timeout;
	__u8  sq_sig_all;
	__u8  reserved[5];
	__u64 driver_data[0];
};

struct ibv_modify_qp {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	struct ibv_qp_dest dest;
	struct ibv_qp_dest alt_dest;
	__u32 qp_handle;
	__u32 attr_mask;
	__u32 qkey;
	__u32 rq_psn;
	__u32 sq_psn;
	__u32 dest_qp_num;
	__u32 qp_access_flags;
	__u16 pkey_index;
	__u16 alt_pkey_index;
	__u8  qp_state;
	__u8  cur_qp_state;
	__u8  path_mtu;
	__u8  path_mig_state;
	__u8  en_sqd_async_notify;
	__u8  max_rd_atomic;
	__u8  max_dest_rd_atomic;
	__u8  min_rnr_timer;
	__u8  port_num;
	__u8  timeout;
	__u8  retry_cnt;
	__u8  rnr_retry;
	__u8  alt_port_num;
	__u8  alt_timeout;
	__u8  reserved[2];
	__u64 driver_data[0];
};

struct ibv_destroy_qp {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 qp_handle;
	__u32 reserved;
};

struct ibv_destroy_qp_resp {
	__u32 events_reported;
};

struct ibv_kern_send_wr {
	__u64 wr_id;
	__u32 num_sge;
	__u32 opcode;
	__u32 send_flags;
	__u32 imm_data;
	union {
		struct {
			__u64 remote_addr;
			__u32 rkey;
			__u32 reserved;
		} rdma;
		struct {
			__u64 remote_addr;
			__u64 compare_add;
			__u64 swap;
			__u32 rkey;
			__u32 reserved;
		} atomic;
		struct {
			__u32 ah;
			__u32 remote_qpn;
			__u32 remote_qkey;
			__u32 reserved;
		} ud;
		struct {
			__u64 reserved[3];
			__u32 reserved2;
			__u32 remote_srqn;
		} xrc;
	} wr;
};

struct ibv_post_send {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 qp_handle;
	__u32 wr_count;
	__u32 sge_count;
	__u32 wqe_size;
	struct ibv_kern_send_wr send_wr[0];
};

struct ibv_post_send_resp {
	__u32 bad_wr;
};

struct ibv_kern_recv_wr {
	__u64 wr_id;
	__u32 num_sge;
	__u32 reserved;
};

struct ibv_post_recv {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 qp_handle;
	__u32 wr_count;
	__u32 sge_count;
	__u32 wqe_size;
	struct ibv_kern_recv_wr recv_wr[0];
};

struct ibv_post_recv_resp {
	__u32 bad_wr;
};

struct ibv_post_srq_recv {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 srq_handle;
	__u32 wr_count;
	__u32 sge_count;
	__u32 wqe_size;
	struct ibv_kern_recv_wr recv_wr[0];
};

struct ibv_post_srq_recv_resp {
	__u32 bad_wr;
};

struct ibv_create_ah {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 user_handle;
	__u32 pd_handle;
	__u32 reserved;
	struct ibv_kern_ah_attr attr;
};

struct ibv_create_ah_resp {
	__u32 handle;
};

struct ibv_destroy_ah {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u32 ah_handle;
};

struct ibv_attach_mcast {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u8  gid[16];
	__u32 qp_handle;
	__u16 mlid;
	__u16 reserved;
	__u64 driver_data[0];
};

struct ibv_detach_mcast {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u8  gid[16];
	__u32 qp_handle;
	__u16 mlid;
	__u16 reserved;
	__u64 driver_data[0];
};

struct ibv_create_srq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 user_handle;
	__u32 pd_handle;
	__u32 max_wr;
	__u32 max_sge;
	__u32 srq_limit;
	__u64 driver_data[0];
};

struct ibv_create_xsrq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u64 user_handle;
	__u32 srq_type;
	__u32 pd_handle;
	__u32 max_wr;
	__u32 max_sge;
	__u32 srq_limit;
	__u32 reserved;
	__u32 xrcd_handle;
	__u32 cq_handle;
	__u64 driver_data[0];
};

struct ibv_create_srq_resp {
	__u32 srq_handle;
	__u32 max_wr;
	__u32 max_sge;
	__u32 srqn;
};

struct ibv_modify_srq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u32 srq_handle;
	__u32 attr_mask;
	__u32 max_wr;
	__u32 srq_limit;
	__u64 driver_data[0];
};

struct ibv_query_srq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 srq_handle;
	__u32 reserved;
	__u64 driver_data[0];
};

struct ibv_query_srq_resp {
	__u32 max_wr;
	__u32 max_sge;
	__u32 srq_limit;
	__u32 reserved;
};

struct ibv_destroy_srq {
	__u32 command;
	__u16 in_words;
	__u16 out_words;
	__u64 response;
	__u32 srq_handle;
	__u32 reserved;
};

struct ibv_destroy_srq_resp {
	__u32 events_reported;
};


struct fi_ops_uverbs {
	size_t	size;
	int	(*get_context)(fid_t fid,
				struct ibv_get_context *cmd, size_t cmd_size,
				struct ibv_get_context_resp *resp, size_t resp_size);
	int	(*query_device)(fid_t fid,
				struct ibv_query_device *cmd, size_t cmd_size,
				struct ibv_query_device_resp *resp, size_t resp_size);
	int	(*query_port)(fid_t fid,
				struct ibv_query_port *cmd, size_t cmd_size,
				struct ibv_query_port_resp *resp, size_t resp_size);
	int	(*alloc_pd)(fid_t fid,
				struct ibv_alloc_pd *cmd, size_t cmd_size,
				struct ibv_alloc_pd_resp *resp, size_t resp_size);
	int	(*dealloc_pd)(fid_t fid,
				struct ibv_dealloc_pd *cmd, size_t cmd_size);
	int	(*create_ah)(fid_t fid,
				struct ibv_create_ah *cmd, size_t cmd_size,
				struct ibv_create_ah_resp *resp, size_t resp_size);
	int	(*destroy_ah)(fid_t fid,
				struct ibv_destroy_ah *cmd, size_t cmd_size);
	int	(*open_xrcd)(fid_t fid,
				struct ibv_open_xrcd *cmd, size_t cmd_size,
				struct ibv_open_xrcd_resp *resp, size_t resp_size);
	int	(*close_xrcd)(fid_t fid,
				struct ibv_close_xrcd *cmd, size_t cmd_size);
	int	(*reg_mr)(fid_t fid,
				struct ibv_reg_mr *cmd, size_t cmd_size,
				struct ibv_reg_mr_resp *resp, size_t resp_size);
	int	(*dereg_mr)(fid_t fid,
				struct ibv_dereg_mr *cd, size_t cmd_size);
	int	(*create_comp_channel)(fid_t fid,
				struct ibv_create_comp_channel *cmd, size_t cmd_size,
				struct ibv_create_comp_channel_resp *resp, size_t resp_size);
	int	(*create_cq)(fid_t fid,
				struct ibv_create_cq *cmd, size_t cmd_size,
				struct ibv_create_cq_resp *resp, size_t resp_size);
	int	(*poll_cq)(fid_t fid,
				struct ibv_poll_cq *cmd, size_t cmd_size,
				struct ibv_poll_cq_resp *resp, size_t resp_size);
	int	(*req_notify_cq)(fid_t fid,
				struct ibv_req_notify_cq *cmd, size_t cmd_size);
	int	(*resize_cq)(fid_t fid,
				struct ibv_resize_cq *cmd, size_t cmd_size,
				struct ibv_resize_cq_resp *resp, size_t resp_size);
	int	(*destroy_cq)(fid_t fid,
				struct ibv_destroy_cq *cmd, size_t cmd_size,
				struct ibv_destroy_cq_resp *resp, size_t resp_size);
	int	(*create_srq)(fid_t fid,
				struct ibv_create_srq *cmd, size_t cmd_size,
				struct ibv_create_srq_resp *resp, size_t resp_size);
	int	(*modify_srq)(fid_t fid,
				struct ibv_modify_srq *cmd, size_t cmd_size);
	int	(*query_srq)(fid_t fid,
				struct ibv_query_srq *cmd, size_t cmd_size,
				struct ibv_query_srq_resp *resp, size_t resp_size);
	int	(*destroy_srq)(fid_t fid,
				struct ibv_destroy_srq *cmd, size_t cmd_size,
				struct ibv_destroy_srq_resp *resp, size_t resp_size);
	int	(*create_qp)(fid_t fid,
				struct ibv_create_qp *cmd, size_t cmd_size,
				struct ibv_create_qp_resp *resp, size_t resp_size);
	int	(*open_qp)(fid_t fid,
				struct ibv_open_qp *cmd, size_t cmd_size,
				struct ibv_create_qp_resp *resp, size_t resp_size);
	int	(*query_qp)(fid_t fid,
				struct ibv_query_qp *cmd, size_t cmd_size,
				struct ibv_query_qp_resp *resp, size_t resp_size);
	int	(*modify_qp)(fid_t fid,
				struct ibv_modify_qp *cmd, size_t cmd_size);
	int	(*destroy_qp)(fid_t fid,
				struct ibv_destroy_qp *cmd, size_t cmd_size,
				struct ibv_destroy_qp_resp *resp, size_t resp_size);
	int	(*post_send)(fid_t fid,
				struct ibv_post_send *cmd, size_t cmd_size,
				struct ibv_post_send_resp *resp, size_t resp_size);
	int	(*post_recv)(fid_t fid,
				struct ibv_post_recv *cmd, size_t cmd_size,
				struct ibv_post_recv_resp *resp, size_t resp_size);
	int	(*post_srq_recv)(fid_t fid,
				struct ibv_post_srq_recv *cmd, size_t cmd_size,
				struct ibv_post_srq_recv_resp *resp, size_t resp_size);
	int	(*attach_mcast)(fid_t fid,
				struct ibv_attach_mcast *cmd, size_t cmd_size);
	int	(*detach_mcast)(fid_t fid,
				struct ibv_detach_mcast *cmd, size_t cmd_size);
};

struct fid_uverbs {
	struct fid		fid;
	int			fd;
	struct fi_ops_uverbs	*ops;
};

#define FI_UVERBS_INTERFACE	"uverbs"

static inline int
uv_get_context(fid_t fid,
	struct ibv_get_context *cmd, size_t cmd_size,
	struct ibv_get_context_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, get_context);
	return uv->ops->get_context(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_query_device(fid_t fid,
	struct ibv_query_device *cmd, size_t cmd_size,
	struct ibv_query_device_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, query_device);
	return uv->ops->query_device(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_query_port(fid_t fid,
	struct ibv_query_port *cmd, size_t cmd_size,
	struct ibv_query_port_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, query_port);
	return uv->ops->query_port(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_alloc_pd(fid_t fid,
	struct ibv_alloc_pd *cmd, size_t cmd_size,
	struct ibv_alloc_pd_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, alloc_pd);
	return uv->ops->alloc_pd(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_dealloc_pd(fid_t fid,
	struct ibv_dealloc_pd *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, dealloc_pd);
	return uv->ops->dealloc_pd(fid, cmd, cmd_size);
}

static inline int
uv_create_ah(fid_t fid,
	struct ibv_create_ah *cmd, size_t cmd_size,
	struct ibv_create_ah_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_ah);
	return uv->ops->create_ah(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_destroy_ah(fid_t fid,
	struct ibv_destroy_ah *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, destroy_ah);
	return uv->ops->destroy_ah(fid, cmd, cmd_size);
}

static inline int
uv_open_xrcd(fid_t fid,
	struct ibv_open_xrcd *cmd, size_t cmd_size,
	struct ibv_open_xrcd_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, open_xrcd);
	return uv->ops->open_xrcd(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_close_xrcd(fid_t fid,
	struct ibv_close_xrcd *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, close_xrcd);
	return uv->ops->close_xrcd(fid, cmd, cmd_size);
}

static inline int
uv_reg_mr(fid_t fid,
	struct ibv_reg_mr *cmd, size_t cmd_size,
	struct ibv_reg_mr_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, reg_mr);
	return uv->ops->reg_mr(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_dereg_mr(fid_t fid,
	struct ibv_dereg_mr *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, dereg_mr);
	return uv->ops->dereg_mr(fid, cmd, cmd_size);
}

static inline int
uv_create_comp_channel(fid_t fid,
	struct ibv_create_comp_channel *cmd, size_t cmd_size,
	struct ibv_create_comp_channel_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_comp_channel);
	return uv->ops->create_comp_channel(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_create_cq(fid_t fid,
	struct ibv_create_cq *cmd, size_t cmd_size,
	struct ibv_create_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_cq);
	return uv->ops->create_cq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_poll_cq(fid_t fid,
	struct ibv_poll_cq *cmd, size_t cmd_size,
	struct ibv_poll_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, poll_cq);
	return uv->ops->poll_cq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_req_notify_cq(fid_t fid,
	struct ibv_req_notify_cq *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, req_notify_cq);
	return uv->ops->req_notify_cq(fid, cmd, cmd_size);
}

static inline int
uv_resize_cq(fid_t fid,
	struct ibv_resize_cq *cmd, size_t cmd_size,
	struct ibv_resize_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, resize_cq);
	return uv->ops->resize_cq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_destroy_cq(fid_t fid,
	struct ibv_destroy_cq *cmd, size_t cmd_size,
	struct ibv_destroy_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, destroy_cq);
	return uv->ops->destroy_cq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_create_srq(fid_t fid,
	struct ibv_create_srq *cmd, size_t cmd_size,
	struct ibv_create_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_srq);
	return uv->ops->create_srq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_modify_srq(fid_t fid,
	struct ibv_modify_srq *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, modify_srq);
	return uv->ops->modify_srq(fid, cmd, cmd_size);
}

static inline int
uv_query_srq(fid_t fid,
	struct ibv_query_srq *cmd, size_t cmd_size,
	struct ibv_query_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, query_srq);
	return uv->ops->query_srq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_destroy_srq(fid_t fid,
	struct ibv_destroy_srq *cmd, size_t cmd_size,
	struct ibv_destroy_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, destroy_srq);
	return uv->ops->destroy_srq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_create_qp(fid_t fid,
	struct ibv_create_qp *cmd, size_t cmd_size,
	struct ibv_create_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_qp);
	return uv->ops->create_qp(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_open_qp(fid_t fid,
	struct ibv_open_qp *cmd, size_t cmd_size,
	struct ibv_create_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, open_qp);
	return uv->ops->open_qp(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_query_qp(fid_t fid,
	struct ibv_query_qp *cmd, size_t cmd_size,
	struct ibv_query_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, query_qp);
	return uv->ops->query_qp(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_modify_qp(fid_t fid,
	struct ibv_modify_qp *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, modify_qp);
	return uv->ops->modify_qp(fid, cmd, cmd_size);
}

static inline int
uv_destroy_qp(fid_t fid,
	struct ibv_destroy_qp *cmd, size_t cmd_size,
	struct ibv_destroy_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, destroy_qp);
	return uv->ops->destroy_qp(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_post_send(fid_t fid,
	struct ibv_post_send *cmd, size_t cmd_size,
	struct ibv_post_send_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, post_send);
	return uv->ops->post_send(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_post_recv(fid_t fid,
	struct ibv_post_recv *cmd, size_t cmd_size,
	struct ibv_post_recv_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, post_recv);
	return uv->ops->post_recv(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_post_srq_recv(fid_t fid,
	struct ibv_post_srq_recv *cmd, size_t cmd_size,
	struct ibv_post_srq_recv_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, post_srq_recv);
	return uv->ops->post_srq_recv(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_attach_mcast(fid_t fid,
	struct ibv_attach_mcast *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, attach_mcast);
	return uv->ops->attach_mcast(fid, cmd, cmd_size);
}

static inline int
uv_detach_mcast(fid_t fid,
		struct ibv_detach_mcast *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, detach_mcast);
	return uv->ops->detach_mcast(fid, cmd, cmd_size);
}


#ifdef __cplusplus
}
#endif

#endif /* _FI_UVERBS_H_ */
