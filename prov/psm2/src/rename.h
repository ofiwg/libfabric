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

#ifndef _FI_PSM_RENAME_H_
#define _FI_PSM_RENAME_H_

#define psmx_active_fabric psmx2_active_fabric
#define psmx_am_ack_rma psmx2_am_ack_rma
#define psmx_am_atomic_completion psmx2_am_atomic_completion
#define psmx_am_atomic_handler psmx2_am_atomic_handler
#define psmx_am_enqueue_recv psmx2_am_enqueue_recv
#define psmx_am_enqueue_rma psmx2_am_enqueue_rma
#define psmx_am_enqueue_send psmx2_am_enqueue_send
#define psmx_am_enqueue_unexp psmx2_am_enqueue_unexp
#define psmx_am_fini psmx2_am_fini
#define psmx_am_handlers psmx2_am_handlers
#define psmx_am_handlers_idx psmx2_am_handlers_idx
#define psmx_am_handlers_initialized psmx2_am_handlers_initialized
#define psmx_am_init psmx2_am_init
#define psmx_am_msg_handler psmx2_am_msg_handler
#define psmx_am_param psmx2_am_param
#define psmx_am_process_rma psmx2_am_process_rma
#define psmx_am_process_send psmx2_am_process_send
#define psmx_am_progress psmx2_am_progress
#define psmx_am_request psmx2_am_request
#define psmx_am_rma_handler psmx2_am_rma_handler
#define psmx_am_search_and_dequeue_recv psmx2_am_search_and_dequeue_recv
#define psmx_am_search_and_dequeue_unexp psmx2_am_search_and_dequeue_unexp
#define _psmx_atomic_compwrite _psmx2_atomic_compwrite
#define psmx_atomic_compwrite psmx2_atomic_compwrite
#define psmx_atomic_compwritemsg psmx2_atomic_compwritemsg
#define psmx_atomic_compwritev psmx2_atomic_compwritev
#define psmx_atomic_compwritevalid psmx2_atomic_compwritevalid
#define psmx_atomic_do_compwrite psmx2_atomic_do_compwrite
#define psmx_atomic_do_readwrite psmx2_atomic_do_readwrite
#define psmx_atomic_do_write psmx2_atomic_do_write
#define psmx_atomic_fini psmx2_atomic_fini
#define psmx_atomic_init psmx2_atomic_init
#define psmx_atomic_inject psmx2_atomic_inject
#define psmx_atomic_lock psmx2_atomic_lock
#define psmx_atomic_ops psmx2_atomic_ops
#define _psmx_atomic_readwrite _psmx2_atomic_readwrite
#define psmx_atomic_readwrite psmx2_atomic_readwrite
#define psmx_atomic_readwritemsg psmx2_atomic_readwritemsg
#define psmx_atomic_readwritev psmx2_atomic_readwritev
#define psmx_atomic_readwritevalid psmx2_atomic_readwritevalid
#define psmx_atomic_self psmx2_atomic_self
#define _psmx_atomic_write _psmx2_atomic_write
#define psmx_atomic_write psmx2_atomic_write
#define psmx_atomic_writemsg psmx2_atomic_writemsg
#define psmx_atomic_writev psmx2_atomic_writev
#define psmx_atomic_writevalid psmx2_atomic_writevalid
#define psmx_av_bind psmx2_av_bind
#define psmx_av_check_table_size psmx2_av_check_table_size
#define psmx_av_close psmx2_av_close
#define psmx_av_insert psmx2_av_insert
#define psmx_av_lookup psmx2_av_lookup
#define psmx_av_open psmx2_av_open
#define psmx_av_ops psmx2_av_ops
#define psmx_av_remove psmx2_av_remove
#define psmx_av_straddr psmx2_av_straddr
#define psmx_cm_getname psmx2_cm_getname
#define psmx_cm_ops psmx2_cm_ops
#define psmx_cntr_add psmx2_cntr_add
#define psmx_cntr_add_trigger psmx2_cntr_add_trigger
#define psmx_cntr_check_trigger psmx2_cntr_check_trigger
#define psmx_cntr_close psmx2_cntr_close
#define psmx_cntr_control psmx2_cntr_control
#define psmx_cntr_inc psmx2_cntr_inc
#define psmx_cntr_open psmx2_cntr_open
#define psmx_cntr_ops psmx2_cntr_ops
#define psmx_cntr_read psmx2_cntr_read
#define psmx_cntr_readerr psmx2_cntr_readerr
#define psmx_cntr_set psmx2_cntr_set
#define psmx_cntr_wait psmx2_cntr_wait
#define psmx_context_type psmx2_context_type
#define psmx_cq_alloc_event psmx2_cq_alloc_event
#define psmx_cq_close psmx2_cq_close
#define psmx_cq_control psmx2_cq_control
#define psmx_cq_create_event psmx2_cq_create_event
#define psmx_cq_create_event_from_status psmx2_cq_create_event_from_status
#define psmx_cq_dequeue_event psmx2_cq_dequeue_event
#define psmx_cq_enqueue_event psmx2_cq_enqueue_event
#define psmx_cq_event psmx2_cq_event
#define psmx_cq_free_event psmx2_cq_free_event
#define psmx_cq_get_event_src_addr psmx2_cq_get_event_src_addr
#define psmx_cq_open psmx2_cq_open
#define psmx_cq_ops psmx2_cq_ops
#define psmx_cq_poll_mq psmx2_cq_poll_mq
#define psmx_cq_read psmx2_cq_read
#define psmx_cq_readerr psmx2_cq_readerr
#define psmx_cq_readfrom psmx2_cq_readfrom
#define psmx_cq_signal psmx2_cq_signal
#define psmx_cq_sread psmx2_cq_sread
#define psmx_cq_sreadfrom psmx2_cq_sreadfrom
#define psmx_cq_strerror psmx2_cq_strerror
#define psmx_domain_check_features psmx2_domain_check_features
#define psmx_domain_close psmx2_domain_close
#define psmx_domain_disable_ep psmx2_domain_disable_ep
#define psmx_domain_enable_ep psmx2_domain_enable_ep
#define psmx_domain_open psmx2_domain_open
#define psmx_domain_ops psmx2_domain_ops
#define psmx_domain_start_progress psmx2_domain_start_progress
#define psmx_domain_stop_progress psmx2_domain_stop_progress
#define psmx_env psmx2_env
#define psmx_epaddr_context psmx2_epaddr_context
#define psmx_ep_bind psmx2_ep_bind
#define psmx_ep_cancel psmx2_ep_cancel
#define psmx_ep_close psmx2_ep_close
#define psmx_ep_control psmx2_ep_control
#define psmx_ep_getopt psmx2_ep_getopt
#define psmx_epid_to_epaddr psmx2_epid_to_epaddr
#define psmx_ep_open psmx2_ep_open
#define psmx_ep_ops psmx2_ep_ops
#define psmx_ep_optimize_ops psmx2_ep_optimize_ops
#define psmx_ep_setopt psmx2_ep_setopt
#define psmx_eq_alloc_event psmx2_eq_alloc_event
#define psmx_eq_close psmx2_eq_close
#define psmx_eq_control psmx2_eq_control
#define psmx_eq_create_event psmx2_eq_create_event
#define psmx_eq_dequeue_error psmx2_eq_dequeue_error
#define psmx_eq_dequeue_event psmx2_eq_dequeue_event
#define psmx_eq_enqueue_event psmx2_eq_enqueue_event
#define psmx_eq_event psmx2_eq_event
#define psmx_eq_free_event psmx2_eq_free_event
#define psmx_eq_open psmx2_eq_open
#define psmx_eq_ops psmx2_eq_ops
#define psmx_eq_peek_event psmx2_eq_peek_event
#define psmx_eq_read psmx2_eq_read
#define psmx_eq_readerr psmx2_eq_readerr
#define psmx_eq_sread psmx2_eq_sread
#define psmx_eq_strerror psmx2_eq_strerror
#define psmx_eq_write psmx2_eq_write
#define psmx_errno psmx2_errno
#define psmx_errno_table psmx2_errno_table
#define psmx_fabric psmx2_fabric
#define psmx_fabric_close psmx2_fabric_close
#define psmx_fabric_fi_ops psmx2_fabric_fi_ops
#define psmx_fabric_ops psmx2_fabric_ops
#define psmx_fid_av psmx2_fid_av
#define psmx_fid_cntr psmx2_fid_cntr
#define psmx_fid_cq psmx2_fid_cq
#define psmx_fid_domain psmx2_fid_domain
#define psmx_fid_ep psmx2_fid_ep
#define psmx_fid_eq psmx2_fid_eq
#define psmx_fid_fabric psmx2_fid_fabric
#define psmx_fid_mr psmx2_fid_mr
#define psmx_fid_poll psmx2_fid_poll
#define psmx_fid_stx psmx2_fid_stx
#define psmx_fid_wait psmx2_fid_wait
#define psmx_fini psmx2_fini
#define psmx_fi_ops psmx2_fi_ops
#define psmx_fi_ops_stx psmx2_fi_ops_stx
#define psmx_getinfo psmx2_getinfo
#define psmx_get_uuid psmx2_get_uuid
#define psmx_info psmx2_info
#define psmx_init_count psmx2_init_count
#define psmx_init_env psmx2_init_env
#define psmx_inject psmx2_inject
#define psmx_inject2 psmx2_inject2
#define psmx_mr_bind psmx2_mr_bind
#define psmx_mr_close psmx2_mr_close
#define psmx_mr_hash psmx2_mr_hash
#define psmx_mr_hash_add psmx2_mr_hash_add
#define psmx_mr_hash_del psmx2_mr_hash_del
#define psmx_mr_hash_entry psmx2_mr_hash_entry
#define psmx_mr_hash_get psmx2_mr_hash_get
#define psmx_mr_normalize_iov psmx2_mr_normalize_iov
#define psmx_mr_ops psmx2_mr_ops
#define psmx_mr_reg psmx2_mr_reg
#define psmx_mr_regattr psmx2_mr_regattr
#define psmx_mr_regv psmx2_mr_regv
#define psmx_mr_validate psmx2_mr_validate
#define psmx_msg2_ops psmx2_msg2_ops
#define psmx_msg_ops psmx2_msg_ops
#define psmx_multi_recv psmx2_multi_recv
#define psmx_name_server psmx2_name_server
#define psmx_name_server_cleanup psmx2_name_server_cleanup
#define psmx_pi psmx2_pi
#define psmx_poll_add psmx2_poll_add
#define psmx_poll_close psmx2_poll_close
#define psmx_poll_del psmx2_poll_del
#define psmx_poll_list psmx2_poll_list
#define psmx_poll_open psmx2_poll_open
#define psmx_poll_ops psmx2_poll_ops
#define psmx_poll_poll psmx2_poll_poll
#define psmx_process_trigger psmx2_process_trigger
#define psmx_progress psmx2_progress
#define psmx_progress_func psmx2_progress_func
#define psmx_progress_set_affinity psmx2_progress_set_affinity
#define psmx_prov psmx2_prov
#define psmx_query_mpi psmx2_query_mpi
#define _psmx_read _psmx2_read
#define psmx_read psmx2_read
#define psmx_readmsg psmx2_readmsg
#define psmx_readv psmx2_readv
#define _psmx_recv _psmx2_recv
#define psmx_recv psmx2_recv
#define _psmx_recv2 _psmx2_recv2
#define psmx_recv2 psmx2_recv2
#define psmx_recvmsg psmx2_recvmsg
#define psmx_recvmsg2 psmx2_recvmsg2
#define psmx_recvv psmx2_recvv
#define psmx_recvv2 psmx2_recvv2
#define psmx_req_queue psmx2_req_queue
#define psmx_reserve_tag_bits psmx2_reserve_tag_bits
#define psmx_resolve_name psmx2_resolve_name
#define psmx_rma_ops psmx2_rma_ops
#define psmx_rma_self psmx2_rma_self
#define psmx_rx_size_left psmx2_rx_size_left
#define _psmx_send _psmx2_send
#define psmx_send psmx2_send
#define _psmx_send2 _psmx2_send2
#define psmx_send2 psmx2_send2
#define psmx_senddata psmx2_senddata
#define psmx_sendmsg psmx2_sendmsg
#define psmx_sendmsg2 psmx2_sendmsg2
#define psmx_sendv psmx2_sendv
#define psmx_sendv2 psmx2_sendv2
#define psmx_set_epaddr_context psmx2_set_epaddr_context
#define psmx_string_to_uuid psmx2_string_to_uuid
#define psmx_stx_close psmx2_stx_close
#define psmx_stx_ctx psmx2_stx_ctx
#define psmx_tagged_inject psmx2_tagged_inject
#define psmx_tagged_inject_no_flag_av_map psmx2_tagged_inject_no_flag_av_map
#define psmx_tagged_inject_no_flag_av_table psmx2_tagged_inject_no_flag_av_table
#define psmx_tagged_ops psmx2_tagged_ops
#define psmx_tagged_ops_no_event_av_map psmx2_tagged_ops_no_event_av_map
#define psmx_tagged_ops_no_event_av_table psmx2_tagged_ops_no_event_av_table
#define psmx_tagged_ops_no_flag_av_map psmx2_tagged_ops_no_flag_av_map
#define psmx_tagged_ops_no_flag_av_table psmx2_tagged_ops_no_flag_av_table
#define psmx_tagged_ops_no_recv_event_av_map psmx2_tagged_ops_no_recv_event_av_map
#define psmx_tagged_ops_no_recv_event_av_table psmx2_tagged_ops_no_recv_event_av_table
#define psmx_tagged_ops_no_send_event_av_map psmx2_tagged_ops_no_send_event_av_map
#define psmx_tagged_ops_no_send_event_av_table psmx2_tagged_ops_no_send_event_av_table
#define _psmx_tagged_peek _psmx2_tagged_peek
#define _psmx_tagged_recv _psmx2_tagged_recv
#define psmx_tagged_recv psmx2_tagged_recv
#define psmx_tagged_recvmsg psmx2_tagged_recvmsg
#define psmx_tagged_recv_no_event psmx2_tagged_recv_no_event
#define psmx_tagged_recv_no_event_av_map psmx2_tagged_recv_no_event_av_map
#define psmx_tagged_recv_no_event_av_table psmx2_tagged_recv_no_event_av_table
#define psmx_tagged_recv_no_flag psmx2_tagged_recv_no_flag
#define psmx_tagged_recv_no_flag_av_map psmx2_tagged_recv_no_flag_av_map
#define psmx_tagged_recv_no_flag_av_table psmx2_tagged_recv_no_flag_av_table
#define psmx_tagged_recvv psmx2_tagged_recvv
#define psmx_tagged_recvv_no_event psmx2_tagged_recvv_no_event
#define psmx_tagged_recvv_no_flag psmx2_tagged_recvv_no_flag
#define _psmx_tagged_send _psmx2_tagged_send
#define psmx_tagged_send psmx2_tagged_send
#define psmx_tagged_senddata psmx2_tagged_senddata
#define psmx_tagged_sendmsg psmx2_tagged_sendmsg
#define psmx_tagged_send_no_event_av_map psmx2_tagged_send_no_event_av_map
#define psmx_tagged_send_no_event_av_table psmx2_tagged_send_no_event_av_table
#define psmx_tagged_send_no_flag_av_map psmx2_tagged_send_no_flag_av_map
#define psmx_tagged_send_no_flag_av_table psmx2_tagged_send_no_flag_av_table
#define psmx_tagged_sendv psmx2_tagged_sendv
#define psmx_tagged_sendv_no_event_av_map psmx2_tagged_sendv_no_event_av_map
#define psmx_tagged_sendv_no_event_av_table psmx2_tagged_sendv_no_event_av_table
#define psmx_tagged_sendv_no_flag_av_map psmx2_tagged_sendv_no_flag_av_map
#define psmx_tagged_sendv_no_flag_av_table psmx2_tagged_sendv_no_flag_av_table
#define psmx_trigger psmx2_trigger
#define psmx_triggered_op psmx2_triggered_op
#define psmx_tx_size_left psmx2_tx_size_left
#define psmx_unexp psmx2_unexp
#define psmx_uuid_to_port psmx2_uuid_to_port
#define psmx_uuid_to_string psmx2_uuid_to_string
#define psmx_wait_close psmx2_wait_close
#define psmx_wait_cond psmx2_wait_cond
#define psmx_wait_get_obj psmx2_wait_get_obj
#define psmx_wait_init psmx2_wait_init
#define psmx_wait_mutex psmx2_wait_mutex
#define psmx_wait_open psmx2_wait_open
#define psmx_wait_ops psmx2_wait_ops
#define psmx_wait_progress psmx2_wait_progress
#define psmx_wait_signal psmx2_wait_signal
#define psmx_wait_start_progress psmx2_wait_start_progress
#define psmx_wait_stop_progress psmx2_wait_stop_progress
#define psmx_wait_thread psmx2_wait_thread
#define psmx_wait_thread_busy psmx2_wait_thread_busy
#define psmx_wait_thread_enabled psmx2_wait_thread_enabled
#define psmx_wait_thread_ready psmx2_wait_thread_ready
#define psmx_wait_wait psmx2_wait_wait
#define _psmx_write _psmx2_write
#define psmx_write psmx2_write
#define psmx_writedata psmx2_writedata
#define psmx_writemsg psmx2_writemsg
#define psmx_writev psmx2_writev

#endif
