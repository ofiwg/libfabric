/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI send bandwidth benchmark */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <inttypes.h>
#include <err.h>

#include "libcxi.h"
#include "utils_common.h"

// clang-format off
#define MAX_BUF_SIZE_DFLT (1UL << 32) /* 4GiB */
#define BUF_ALIGN_DFLT    64
#define ITERS_DFLT        1000
#define SIZE_DFLT         65536
#define LIST_SIZE_DFLT    256

#define MATCH_BITS        0x123456789ABCDEF0
#define RDZV_IGNORE_BITS  0xF00 /* Ignore rdzv transaction type */
#define SRVR_RDZV_ID      0xAA
#define CLNT_RDZV_ID      0xCC
#define HEADER_DATA       0x77
#define USER_PTR          0x88
// clang-format on

static const char *name = "cxi_send_bw";
static const char *version = "2.4.0";

/* Increment the PUT event counter for either the next response or the next set
 * of puts
 */
int inc_put_ct(struct util_context *util, bool rsp_is_next)
{
	int rc = 0;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;

	if (rsp_is_next)
		util->tgt_ct_cmd.ct.threshold += 1;
	else if (opts->use_rdzv)
		util->tgt_ct_cmd.ct.threshold += (2 * opts->list_size);
	else
		util->tgt_ct_cmd.ct.threshold += opts->list_size;
	/* RDZV only uses CT with rsp */
	if (!opts->use_rdzv || rsp_is_next) {
		rc = cxi_cq_emit_ct(cxi->tgt_trig_cq, C_CMD_CT_TRIG_EVENT,
				    &util->tgt_ct_cmd.ct);
		if (rc) {
			fprintf(stderr,
				"Failed to issue RX CT_TRIG_EVENT command: %s\n",
				strerror(-rc));
			return rc;
		}
		cxi_cq_ring(cxi->tgt_trig_cq);
	}

	return rc;
}

/* Allocate TX resources */
int send_bw_alloc_tx(struct util_context *util)
{
	int rc;
	struct cxi_ctx_ini_opts ini_opts = { 0 };
	union c_cmdu c_st_cmd = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct ctrl_connection *ctrl = &util->ctrl;
	struct util_opts *opts = &util->opts;
	uint64_t rdzv_match_id;
	uint64_t initiator;
	uint64_t rdzv_match_bits;
	uint32_t flags;

	/* Config */
	ini_opts.alloc_ct = true;
	ini_opts.alloc_rdzv = opts->use_rdzv;

	ini_opts.eq_attr.queue_len = (opts->list_size + 1) * 64;
	ini_opts.eq_attr.queue_len =
		NEXT_MULTIPLE(ini_opts.eq_attr.queue_len, s_page_size);
	ini_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	if (opts->bidirectional || !ctrl->is_server) {
		ini_opts.buf_opts.length = opts->buf_size;
		ini_opts.buf_opts.hp = opts->hugepages;
	} else {
		ini_opts.buf_opts.length = s_page_size;
		ini_opts.buf_opts.hp = HP_DISABLED;
	}
	ini_opts.buf_opts.pattern = CTX_BUF_PAT_A5;
	ini_opts.use_gpu_buf = opts->loc_opts.use_tx_gpu;
	ini_opts.gpu_id = opts->loc_opts.tx_gpu;

	/* count granularity is 64B, and IDC commands can be up to 256B each
	 * with a preceding 64B CT State command. So worst case we need 4x + 1
	 */
	ini_opts.cq_opts.count = (opts->list_size * 4) + 1;
	ini_opts.cq_opts.flags = CXI_CQ_IS_TX;

	/* Allocate */
	rc = alloc_ini(cxi, &ini_opts);
	if (rc < 0)
		return rc;

	/* Physical matching (use_logical is not set) */
	initiator = CXI_MATCH_ID(cxi->dev->info.pid_bits, cxi->dom->pid,
				 cxi->dev->info.nid);

	/* Match DMA command setup */
	util->dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
	if (opts->use_rdzv) {
		util->dma_cmd.full_dma.command.opcode = C_CMD_RENDEZVOUS_PUT;
		util->dma_cmd.full_dma.match_bits = MATCH_BITS;
		if (!ctrl->is_server)
			util->dma_cmd.full_dma.rendezvous_id = CLNT_RDZV_ID;
		else
			util->dma_cmd.full_dma.rendezvous_id = SRVR_RDZV_ID;
		util->dma_cmd.full_dma.initiator = initiator;
		util->dma_cmd.full_dma.header_data = HEADER_DATA;
		util->dma_cmd.full_dma.user_ptr = USER_PTR;
	} else {
		util->dma_cmd.full_dma.command.opcode = C_CMD_PUT;
	}
	util->dma_cmd.full_dma.index_ext = cxi->index_ext;
	util->dma_cmd.full_dma.lac = cxi->ini_buf->md->lac;
	util->dma_cmd.full_dma.event_send_disable = 1;
	util->dma_cmd.full_dma.event_success_disable = 1;
	util->dma_cmd.full_dma.event_ct_ack = 1;
	util->dma_cmd.full_dma.dfa = cxi->dfa;
	util->dma_cmd.full_dma.eq = cxi->ini_eq->eqn;
	util->dma_cmd.full_dma.ct = cxi->ini_ct->ctn;

	/* IDC command setup */
	if (opts->use_idc && opts->min_size <= MAX_IDC_UNRESTRICTED_SIZE) {
		/* C State */
		c_st_cmd.c_state.command.opcode = C_CMD_CSTATE;
		c_st_cmd.c_state.index_ext = cxi->index_ext;
		c_st_cmd.c_state.event_send_disable = 1;
		c_st_cmd.c_state.event_success_disable = 1;
		c_st_cmd.c_state.event_ct_ack = 1;
		c_st_cmd.c_state.eq = cxi->ini_eq->eqn;
		c_st_cmd.c_state.ct = cxi->ini_ct->ctn;

		rc = cxi_cq_emit_c_state(cxi->ini_cq, &c_st_cmd.c_state);
		if (rc) {
			fprintf(stderr, "Failed to issue C State command: %s\n",
				strerror(-rc));
			return rc;
		}
		cxi_cq_ring(cxi->ini_cq);
		/* IDC Put */
		util->idc_cmd.idc_put.idc_header.command.opcode = C_CMD_PUT;
		util->idc_cmd.idc_put.idc_header.dfa = cxi->dfa;
	}

	/* CT Event command setup */
	util->ct_cmd.ct.eq = cxi->ini_eq->eqn;
	util->ct_cmd.ct.trig_ct = cxi->ini_ct->ctn;

	if (opts->use_rdzv) {
		rdzv_match_id = CXI_MATCH_ID(cxi->dev->info.pid_bits, C_PID_ANY,
					     cxi->rmt_addr.nic);
		if (!ctrl->is_server)
			rdzv_match_bits = (MATCH_BITS & 0xFFFF000000000000) |
				CLNT_RDZV_ID;
		else
			rdzv_match_bits = (MATCH_BITS & 0xFFFF000000000000) |
				SRVR_RDZV_ID;

		/* Append Persistent ME - RDZV */
		flags = C_LE_EVENT_UNLINK_DISABLE | C_LE_EVENT_SUCCESS_DISABLE |
			C_LE_EVENT_CT_COMM | C_LE_OP_GET;
		rc = append_me(cxi->ini_rdzv_pte_cq, cxi->ini_rdzv_eq,
			       cxi->ini_buf, 0, flags, cxi->ini_rdzv_pte->ptn,
			       cxi->ini_rdzv_ct->ctn, rdzv_match_id,
			       rdzv_match_bits, RDZV_IGNORE_BITS, 0);
		if (rc)
			return rc;

		/* TX RDZV CT Event command setup */
		util->ini_rdzv_ct_cmd.ct.eq = cxi->ini_rdzv_eq->eqn;
		util->ini_rdzv_ct_cmd.ct.trig_ct = cxi->ini_rdzv_ct->ctn;
	}

	return 0;
}

/* Allocate RDMA RX resources */
int send_bw_alloc_rx(struct util_context *util)
{
	int rc;
	struct cxi_ctx_tgt_opts tgt_opts = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct ctrl_connection *ctrl = &util->ctrl;
	struct util_opts *opts = &util->opts;
	uint64_t initiator;
	union c_fab_addr get_dfa;
	uint8_t get_index_ext;
	uint32_t flags;

	/* Config */
	tgt_opts.alloc_ct = true;

	tgt_opts.eq_attr.queue_len = (opts->list_size + 1) * 64;
	tgt_opts.eq_attr.queue_len =
		NEXT_MULTIPLE(tgt_opts.eq_attr.queue_len, s_page_size);
	if (opts->use_rdzv) {
		/* RDZV, PUT, REPLY per initiator RDZV */
		tgt_opts.eq_attr.queue_len *= 3;
		if (opts->bidirectional)
			/* Need more room for bidirectional */
			tgt_opts.eq_attr.queue_len *= 2;
	}
	tgt_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	if (opts->bidirectional || ctrl->is_server) {
		tgt_opts.buf_opts.length = opts->buf_size;
		tgt_opts.buf_opts.hp = opts->hugepages;
	} else {
		tgt_opts.buf_opts.length = s_page_size;
		tgt_opts.buf_opts.hp = HP_DISABLED;
	}
	tgt_opts.buf_opts.pattern = CTX_BUF_PAT_ZERO;
	tgt_opts.use_gpu_buf = opts->loc_opts.use_rx_gpu;
	tgt_opts.gpu_id = opts->loc_opts.rx_gpu;

	tgt_opts.cq_opts.count = 1;

	tgt_opts.pt_opts.is_matching = 1;

	/* Allocate */
	rc = alloc_tgt(cxi, &tgt_opts);
	if (rc < 0)
		return rc;

	/* Physical matching (use_logical is not set) */
	initiator = CXI_MATCH_ID(cxi->dev->info.pid_bits, cxi->rmt_addr.pid,
				 cxi->rmt_addr.nic);

	/* Append Persistent ME */
	/* RDZV events won't be suppressed by this */
	flags = C_LE_EVENT_UNLINK_DISABLE | C_LE_EVENT_SUCCESS_DISABLE |
		C_LE_EVENT_CT_COMM | C_LE_OP_PUT;
	if (opts->use_rdzv) {
		rc = append_me(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0, flags,
			       cxi->tgt_pte->ptn, cxi->tgt_ct->ctn, initiator,
			       MATCH_BITS, 0, 0);
	} else {
		flags |= C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO;
		rc = append_le(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0, flags,
			       cxi->tgt_pte->ptn, cxi->tgt_ct->ctn, 0);
	}
	if (rc)
		return rc;

	/* CT Event command setup */
	util->tgt_ct_cmd.ct.eq = cxi->tgt_eq->eqn;
	util->tgt_ct_cmd.ct.trig_ct = cxi->tgt_ct->ctn;

	/* Prepare counter to capture first set of PUTs */
	rc = inc_put_ct(util, (!opts->bidirectional && !ctrl->is_server));
	if (rc)
		return rc;

	/* SW Get command setup */
	if (opts->use_rdzv) {
		cxi_build_dfa(cxi->rmt_addr.nic, cxi->rmt_addr.pid,
			      cxi->dev->info.pid_bits, C_PID_ANY, &get_dfa,
			      &get_index_ext);
		initiator = CXI_MATCH_ID(cxi->dev->info.pid_bits,
					 cxi->loc_addr.pid, cxi->loc_addr.nic);

		util->tgt_rdzv_get_cmd.command.cmd_type = C_CMD_TYPE_DMA;
		util->tgt_rdzv_get_cmd.full_dma.command.opcode = C_CMD_GET;
		util->tgt_rdzv_get_cmd.full_dma.dfa = get_dfa;
		util->tgt_rdzv_get_cmd.full_dma.index_ext = get_index_ext;
		util->tgt_rdzv_get_cmd.full_dma.initiator = initiator;
		util->tgt_rdzv_get_cmd.full_dma.lac = cxi->tgt_buf->md->lac;
		util->tgt_rdzv_get_cmd.full_dma.event_send_disable = 1;
		util->tgt_rdzv_get_cmd.full_dma.event_ct_reply = 1;
		util->tgt_rdzv_get_cmd.full_dma.eq = cxi->tgt_eq->eqn;
		util->tgt_rdzv_get_cmd.full_dma.ct = cxi->tgt_ct->ctn;
	}

	return 0;
}

/* Send list_size ops and wait for their ACKs/REPLYs */
int do_single_send(struct util_context *util)
{
	int rc = 0;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	int i;
	uint64_t rmt_offset;
	uint64_t local_addr;
	uint64_t rdzv_get_tmo;

	/* Had to issue SW Rdzv Get, pick up where we left off */
	if (util->istate == RESUME_SEND)
		goto resume_send;

	if (util->istate != SEND)
		return 0;

	rc = inc_ct(cxi->ini_trig_cq, &util->ct_cmd.ct, opts->list_size);
	if (rc)
		return rc;

	if (opts->use_rdzv) {
		rc = inc_ct(cxi->ini_rdzv_trig_cq, &util->ini_rdzv_ct_cmd.ct,
			    opts->list_size);
		if (rc)
			return rc;
	}

	rmt_offset = 0;
	local_addr = (uintptr_t)cxi->ini_buf->buf;

	/* Enqueue TX command and ring doorbell */
	for (i = 0; i < opts->list_size; i++) {
		if (!opts->use_idc || util->size > MAX_IDC_UNRESTRICTED_SIZE) {
			util->dma_cmd.full_dma.request_len = util->size;
			util->dma_cmd.full_dma.remote_offset = rmt_offset;
			util->dma_cmd.full_dma.local_addr =
				CXI_VA_TO_IOVA(cxi->ini_buf->md, local_addr);
			/* Rsp is PUT, switch back to RDZV */
			if (opts->use_rdzv)
				util->dma_cmd.full_dma.command.opcode =
					C_CMD_RENDEZVOUS_PUT;
			rc = cxi_cq_emit_dma(cxi->ini_cq,
					     &util->dma_cmd.full_dma);
			if (rc) {
				fprintf(stderr,
					"Failed to issue DMA command: %s\n",
					strerror(-rc));
				return rc;
			}
		} else {
			util->idc_cmd.idc_put.idc_header.remote_offset =
				rmt_offset;
			rc = cxi_cq_emit_idc_put(cxi->ini_cq,
						 &util->idc_cmd.idc_put,
						 (void *)local_addr,
						 util->size);
			if (rc) {
				fprintf(stderr,
					"Failed to issue IDC command: %s\n",
					strerror(-rc));
				return rc;
			}
		}

		inc_tx_buf_offsets(util, &rmt_offset, &local_addr);
	}
	cxi_cq_ring(cxi->ini_cq);

	/* Wait for ACK Event(s) */
	rc = wait_for_ct(cxi->ini_eq, NO_TIMEOUT, "initiator ACK");
	if (rc)
		return rc;

resume_send:
	if (opts->use_rdzv) {
		/* Wait for RDZV GET Event(s) */
		rdzv_get_tmo = opts->bidirectional ? POLL_ONCE : NO_TIMEOUT;
		rc = wait_for_ct(cxi->ini_rdzv_eq, rdzv_get_tmo, "RDZV GET");
		/* Check if SW-issued Rdzv Get is needed */
		if (rc == -ETIME && opts->bidirectional) {
			util->istate = SINGLE_RECV;
			rc = EAGAIN;
		}
	}
	if (rc)
		return rc;

	if (opts->bidirectional)
		util->istate = RECV;
	else
		util->istate = RESP;

	return rc;
}

/* Wait for single PUT response to confirm target reception of PUTs */
int do_single_recv_rsp(struct util_context *util)
{
	int rc = 0;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;

	if (util->istate != RESP)
		return 0;

	/* Wait for PUT Event */
	/* This causes problems on duration runs where the client or server
	 * can do one extra iteration, so instead of waiting with NO_TIMEOUT
	 * try once and continue so the duration can complete.
	 */
	rc = wait_for_ct(cxi->tgt_eq, POLL_ONCE, "response PUT");
	if (rc == -ETIME)
		return EAGAIN;
	else if (rc)
		return rc;

	/* Prepare for the next set of PUTs */
	rc = inc_put_ct(util, !opts->bidirectional);
	if (rc)
		return rc;

	util->istate = DONE;

	return rc;
}

/* Receive list_size PUTs and send PUT response to confirm reception */
int do_single_recv(struct util_context *util)
{
	int rc = 0;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	uint64_t rmt_offset;
	uint64_t local_addr;
	static uint16_t rx_ev_cnt;
	uint16_t local_ev_cnt = 0;
	const union c_event *event = NULL;

	if (util->istate != RECV && util->istate != SINGLE_RECV)
		return 0;

	/* Wait for PUT Event(s) */
	/* This causes problems on duration runs where the client or server
	 * can do one extra iteration, so instead of waiting with NO_TIMEOUT
	 * try once and continue so the duration can complete.
	 */
	if (!opts->use_rdzv) {
		rc = wait_for_ct(cxi->tgt_eq, POLL_ONCE, "target PUT");
		if (rc == -ETIME)
			return EAGAIN;
		else if (!rc)
			rx_ev_cnt++;
	} else {
		/* Wait for PUT, RENDEZVOUS, and REPLY events */
		while (rx_ev_cnt < (3 * opts->list_size) &&
		       (event = cxi_eq_get_event(cxi->tgt_eq)) != NULL) {
			rx_ev_cnt++;
			local_ev_cnt++;
			if (event->hdr.event_type == C_EVENT_RENDEZVOUS) {
				if (!event->tgt_long.get_issued) {
					/* Manually issue Get */
					rc = sw_rdzv_get(util, event->tgt_long);
					if (rc)
						break;
				}
			}
		}
		if (local_ev_cnt)
			cxi_eq_ack_events(cxi->tgt_eq);
		if (!rc && rx_ev_cnt < (3 * opts->list_size))
			rc = EAGAIN;
	}
	if (rc >= 0 && util->istate == SINGLE_RECV) {
		util->istate = RESUME_SEND;
		rc = EAGAIN;
	}
	if (rc)
		return rc;

	rc = inc_ct(cxi->ini_trig_cq, &util->ct_cmd.ct, 1);
	if (rc)
		return rc;

	/* Prepare for the next set of PUTs */
	rc = inc_put_ct(util, opts->bidirectional);
	if (rc)
		return rc;

	rmt_offset = 0;
	local_addr = (uintptr_t)cxi->ini_buf->buf;

	/* Enqueue TX command and ring doorbell */
	if (!opts->use_idc || util->size > MAX_IDC_UNRESTRICTED_SIZE) {
		util->dma_cmd.full_dma.request_len = 0;
		util->dma_cmd.full_dma.remote_offset = rmt_offset;
		util->dma_cmd.full_dma.local_addr =
			CXI_VA_TO_IOVA(cxi->ini_buf->md, local_addr);
		/* Cmd last used for RDZV, switch to PUT */
		if (opts->use_rdzv)
			util->dma_cmd.full_dma.command.opcode = C_CMD_PUT;
		rc = cxi_cq_emit_dma(cxi->ini_cq, &util->dma_cmd.full_dma);
		if (rc) {
			fprintf(stderr, "Failed to issue DMA command: %s\n",
				strerror(-rc));
			return rc;
		}
	} else {
		util->idc_cmd.idc_put.idc_header.remote_offset = rmt_offset;
		rc = cxi_cq_emit_idc_put(cxi->ini_cq, &util->idc_cmd.idc_put,
					 (void *)local_addr, 0);
		if (rc) {
			fprintf(stderr, "Failed to issue IDC RSP command: %s\n",
				strerror(-rc));
			return rc;
		}
	}
	cxi_cq_ring(cxi->ini_cq);

	/* Wait for ACK Event */
	rc = wait_for_ct(cxi->ini_eq, NO_TIMEOUT, "response ACK");
	if (rc)
		return rc;

	if (opts->bidirectional)
		util->istate = RESP;
	else
		util->istate = DONE;

	rx_ev_cnt = 0;
	return rc;
}

/* Send list_size ops and wait for their ACKs/REPLYs */
int do_single_iteration(struct util_context *util)
{
	int rc = 0;
	struct ctrl_connection *ctrl = &util->ctrl;
	struct util_opts *opts = &util->opts;

	if (util->istate == DONE) {
		if (!opts->bidirectional && ctrl->is_server)
			util->istate = RECV;
		else
			util->istate = SEND;
	}

	rc = do_single_send(util);
	if (rc)
		return rc;
	rc = do_single_recv(util);
	if (rc)
		return rc;
	rc = do_single_recv_rsp(util);
	if (rc)
		return rc;

	if (!rc && util->istate != DONE)
		rc = EAGAIN;

	return rc;
}

/* Complete any unfinished RDZV transfers */
int wait_for_rdzv_done(struct util_context *util)
{
	int rc = 0;
	int retries = 0;
	struct ctrl_connection *ctrl = &util->ctrl;

	while (true) {
		if (util->istate == DONE) {
			/*
			 * Handle an edge case where only one of the peers
			 * started a new iteration before the duration
			 * time had expired.  A sync timeout here indicates
			 * the peer has a transfer in progress that needs to
			 * be cleared.  Only retry once.
			 */
			rc = ctrl_barrier(ctrl, DFLT_HANDSHAKE_TIMEOUT,
					  "RDZV cleanup");
			if (retries++ ||(rc != -EAGAIN && rc != -ETIMEDOUT))
				break;
			fprintf(stderr, "Retrying...\n");
		}

		rc = do_single_iteration(util);
		if (rc != 0) {
			if (rc != EAGAIN) {
				fprintf(stderr, "Iteration failed: %s\n",
					strerror(abs(rc)));
				return rc;
			}
			usleep(1000);
		}
	}

	return rc;
}

void usage(void)
{
	printf("Usage:\n");
	printf("  cxi_send_bw [-d DEV] [-p PORT]\n");
	printf("                         Start a server and wait for connection\n");
	printf("  cxi_send_bw ADDR [OPTIONS]\n");
	printf("                         Connect to server with address ADDR\n");
	printf("\n");
	printf("Options:\n");
	printf("  -d, --device=DEV       Device name (default: \"cxi0\")\n");
	printf("  -v, --svc-id=SVC_ID    Service ID (default: 1)\n");
	printf("  -p, --port=PORT        The port to listen on/connect to (default: %u)\n",
	       DFLT_PORT);
	printf("  -t, --tx-gpu=GPU       GPU index for allocating TX buf (default: no GPU)\n");
	printf("  -r, --rx-gpu=GPU       GPU index for allocating RX buf (default: no GPU)\n");
	printf("  -g, --gpu-type=TYPE    GPU type (AMD or NVIDIA or INTEL) (default type determined\n");
	printf("                         by discovered GPU files on system)\n");
	printf("  -n, --iters=ITERS      Number of iterations to run (default: %u)\n",
	       ITERS_DFLT);
	printf("  -D, --duration=SEC     Run for the specified number of seconds\n");
	printf("  -s, --size=MIN[:MAX]   Send size or range to use (default: %u)\n",
	       SIZE_DFLT);
	printf("                         Ranges must be powers of 2 (e.g. \"1:8192\")\n");
	printf("                         The maximum size is %lu\n",
	       MAX_MSG_SIZE);
	printf("  -l, --list-size=SIZE   Number of sends per iteration, all pushed to\n");
	printf("                         the Tx CQ prior to initiating xfer (default: %u)\n",
	       LIST_SIZE_DFLT);
	printf("  -b, --bidirectional    Measure bidirectional bandwidth\n");
	printf("  -R, --rdzv             Use rendezvous PUTs\n");
	printf("      --no-idc           Do not use Immediate Data Cmds for sizes <= %u bytes\n",
	       MAX_IDC_UNRESTRICTED_SIZE);
	printf("      --buf-sz=SIZE      The max TX/RDMA buffer size, specified in bytes\n");
	printf("                         If (size * list_size) > buf_sz, sends will overlap\n");
	printf("                         (default: %lu)\n", MAX_BUF_SIZE_DFLT);
	printf("      --buf-align=ALIGN  Byte-alignment of sends in the buffer (default: %u)\n",
	       BUF_ALIGN_DFLT);
	printf("      --use-hp=SIZE      Attempt to use huge pages when allocating\n");
	printf("                         Size may be \"2M\" or \"1G\"\n");
	printf("  -h, --help             Print this help text and exit\n");
	printf("  -V, --version          Print the version and exit\n");
}

int main(int argc, char **argv)
{
	int c;
	int rc;
	struct util_context util = { 0 };
	struct ctrl_connection *ctrl = &util.ctrl;
	struct cxi_context *cxi = &util.cxi;
	struct util_opts *opts = &util.opts;
	int count;
	int num_hp;

	opts->loc_opts.svc_id = CXI_DEFAULT_SVC_ID;
	opts->loc_opts.port = DFLT_PORT;
	opts->iters = ITERS_DFLT;
	opts->min_size = SIZE_DFLT;
	opts->max_size = SIZE_DFLT;
	opts->list_size = LIST_SIZE_DFLT;
	opts->max_buf_size = MAX_BUF_SIZE_DFLT;
	opts->buf_align = BUF_ALIGN_DFLT;
	opts->use_idc = 1;

	/* Set default GPU type based on discovered GPU files. */
#if HAVE_HIP_SUPPORT
	opts->loc_opts.gpu_type = AMD;
#elif defined(HAVE_CUDA_SUPPORT)
	opts->loc_opts.gpu_type = NVIDIA;
#elif defined(HAVE_ZE_SUPPORT)
	opts->loc_opts.gpu_type = INTEL;
#else
	opts->loc_opts.gpu_type = -1;
#endif

	struct option longopts[] = {
		{ "no-idc", no_argument, NULL, VAL_NO_IDC },
		{ "buf-sz", required_argument, NULL, VAL_BUF_SZ },
		{ "buf-align", required_argument, NULL, VAL_BUF_ALIGN },
		{ "use-hp", required_argument, NULL, VAL_USE_HP },
		{ "device", required_argument, NULL, 'd' },
		{ "svc-id", required_argument, NULL, 'v' },
		{ "port", required_argument, NULL, 'p' },
		{ "tx-gpu", required_argument, NULL, 't' },
		{ "rx-gpu", required_argument, NULL, 'r' },
		{ "gpu-type", required_argument, NULL, 'g' },
		{ "iters", required_argument, NULL, 'n' },
		{ "duration", required_argument, NULL, 'D' },
		{ "size", required_argument, NULL, 's' },
		{ "list-size", required_argument, NULL, 'l' },
		{ "bidirectional", no_argument, NULL, 'b' },
		{ "emu-mode", no_argument, NULL, VAL_EMU_MODE },
		{ "rdzv", no_argument, NULL, 'R' },
		{ "help", no_argument, NULL, 'h' },
		{ "version", no_argument, NULL, 'V' },
		{ NULL, 0, NULL, 0 }
	};

	while (1) {
		c = getopt_long(argc, argv, "d:v:p:t:r:g:n:D:s:l:bhVR",
				longopts, NULL);
		if (c == -1)
			break;
		parse_common_opt(c, opts, name, version, usage);
	}
	if (opts->use_rdzv)
		opts->use_idc = 0;
	if (opts->max_size > opts->max_buf_size)
		errx(1, "Max RDMA buffer size (%lu) < max send size (%lu)!",
		     opts->max_buf_size, opts->max_size);
	if ((opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu) &&
	    opts->loc_opts.gpu_type < 0)
		errx(1, "Invalid GPU type or unable to find GPU libraries");

	parse_server_addr(argc, argv, ctrl, opts->loc_opts.port);

	/* Allocate CXI context */
	rc = ctx_alloc(cxi, opts->loc_opts.dev_id, opts->loc_opts.svc_id);
	if (rc < 0)
		goto cleanup;

	/* Connect to peer, pass client opts to server */
	rc = ctrl_connect(ctrl, name, version, opts, &cxi->loc_addr,
			  &cxi->rmt_addr);
	if (rc < 0)
		goto cleanup;

	/* Determine hugepages needed and check if available */
	opts->buf_size = NEXT_MULTIPLE(opts->max_size, opts->buf_align);
	opts->buf_size *= opts->list_size;
	if (opts->buf_size > opts->max_buf_size)
		opts->buf_size = opts->max_buf_size;
	num_hp = get_hugepages_needed(opts,
		(opts->bidirectional || !ctrl->is_server),
		(opts->bidirectional || ctrl->is_server));

	/* Print out the configuration */
	snprintf(util.header, MAX_HDR_LEN, "%*s  %*s  %*s  %*s", SIZE_W,
		 "Send Size[B]", COUNT_W, "Sends", BW_W, "BW[MB/s]", MRATE_W,
		 "PktRate[Mpkt/s]");
	print_separator(strlen(util.header));
	printf("    CXI RDMA Send Bandwidth Test\n");
	print_loc_opts(opts, ctrl->is_server);
	if (opts->duration) {
		printf("Test Type        : Duration\n");
		printf("Duration         : %u seconds\n", opts->duration);
	} else {
		printf("Test Type        : Iteration\n");
		printf("Iterations       : %lu\n", opts->iters);
	}
	if (opts->min_size == opts->max_size) {
		printf("Send Size        : %lu\n", opts->max_size);
	} else {
		printf("Min Send Size    : %lu\n", opts->min_size);
		printf("Max Send Size    : %lu\n", opts->max_size);
	}
	printf("List Size        : %u\n", opts->list_size);
	printf("IDC              : %s\n",
	       opts->use_idc ? "Enabled" : "Disabled");
	printf("Bidirectional    : %s\n",
	       opts->bidirectional ? "Enabled" : "Disabled");
	printf("Rendezvous PUTs  : %s\n",
	       opts->use_rdzv ? "Enabled" : "Disabled");
	printf("Max RDMA Buf     : %lu (%lu used)\n", opts->max_buf_size,
	       opts->buf_size);
	printf("RDMA Buf Align   : %lu\n", opts->buf_align);
	print_hugepage_opts(opts, num_hp);
	printf("Local (%s)   : NIC 0x%x PID %u VNI %u\n",
	       ctrl->is_server ? "server" : "client", cxi->loc_addr.nic,
	       cxi->loc_addr.pid, cxi->vni);
	printf("Remote (%s)  : NIC 0x%x PID %u%s\n",
	       ctrl->is_server ? "client" : "server", cxi->rmt_addr.nic,
	       cxi->rmt_addr.pid, ctrl->is_loopback ? " (loopback)" : "");

	// Initialize reading NIC hardware counters
	if (opts->emu_mode) {
		rc = init_time_counters(cxi->dev);
		if (rc) {
			goto cleanup;
		}
	}

	/* Initialize GPU library if we are using GPU memory */
	if (opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu) {
		rc = gpu_lib_init(opts->loc_opts.gpu_type);
		if (rc < 0)
			goto cleanup;
		count = get_gpu_device_count();
		printf("Found %d GPU(s)\n", count);
	}

	/* Allocate remaining CXI resources */
	rc = send_bw_alloc_tx(&util);
	if (rc < 0)
		goto cleanup;
	rc = send_bw_alloc_rx(&util);
	if (rc < 0)
		goto cleanup;

	/* Signal ready */
	rc = ctrl_barrier(ctrl, NO_TIMEOUT,
			  ctrl->is_server ? "Client" : "Server");
	if (rc)
		goto cleanup;

	print_separator(strlen(util.header));
	printf("%s\n", util.header);

	for (util.size = opts->min_size; util.size <= opts->max_size;
	     util.size *= 2) {
		util.buf_granularity =
			NEXT_MULTIPLE(util.size, opts->buf_align);
		util.istate = DONE;
		rc = run_bw_active(&util, do_single_iteration);
		if (rc)
			break;

		if (opts->min_size != opts->max_size) {
			/*
			 * Specifying a duration along with bidirectional rendezvous
			 * transfers can leave incomplete transfers in progress.  Clear
			 * any incomplete transfers before starting the next size interval.
			 */
			if (opts->duration && opts->use_rdzv && opts->bidirectional) {
				rc = wait_for_rdzv_done(&util);
				if (rc)
					break;
			}
		}
	}
	print_separator(strlen(util.header));

cleanup:
	ctx_destroy(cxi);
	ctrl_close(ctrl);
	if (opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu)
		gpu_lib_fini(opts->loc_opts.gpu_type);
	return rc ? 1 : 0;
}
