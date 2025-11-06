/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI Atomic Memory Operation bandwidth benchmark */

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
#include <stdbool.h>

#include "libcxi.h"
#include "utils_common.h"

#define BUF_ALIGN 64
#define ITERS_DFLT 10000
#define LIST_SIZE_DFLT 4096
#define OP_DFLT C_AMO_OP_SUM
#define CSWAP_OP_DFLT C_AMO_OP_CSWAP_EQ
#define TYPE_DFLT C_AMO_TYPE_UINT64_T

#define CAS_1_DEV_ID 0x0501

/* Errata 3236 - HRP 32-bit Non-fetching AMO using VS packet type dropped by
 * IXE. Affects Cassini v1.0 and v1.1.
 */
#define NONFETCH_U32_HRP_DISABLE_MAX_REV 2

static const char *name = "cxi_atomic_bw";
static const char *version = "2.4.0";

/* Allocate INI resources */
int atomic_bw_alloc_ini(struct util_context *util)
{
	int rc;
	struct cxi_ctx_ini_opts ini_opts = { 0 };
	union c_cmdu c_st_cmd = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;

	/* Config */
	ini_opts.alloc_hrp = opts->use_hrp;
	ini_opts.alloc_ct = true;

	ini_opts.eq_attr.queue_len = (opts->list_size + 1) * 64;
	ini_opts.eq_attr.queue_len =
		NEXT_MULTIPLE(ini_opts.eq_attr.queue_len, s_page_size);
	ini_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	ini_opts.buf_opts.length = BUF_ALIGN * opts->list_size;
	if (opts->fetching) {
		util->fetch_offset = ini_opts.buf_opts.length;
		ini_opts.buf_opts.length *= 2;
	}
	ini_opts.buf_opts.pattern = CTX_BUF_PAT_NONE;

	/* count granularity is 64B, and IDC commands can be up to 256B each
	 * with a preceding 64B CT State command. So worst case we need 4x + 1
	 */
	ini_opts.cq_opts.count = (opts->list_size * 4) + 1;
	ini_opts.cq_opts.flags = CXI_CQ_IS_TX;

	/* Allocate */
	rc = alloc_ini(cxi, &ini_opts);
	if (rc < 0)
		return rc;

	if (opts->use_hrp) {
		rc = set_to_hrp_cp(util, cxi->ini_cq);
		if (rc)
			return rc;
	}

	/* Operands and pattern initiator buffer */
	amo_init_op1(util);
	amo_init_op2(util);

	if (!opts->use_idc) {
		/* DMA AMO */
		util->dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
		util->dma_cmd.dma_amo.atomic_op = opts->atomic_op;
		util->dma_cmd.dma_amo.cswap_op = opts->cswap_op;
		util->dma_cmd.dma_amo.atomic_type = opts->atomic_type;
		util->dma_cmd.dma_amo.index_ext = cxi->index_ext;
		util->dma_cmd.dma_amo.lac = cxi->ini_buf->md->lac;
		if (opts->fetching)
			util->dma_cmd.dma_amo.write_lac = cxi->ini_buf->md->lac;
		util->dma_cmd.dma_amo.event_send_disable = 1;
		util->dma_cmd.dma_amo.event_success_disable = 1;
		if (opts->fetching)
			util->dma_cmd.dma_amo.event_ct_reply = 1;
		else
			util->dma_cmd.dma_amo.event_ct_ack = 1;
		if (!opts->unrestricted)
			util->dma_cmd.dma_amo.restricted = 1;
		util->dma_cmd.dma_amo.dfa = cxi->dfa;
		util->dma_cmd.dma_amo.eq = cxi->ini_eq->eqn;
		if (opts->matching)
			util->dma_cmd.dma_amo.match_bits = 1;
		util->dma_cmd.dma_amo.ct = cxi->ini_ct->ctn;
	} else {
		/* C State */
		c_st_cmd.c_state.command.opcode = C_CMD_CSTATE;
		c_st_cmd.c_state.index_ext = cxi->index_ext;
		c_st_cmd.c_state.event_send_disable = 1;
		c_st_cmd.c_state.event_success_disable = 1;
		if (opts->fetching)
			c_st_cmd.c_state.event_ct_reply = 1;
		else
			c_st_cmd.c_state.event_ct_ack = 1;
		if (!opts->unrestricted)
			c_st_cmd.c_state.restricted = 1;
		c_st_cmd.c_state.eq = cxi->ini_eq->eqn;
		if (opts->fetching)
			c_st_cmd.c_state.write_lac = cxi->ini_buf->md->lac;
		c_st_cmd.c_state.ct = cxi->ini_ct->ctn;

		rc = cxi_cq_emit_c_state(cxi->ini_cq, &c_st_cmd.c_state);
		if (rc) {
			fprintf(stderr, "Failed to issue C State command: %s\n",
				strerror(-rc));
			return rc;
		}
		cxi_cq_ring(cxi->ini_cq);
		/* IDC AMO */
		util->idc_cmd.idc_amo.idc_header.dfa = cxi->dfa;
		util->idc_cmd.idc_amo.atomic_op = opts->atomic_op;
		util->idc_cmd.idc_amo.cswap_op = opts->cswap_op;
		util->idc_cmd.idc_amo.atomic_type = opts->atomic_type;
	}

	/* CT Event command setup */
	util->ct_cmd.ct.eq = cxi->ini_eq->eqn;
	util->ct_cmd.ct.trig_ct = cxi->ini_ct->ctn;

	return 0;
}

/* Allocate AMO TGT resources */
int atomic_bw_alloc_tgt(struct util_context *util)
{
	int rc;
	struct cxi_ctx_tgt_opts tgt_opts = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	uint32_t flags;
	uint64_t match_bits;

	/* Config */
	if (opts->matching) {
		tgt_opts.eq_attr.queue_len = (opts->list_size + 1) * 64;
		tgt_opts.eq_attr.queue_len =
			NEXT_MULTIPLE(tgt_opts.eq_attr.queue_len, s_page_size);
	} else {
		tgt_opts.eq_attr.queue_len = s_page_size;
	}
	tgt_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	tgt_opts.buf_opts.length = BUF_ALIGN * opts->list_size;
	tgt_opts.buf_opts.pattern = CTX_BUF_PAT_NONE;

	tgt_opts.cq_opts.count = 1;

	tgt_opts.pt_opts.is_matching = opts->matching ? 1 : 0;

	/* Allocate */
	rc = alloc_tgt(cxi, &tgt_opts);
	if (rc < 0)
		return rc;

	/* Set initial target buffer values */
	amo_init_tgt_op(util);

	/* Append Persistent LE/ME */
	flags = C_LE_EVENT_SUCCESS_DISABLE | C_LE_EVENT_COMM_DISABLE |
		C_LE_EVENT_UNLINK_DISABLE | C_LE_UNRESTRICTED_BODY_RO |
		C_LE_UNRESTRICTED_END_RO | C_LE_OP_PUT;
	match_bits = 1;
	if (opts->fetching)
		flags |= C_LE_OP_GET;
	if (opts->matching)
		rc = append_me(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0, flags,
			       cxi->tgt_pte->ptn, 0, 0, match_bits, 0, 0);
	else
		rc = append_le(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0, flags,
			       cxi->tgt_pte->ptn, 0, 0);
	if (rc)
		return rc;

	return 0;
}

/* Send list_size AMOs and wait for their REPLY/ACKs */
int do_single_iteration(struct util_context *util)
{
	int rc = 0;
	int i;
	uint64_t rmt_offset;
	uint64_t local_read_addr;
	uint64_t local_write_addr;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;

	rc = inc_ct(cxi->ini_trig_cq, &util->ct_cmd.ct, opts->list_size);
	if (rc)
		return rc;

	rmt_offset = 0;
	local_read_addr = (uintptr_t)cxi->ini_buf->buf;
	if (opts->fetching)
		local_write_addr = local_read_addr + util->fetch_offset;
	else
		local_write_addr = 0;

	/* Enqueue initiator command and ring doorbell */
	for (i = 0; i < opts->list_size; i++) {
		if (!opts->use_idc) {
			util->dma_cmd.dma_amo.remote_offset = rmt_offset;
			util->dma_cmd.dma_amo.local_read_addr = CXI_VA_TO_IOVA(
				cxi->ini_buf->md, local_read_addr);
			if (opts->fetching)
				util->dma_cmd.dma_amo.local_write_addr =
					CXI_VA_TO_IOVA(cxi->ini_buf->md,
						       local_write_addr);
			rc = cxi_cq_emit_dma_amo(cxi->ini_cq,
						 &util->dma_cmd.dma_amo,
						 opts->fetching);
			if (rc) {
				fprintf(stderr,
					"Failed to issue DMA AMO command: %s\n",
					strerror(-rc));
				return rc;
			}
		} else {
			util->idc_cmd.idc_amo.idc_header.remote_offset =
				rmt_offset;
			if (opts->fetching)
				util->idc_cmd.idc_amo.local_addr =
					CXI_VA_TO_IOVA(cxi->ini_buf->md,
						       local_write_addr);
			rc = cxi_cq_emit_idc_amo(cxi->ini_cq,
						 &util->idc_cmd.idc_amo,
						 opts->fetching);
			if (rc) {
				fprintf(stderr,
					"Failed to issue IDC AMO command: %s\n",
					strerror(-rc));
				return rc;
			}
		}

		inc_tx_buf_offsets(util, &rmt_offset, &local_read_addr);
		if (opts->fetching)
			local_write_addr = local_read_addr + util->fetch_offset;
	}
	cxi_cq_ring(cxi->ini_cq);

	/* Wait for ACK or REPLY event(s) */
	rc = wait_for_ct(cxi->ini_eq, NO_TIMEOUT, "initiator");
	if (rc)
		return rc;

	/* Update operands to maximize target writes */
	amo_update_op1(util);
	amo_update_op2(util);

	return rc;
}

void usage(void)
{
	printf("Usage:\n");
	printf("  cxi_atomic_bw [-d DEV] [-p PORT]\n");
	printf("                         Start a server and wait for connection\n");
	printf("  cxi_atomic_bw ADDR [OPTIONS]\n");
	printf("                         Connect to server with address ADDR\n");
	printf("\n");
	printf("Options:\n");
	printf("  -d, --device=DEV       Device name (default: \"cxi0\")\n");
	printf("  -v, --svc-id=SVC_ID    Service ID (default: 1)\n");
	printf("  -p, --port=PORT        The port to listen on/connect to (default: %u)\n",
	       DFLT_PORT);
	printf("  -n, --iters=ITERS      Number of iterations to run (default: %u)\n",
	       ITERS_DFLT);
	printf("  -D, --duration=SEC     Run for the specified number of seconds\n");
	printf("  -l, --list-size=SIZE   Number of writes per iteration, all pushed to the\n");
	printf("                         initiator CQ prior to initiating xfer (default: %u)\n",
	       LIST_SIZE_DFLT);
	printf("  -A, --atomic-op        The atomic operation to use (default: %s)\n",
	       amo_op_strs[OP_DFLT]);
	printf("  -C, --cswap-op         The CSWAP operation to use (default: %s)\n",
	       amo_cswap_op_strs[CSWAP_OP_DFLT]);
	printf("  -T, --atomic-type      The atomic type to use (default: %s)\n",
	       amo_type_strs[TYPE_DFLT]);
	printf("  -b, --bidirectional    Measure bidirectional bandwidth\n");
	printf("      --fetching         Use fetching AMOs\n");
	printf("      --matching         Use matching list entries at the target\n");
	printf("      --unrestricted     Use unrestricted AMOs\n");
	printf("      --no-hrp           Do not use High Rate Puts\n");
	printf("      --no-idc           Do not use Immediate Data Cmds\n");
	printf("  -h, --help             Print this help text and exit\n");
	printf("  -V, --version          Print the version and exit\n");
	printf("\n");
	printf("Atomic Ops:\n");
	printf("  MIN, MAX, SUM, LOR, LAND, BOR, BAND, LXOR, BXOR, SWAP, CSWAP, AXOR\n");
	printf("CSWAP Ops:\n");
	printf("  EQ, NE, LE, LT, GE, GT\n");
	printf("Atomic Types:\n");
	printf("  INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, DOUBLE,\n");
	printf("  FLOAT_COMPLEX, DOUBLE_COMPLEX, UINT128\n");
}

int main(int argc, char **argv)
{
	int c;
	int rc;
	struct util_context util = { 0 };
	struct ctrl_connection *ctrl = &util.ctrl;
	struct cxi_context *cxi = &util.cxi;
	struct util_opts *opts = &util.opts;
	bool hrp_opt_override = false;
	bool idc_opt_override = false;
	bool unrestrict_opt_override = false;

	opts->loc_opts.svc_id = CXI_DEFAULT_SVC_ID;
	opts->loc_opts.port = DFLT_PORT;
	opts->iters = ITERS_DFLT;
	opts->list_size = LIST_SIZE_DFLT;
	opts->atomic_op = OP_DFLT;
	opts->cswap_op = CSWAP_OP_DFLT;
	opts->atomic_type = TYPE_DFLT;
	opts->use_hrp = 1;
	opts->use_idc = 1;

	struct option longopts[] = {
		{ "fetching", no_argument, NULL, VAL_FETCHING },
		{ "matching", no_argument, NULL, VAL_MATCHING },
		{ "unrestricted", no_argument, NULL, VAL_UNRESTRICTED },
		{ "no-hrp", no_argument, NULL, VAL_NO_HRP },
		{ "no-idc", no_argument, NULL, VAL_NO_IDC },
		{ "device", required_argument, NULL, 'd' },
		{ "svc-id", required_argument, NULL, 'v' },
		{ "port", required_argument, NULL, 'p' },
		{ "iters", required_argument, NULL, 'n' },
		{ "duration", required_argument, NULL, 'D' },
		{ "atomic-op", required_argument, NULL, 'A' },
		{ "cswap-op", required_argument, NULL, 'C' },
		{ "atomic-type", required_argument, NULL, 'T' },
		{ "list-size", required_argument, NULL, 'l' },
		{ "bidirectional", no_argument, NULL, 'b' },
		{ "emu-mode", no_argument, NULL, VAL_EMU_MODE },
		{ "help", no_argument, NULL, 'h' },
		{ "version", no_argument, NULL, 'V' },
		{ NULL, 0, NULL, 0 }
	};

	while (1) {
		c = getopt_long(argc, argv, "d:v:p:n:D:A:C:T:l:bhV", longopts,
				NULL);
		if (c == -1)
			break;
		parse_common_opt(c, opts, name, version, usage);
	}
	amo_validate_op_and_type(opts->atomic_op, opts->cswap_op,
				 opts->atomic_type);

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

	/* Some options are mutually exclusive */
	if (!opts->unrestricted && opts->matching) {
		opts->unrestricted = 1;
		unrestrict_opt_override = true;
	}
	if (opts->use_hrp &&
	    (opts->fetching || opts->unrestricted ||
	     (cxi->dev->info.device_id == CAS_1_DEV_ID &&
	      cxi->dev->info.device_rev <= NONFETCH_U32_HRP_DISABLE_MAX_REV &&
	      opts->atomic_type == C_AMO_TYPE_UINT32_T))) {
		opts->use_hrp = 0;
		hrp_opt_override = true;
	}
	if (opts->use_idc && opts->matching) {
		opts->use_idc = 0;
		idc_opt_override = true;
	}

	/* Print out the configuration */
	snprintf(util.header, MAX_HDR_LEN, "%*s  %*s  %*s  %*s", SIZE_W,
		 "AMO Size[B]", COUNT_W, "Ops", BW_W, "BW[MB/s]", MRATE_W,
		 "OpRate[M/s]");
	print_separator(strlen(util.header));
	printf("    CXI Atomic Memory Operation Bandwidth Test\n");
	printf("Device          : cxi%u\n", opts->loc_opts.dev_id);
	printf("Service ID      : %u\n", opts->loc_opts.svc_id);
	if (opts->duration) {
		printf("Test Type       : Duration\n");
		printf("Duration        : %u seconds\n", opts->duration);
	} else {
		printf("Test Type       : Iteration\n");
		printf("Iterations      : %lu\n", opts->iters);
	}
	printf("Atomic Op       : %sFETCHING %s %s\n",
	       opts->fetching ? "" : "NON-", amo_op_strs[opts->atomic_op],
	       (opts->atomic_op == C_AMO_OP_CSWAP ?
	       amo_cswap_op_strs[opts->cswap_op] :
	       ""));
	printf("Atomic Type     : %s\n", amo_type_strs[opts->atomic_type]);
	printf("List Size       : %u\n", opts->list_size);
	printf("HRP             : %s\n",
	       hrp_opt_override ? "Disabled - Not Applicable" :
	       opts->use_hrp ? "Enabled" : "Disabled");
	printf("IDC             : %s\n", idc_opt_override ?
	       "Disabled - Not Applicable" :
	       opts->use_idc ? "Enabled" : "Disabled");
	printf("Matching LEs    : %s\n",
	       opts->matching ? "Enabled" : "Disabled");
	printf("Restricted      : %s\n", unrestrict_opt_override ?
	       "Disabled - Not Applicable" :
	       opts->unrestricted ? "Disabled" : "Enabled");
	printf("Bidirectional   : %s\n",
	       opts->bidirectional ? "Enabled" : "Disabled");
	printf("Local (%s)  : NIC 0x%x PID %u VNI %u\n",
	       ctrl->is_server ? "server" : "client", cxi->loc_addr.nic,
	       cxi->loc_addr.pid, cxi->vni);
	printf("Remote (%s) : NIC 0x%x PID %u%s\n",
	       ctrl->is_server ? "client" : "server", cxi->rmt_addr.nic,
	       cxi->rmt_addr.pid, ctrl->is_loopback ? " (loopback)" : "");

	// Initialize reading NIC hardware counters
	if (opts->emu_mode) {
		rc = init_time_counters(cxi->dev);
		if (rc) {
			goto cleanup;
		}
	}

	/* Allocate remaining CXI resources */
	if (!ctrl->is_server || opts->bidirectional) {
		rc = atomic_bw_alloc_ini(&util);
		if (rc < 0)
			goto cleanup;
	}
	if (ctrl->is_server || opts->bidirectional) {
		rc = atomic_bw_alloc_tgt(&util);
		if (rc < 0)
			goto cleanup;
	}

	/* Signal ready */
	rc = ctrl_barrier(ctrl, NO_TIMEOUT,
			  ctrl->is_server ? "Client" : "Server");
	if (rc)
		goto cleanup;

	print_separator(strlen(util.header));
	printf("%s\n", util.header);

	util.buf_granularity = BUF_ALIGN;
	util.size = amo_type_sizes[opts->atomic_type];
	if (!ctrl->is_server || opts->bidirectional)
		rc = run_bw_active(&util, do_single_iteration);
	else
		rc = run_bw_passive(&util);

	print_separator(strlen(util.header));

cleanup:
	ctx_destroy(cxi);
	ctrl_close(ctrl);
	return rc ? 1 : 0;
}
