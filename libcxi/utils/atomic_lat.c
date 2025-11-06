/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI Atomic Memory Operation latency benchmark */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <inttypes.h>
#include <err.h>
#include <stdbool.h>
#include <math.h>

#include "libcxi.h"
#include "utils_common.h"

#define BUF_ALIGN 64
#define ITERS_DFLT 100
#define WARMUP_DFLT 10
#define DELAY_DFLT 1000
#define OP_DFLT C_AMO_OP_SUM
#define CSWAP_OP_DFLT C_AMO_OP_CSWAP_EQ
#define TYPE_DFLT C_AMO_TYPE_UINT64_T

static const char *name = "cxi_atomic_lat";
static const char *version = "2.4.0";

/* Allocate INI resources */
int atomic_lat_alloc_ini(struct util_context *util)
{
	int rc;
	struct cxi_ctx_ini_opts ini_opts = { 0 };
	union c_cmdu c_st_cmd = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;

	/* Config */
	ini_opts.eq_attr.queue_len = s_page_size;
	ini_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	ini_opts.buf_opts.length = BUF_ALIGN;
	if (opts->fetching) {
		util->fetch_offset = ini_opts.buf_opts.length;
		ini_opts.buf_opts.length *= 2;
	}
	ini_opts.buf_opts.pattern = CTX_BUF_PAT_NONE;

	/* count granularity is 64B, and IDC commands can be up to 256B each
	 * with a preceding 64B CT State command. So worst case we need 5
	 */
	ini_opts.cq_opts.count = 5;
	ini_opts.cq_opts.flags = CXI_CQ_IS_TX;

	/* Allocate */
	rc = alloc_ini(cxi, &ini_opts);
	if (rc < 0)
		return rc;

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
		if (!opts->unrestricted)
			util->dma_cmd.dma_amo.restricted = 1;
		util->dma_cmd.dma_amo.dfa = cxi->dfa;
		util->dma_cmd.dma_amo.local_read_addr =
			CXI_VA_TO_IOVA(cxi->ini_buf->md, cxi->ini_buf->buf);
		if (opts->fetching)
			util->dma_cmd.dma_amo.local_write_addr =
				CXI_VA_TO_IOVA(cxi->ini_buf->md,
					       ((uintptr_t)cxi->ini_buf->buf +
						util->fetch_offset));
		util->dma_cmd.dma_amo.eq = cxi->ini_eq->eqn;
		if (opts->matching)
			util->dma_cmd.dma_amo.match_bits = 1;
	} else {
		/* C State */
		c_st_cmd.c_state.command.opcode = C_CMD_CSTATE;
		c_st_cmd.c_state.index_ext = cxi->index_ext;
		c_st_cmd.c_state.event_send_disable = 1;
		if (!opts->unrestricted)
			c_st_cmd.c_state.restricted = 1;
		c_st_cmd.c_state.eq = cxi->ini_eq->eqn;
		if (opts->fetching)
			c_st_cmd.c_state.write_lac = cxi->ini_buf->md->lac;

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
		if (opts->fetching)
			util->idc_cmd.idc_amo.local_addr =
				CXI_VA_TO_IOVA(cxi->ini_buf->md,
					       ((uintptr_t)cxi->ini_buf->buf +
						util->fetch_offset));
	}

	return 0;
}

/* Allocate AMO TGT resources */
int atomic_lat_alloc_tgt(struct util_context *util)
{
	int rc;
	struct cxi_ctx_tgt_opts tgt_opts = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	uint32_t flags;
	uint64_t match_bits;

	/* Config */
	tgt_opts.eq_attr.queue_len = s_page_size;
	tgt_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	tgt_opts.buf_opts.length = BUF_ALIGN;
	tgt_opts.buf_opts.pattern = CTX_BUF_PAT_NONE;

	tgt_opts.cq_opts.count = 1;

	tgt_opts.pt_opts.is_matching = opts->matching ? 1 : 0;

	/* Allocate */
	rc = alloc_tgt(cxi, &tgt_opts);
	if (rc < 0)
		return rc;

	/* Set initial target buffer values */
	amo_init_tgt_op(util);

	/* Append Persistent LE */
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

/* Send a single AMO and measure the time until its ACK or REPLY arrives */
int do_single_iteration(struct util_context *util)
{
	int rc = 0;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	struct timespec lat_ts_0;
	struct timespec lat_ts_1;

	/* Get start time */
	rc = clock_gettime_or_counters(CLOCK_MONOTONIC_RAW, &lat_ts_0,
				       cxi->dev);
	if (rc < 0) {
		rc = -errno;
		fprintf(stderr, "clock_gettime() failed: %s\n", strerror(-rc));
		return rc;
	}

	/* Enqueue initiator command and ring doorbell */
	if (opts->use_idc) {
		rc = cxi_cq_emit_idc_amo(cxi->ini_cq, &util->idc_cmd.idc_amo,
					 opts->fetching);
		if (rc) {
			fprintf(stderr, "Failed to issue IDC AMO command: %s\n",
				strerror(-rc));
			return rc;
		}
	} else {
		rc = cxi_cq_emit_dma_amo(cxi->ini_cq, &util->dma_cmd.dma_amo,
					 opts->fetching);
		if (rc) {
			fprintf(stderr, "Failed to issue DMA AMO command: %s\n",
				strerror(-rc));
			return rc;
		}
	}
	if (opts->use_ll)
		cxi_cq_ll_ring(cxi->ini_cq);
	else
		cxi_cq_ring(cxi->ini_cq);

	/* Get end time when write ACK or REPLY is received */
	rc = get_event(cxi->ini_eq,
		       (opts->fetching ? C_EVENT_REPLY : C_EVENT_ACK), NULL,
		       &lat_ts_1, NO_TIMEOUT, cxi->dev);
	if (rc) {
		fprintf(stderr, "Failed to get initiator ACK: %s\n",
			strerror(-rc));
		return rc;
	}

	util->last_lat = TV_NSEC_DIFF(lat_ts_0, lat_ts_1);

	/* Update operands to maximize target writes */
	amo_update_op1(util);
	amo_update_op2(util);

	return rc;
}

void usage(void)
{
	printf("Usage:\n");
	printf("  cxi_atomic_lat [-d DEV] [-p PORT]\n");
	printf("                          Start a server and wait for connection\n");
	printf("  cxi_atomic_lat ADDR [OPTIONS]\n");
	printf("                          Connect to server with address ADDR\n");
	printf("\n");
	printf("Options:\n");
	printf("  -d, --device=DEV        Device name (default: \"cxi0\")\n");
	printf("  -v, --svc-id=SVC_ID     Service ID (default: 1)\n");
	printf("  -p, --port=PORT         The port to listen on/connect to (default: %u)\n",
	       DFLT_PORT);
	printf("  -n, --iters=ITERS       Number of iterations to run (default: %u)\n",
	       ITERS_DFLT);
	printf("  -D, --duration=SEC      Run for the specified number of seconds\n");
	printf("      --warmup=WARMUP     Number of warmup iterations to run (default: %u)\n",
	       WARMUP_DFLT);
	printf("      --latency-gap=USEC  Number of microseconds to wait between each\n");
	printf("                          iteration (default: %u)\n",
	       DELAY_DFLT);
	printf("  -A, --atomic-op         The atomic operation to use (default: %s)\n",
	       amo_op_strs[OP_DFLT]);
	printf("  -C, --cswap-op          The CSWAP operation to use (default: %s)\n",
	       amo_cswap_op_strs[CSWAP_OP_DFLT]);
	printf("  -T, --atomic-type       The atomic type to use (default: %s)\n",
	       amo_type_strs[TYPE_DFLT]);
	printf("      --fetching          Use fetching AMOs\n");
	printf("      --matching          Use matching list entries at the target\n");
	printf("      --unrestricted      Use unrestricted AMOs\n");
	printf("      --no-idc            Do not use Immediate Data Cmds\n");
	printf("      --no-ll             Do not use Low-Latency command issuing\n");
	printf("      --report-all        Report all latency measurements individually\n");
	printf("  -h, --help              Print this help text and exit\n");
	printf("  -V, --version           Print the version and exit\n");
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
	bool idc_opt_override = false;
	bool unrestrict_opt_override = false;

	opts->loc_opts.svc_id = CXI_DEFAULT_SVC_ID;
	opts->loc_opts.port = DFLT_PORT;
	opts->iters = ITERS_DFLT;
	opts->warmup = WARMUP_DFLT;
	opts->iter_delay = DELAY_DFLT;
	opts->atomic_op = OP_DFLT;
	opts->cswap_op = CSWAP_OP_DFLT;
	opts->atomic_type = TYPE_DFLT;
	opts->use_idc = 1;
	opts->use_ll = 1;

	struct option longopts[] = {
		{ "fetching", no_argument, NULL, VAL_FETCHING },
		{ "matching", no_argument, NULL, VAL_MATCHING },
		{ "unrestricted", no_argument, NULL, VAL_UNRESTRICTED },
		{ "no-idc", no_argument, NULL, VAL_NO_IDC },
		{ "no-ll", no_argument, NULL, VAL_NO_LL },
		{ "report-all", no_argument, NULL, VAL_REPORT_ALL },
		{ "warmup", required_argument, NULL, VAL_WARMUP },
		{ "latency-gap", required_argument, NULL, VAL_ITER_DELAY },
		{ "device", required_argument, NULL, 'd' },
		{ "svc-id", required_argument, NULL, 'v' },
		{ "port", required_argument, NULL, 'p' },
		{ "iters", required_argument, NULL, 'n' },
		{ "duration", required_argument, NULL, 'D' },
		{ "emu-mode", no_argument, NULL, VAL_EMU_MODE },
		{ "atomic-op", required_argument, NULL, 'A' },
		{ "cswap-op", required_argument, NULL, 'C' },
		{ "atomic-type", required_argument, NULL, 'T' },
		{ "help", no_argument, NULL, 'h' },
		{ "version", no_argument, NULL, 'V' },
		{ NULL, 0, NULL, 0 }
	};

	while (1) {
		c = getopt_long(argc, argv, "d:v:p:n:D:A:C:T:hV", longopts,
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
	if (opts->use_idc && opts->matching) {
		opts->use_idc = 0;
		idc_opt_override = true;
	}

	/* Print out the configuration */
	snprintf(util.header, MAX_HDR_LEN, "%*s  %*s  %*s  %*s  %*s  %*s",
		 SIZE_W, "AMO Size[B]", COUNT_W, "Ops", LAT_W, "Min[us]", LAT_W,
		 "Max[us]", LAT_W, "Mean[us]", LAT_W, "StdDev[us]");
	print_separator(strlen(util.header));
	printf("    CXI Atomic Memory Operation Latency Test\n");
	printf("Device           : cxi%u\n", opts->loc_opts.dev_id);
	printf("Service ID       : %u\n", opts->loc_opts.svc_id);
	if (opts->duration) {
		printf("Test Type        : Duration\n");
		printf("Duration         : %u seconds\n", opts->duration);
	} else {
		printf("Test Type        : Iteration\n");
		printf("Iterations       : %lu\n", opts->iters);
	}
	printf("Warmup Iters     : %lu\n", opts->warmup);
	printf("Inter-Iter Gap   : %lu microseconds\n", opts->iter_delay);
	printf("Atomic Op        : %sFETCHING %s %s\n",
	       opts->fetching ? "" : "NON-", amo_op_strs[opts->atomic_op],
	       (opts->atomic_op == C_AMO_OP_CSWAP ?
	       amo_cswap_op_strs[opts->cswap_op] : ""));
	printf("Atomic Type      : %s\n", amo_type_strs[opts->atomic_type]);
	printf("IDC              : %s\n", idc_opt_override ?
	       "Disabled - Not Applicable" :
	       opts->use_idc ? "Enabled" : "Disabled");
	printf("Matching LEs     : %s\n",
	       opts->matching ? "Enabled" : "Disabled");
	printf("Restricted       : %s\n", unrestrict_opt_override ?
	       "Disabled - Not Applicable" :
	       opts->unrestricted ? "Disabled" : "Enabled");
	printf("LL Cmd Launch    : %s\n",
	       opts->use_ll ? "Enabled" : "Disabled");
	printf("Results Reported : %s\n", opts->report_all ? "All" : "Summary");
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

	/* Allocate remaining CXI resources */
	if (!ctrl->is_server) {
		rc = atomic_lat_alloc_ini(&util);
		if (rc < 0)
			goto cleanup;
	} else {
		rc = atomic_lat_alloc_tgt(&util);
		if (rc < 0)
			goto cleanup;
	}

	/* Signal ready */
	rc = ctrl_barrier(ctrl, NO_TIMEOUT,
			  ctrl->is_server ? "Client" : "Server");
	if (rc)
		goto cleanup;

	if (!ctrl->is_server && !opts->report_all) {
		print_separator(strlen(util.header));
		printf("%s\n", util.header);
	}

	util.size = amo_type_sizes[opts->atomic_type];
	if (!ctrl->is_server)
		rc = run_lat_active(&util, do_single_iteration);
	else
		rc = run_lat_passive(&util);

	print_separator(strlen(util.header));

cleanup:
	ctx_destroy(cxi);
	ctrl_close(ctrl);
	return rc ? 1 : 0;
}
