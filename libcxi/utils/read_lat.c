/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI RDMA read latency benchmark */

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
#include <time.h>
#include <math.h>

#include "libcxi.h"
#include "utils_common.h"

#define ITERS_DFLT 100
#define WARMUP_DFLT 10
#define SIZE_DFLT 8
#define DELAY_DFLT 1000

static const char *name = "cxi_read_lat";
static const char *version = "2.4.0";

/* Allocate resources */
int read_lat_alloc_ini(struct util_context *util)
{
	int rc;
	struct cxi_ctx_ini_opts ini_opts = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;

	/* Config */
	ini_opts.eq_attr.queue_len = s_page_size;
	ini_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	ini_opts.buf_opts.length = opts->buf_size;
	ini_opts.buf_opts.pattern = CTX_BUF_PAT_A5;
	ini_opts.buf_opts.hp = opts->hugepages;
	ini_opts.use_gpu_buf = opts->loc_opts.use_tx_gpu;
	ini_opts.gpu_id = opts->loc_opts.tx_gpu;

	/* count granularity is 64B, and IDC commands can be up to 256B each
	 * with a preceding 64B CT State command. So worst case we need 5
	 */
	ini_opts.cq_opts.count = 5;
	ini_opts.cq_opts.flags = CXI_CQ_IS_TX;

	/* Allocate */
	rc = alloc_ini(cxi, &ini_opts);
	if (rc < 0)
		return rc;

	/* Nomatch DMA Get command setup */
	util->dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
	util->dma_cmd.nomatch_dma.command.opcode = C_CMD_NOMATCH_GET;
	util->dma_cmd.nomatch_dma.index_ext = cxi->index_ext;
	util->dma_cmd.nomatch_dma.lac = cxi->ini_buf->md->lac;
	util->dma_cmd.nomatch_dma.event_send_disable = 1;
	if (!opts->unrestricted)
		util->dma_cmd.nomatch_dma.restricted = 1;
	util->dma_cmd.nomatch_dma.dfa = cxi->dfa;
	util->dma_cmd.nomatch_dma.local_addr =
		CXI_VA_TO_IOVA(cxi->ini_buf->md, cxi->ini_buf->buf);
	util->dma_cmd.nomatch_dma.eq = cxi->ini_eq->eqn;

	/* IDC command setup - invalid for Gets */

	return 0;
}

/* Allocate resources */
int read_lat_alloc_tgt(struct util_context *util)
{
	int rc;
	struct cxi_ctx_tgt_opts tgt_opts = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	uint32_t flags;

	/* Config */
	tgt_opts.eq_attr.queue_len = s_page_size;
	tgt_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	tgt_opts.buf_opts.length = opts->buf_size;
	tgt_opts.buf_opts.pattern = CTX_BUF_PAT_ZERO;
	tgt_opts.buf_opts.hp = opts->hugepages;
	tgt_opts.use_gpu_buf = opts->loc_opts.use_rx_gpu;
	tgt_opts.gpu_id = opts->loc_opts.rx_gpu;

	tgt_opts.cq_opts.count = 1;

	/* Allocate */
	rc = alloc_tgt(cxi, &tgt_opts);
	if (rc < 0)
		return rc;

	/* Append Persistent LE */
	flags = C_LE_EVENT_SUCCESS_DISABLE | C_LE_EVENT_COMM_DISABLE |
		C_LE_EVENT_UNLINK_DISABLE | C_LE_UNRESTRICTED_BODY_RO |
		C_LE_UNRESTRICTED_END_RO | C_LE_OP_GET;
	rc = append_le(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0, flags,
		       cxi->tgt_pte->ptn, 0, 0);
	if (rc)
		return rc;

	return 0;
}

/* Send a single read and measure the time until its REPLY arrives */
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

	/* Enqueue TX command and ring doorbell */
	util->dma_cmd.nomatch_dma.request_len = util->size;
	rc = cxi_cq_emit_nomatch_dma(cxi->ini_cq, &util->dma_cmd.nomatch_dma);
	if (rc) {
		fprintf(stderr, "Failed to issue DMA command: %s\n",
			strerror(-rc));
		return rc;
	}
	if (opts->use_ll)
		cxi_cq_ll_ring(cxi->ini_cq);
	else
		cxi_cq_ring(cxi->ini_cq);

	/* Get end time when read REPLY is received */
	rc = get_event(cxi->ini_eq, C_EVENT_REPLY, NULL, &lat_ts_1, NO_TIMEOUT,
		       cxi->dev);
	if (rc) {
		fprintf(stderr, "Failed to get initiator REPLY: %s\n",
			strerror(-rc));
		return rc;
	}

	util->last_lat = TV_NSEC_DIFF(lat_ts_0, lat_ts_1);

	return rc;
}

void usage(void)
{
	printf("Usage:\n");
	printf("  cxi_read_lat [-d DEV] [-p PORT]\n");
	printf("                          Start a server and wait for connection\n");
	printf("  cxi_read_lat ADDR [OPTIONS]\n");
	printf("                          Connect to server with address ADDR\n");
	printf("\n");
	printf("Options:\n");
	printf("  -d, --device=DEV        Device name (default: \"cxi0\")\n");
	printf("  -v, --svc-id=SVC_ID     Service ID (default: 1)\n");
	printf("  -p, --port=PORT         The port to listen on/connect to (default: %u)\n",
	       DFLT_PORT);
	printf("  -t, --tx-gpu=GPU        GPU index for allocating TX buf (default: no GPU)\n");
	printf("  -r, --rx-gpu=GPU        GPU index for allocating RX buf (default: no GPU)\n");
	printf("  -g, --gpu-type=TYPE     GPU type (AMD or NVIDIA or INTEL) (default type determined\n");
	printf("                          by discovered GPU files on system)\n");
	printf("  -n, --iters=ITERS       Number of iterations to run (default: %u)\n",
	       ITERS_DFLT);
	printf("  -D, --duration=SEC      Run for the specified number of seconds\n");
	printf("      --warmup=WARMUP     Number of warmup iterations to run (default: %u)\n",
	       WARMUP_DFLT);
	printf("      --latency-gap=USEC  Number of microseconds to wait between each\n");
	printf("                          iteration (default: %u)\n",
	       DELAY_DFLT);
	printf("  -s, --size=MIN[:MAX]    Read size or range to use (default: %u)\n",
	       SIZE_DFLT);
	printf("                          Ranges must be powers of 2 (e.g. \"1:8192\")\n");
	printf("                          The maximum size is %lu\n",
	       MAX_MSG_SIZE);
	printf("      --unrestricted      Use unrestricted reads\n");
	printf("      --no-ll             Do not use Low-Latency command issuing\n");
	printf("      --report-all        Report all latency measurements individually\n");
	printf("                          This option is ignored when using --duration\n");
	printf("      --use-hp=SIZE       Attempt to use huge pages when allocating\n");
	printf("                          Size may be \"2M\" or \"1G\"\n");
	printf("  -h, --help              Print this help text and exit\n");
	printf("  -V, --version           Print the version and exit\n");
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
	opts->warmup = WARMUP_DFLT;
	opts->iter_delay = DELAY_DFLT;
	opts->use_ll = 1;

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
		{ "unrestricted", no_argument, NULL, VAL_UNRESTRICTED },
		{ "no-ll", no_argument, NULL, VAL_NO_LL },
		{ "report-all", no_argument, NULL, VAL_REPORT_ALL },
		{ "warmup", required_argument, NULL, VAL_WARMUP },
		{ "latency-gap", required_argument, NULL, VAL_ITER_DELAY },
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
		{ "emu-mode", no_argument, NULL, VAL_EMU_MODE },
		{ "help", no_argument, NULL, 'h' },
		{ "version", no_argument, NULL, 'V' },
		{ NULL, 0, NULL, 0 }
	};

	while (1) {
		c = getopt_long(argc, argv, "d:v:p:t:r:g:n:D:s:hV", longopts,
				NULL);
		if (c == -1)
			break;
		parse_common_opt(c, opts, name, version, usage);
	}
	if (opts->duration)
		opts->report_all = 0;
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
	opts->buf_size = opts->max_size;
	num_hp = get_hugepages_needed(opts, !ctrl->is_server, ctrl->is_server);

	/* Print out the configuration */
	snprintf(util.header, MAX_HDR_LEN, "%*s  %*s  %*s  %*s  %*s  %*s",
		 SIZE_W, "RDMA Size[B]", COUNT_W, "Reads", LAT_W, "Min[us]",
		 LAT_W, "Max[us]", LAT_W, "Mean[us]", LAT_W, "StdDev[us]");
	print_separator(strlen(util.header));
	printf("    CXI RDMA Read Latency Test\n");
	print_loc_opts(opts, ctrl->is_server);
	if (opts->duration) {
		printf("Test Type        : Duration\n");
		printf("Duration         : %u seconds\n", opts->duration);
	} else {
		printf("Test Type        : Iteration\n");
		printf("Iterations       : %lu\n", opts->iters);
	}
	printf("Warmup Iters     : %lu\n", opts->warmup);
	printf("Inter-Iter Gap   : %lu microseconds\n", opts->iter_delay);
	if (opts->min_size == opts->max_size) {
		printf("Read Size        : %lu\n", opts->max_size);
	} else {
		printf("Min Read Size    : %lu\n", opts->min_size);
		printf("Max Read Size    : %lu\n", opts->max_size);
	}
	printf("Restricted       : %s\n",
	       opts->unrestricted ? "Disabled" : "Enabled");
	printf("LL Cmd Launch    : %s\n",
	       opts->use_ll ? "Enabled" : "Disabled");
	printf("Results Reported : %s\n", opts->report_all ? "All" : "Summary");
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
	if (!ctrl->is_server) {
		rc = read_lat_alloc_ini(&util);
		if (rc < 0)
			goto cleanup;
	} else {
		rc = read_lat_alloc_tgt(&util);
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

	for (util.size = opts->min_size; util.size <= opts->max_size;
	     util.size *= 2) {
		if (!ctrl->is_server)
			rc = run_lat_active(&util, do_single_iteration);
		else
			rc = run_lat_passive(&util);
		if (rc)
			break;
	}
	print_separator(strlen(util.header));

cleanup:
	ctx_destroy(cxi);
	ctrl_close(ctrl);
	if (opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu)
		gpu_lib_fini(opts->loc_opts.gpu_type);
	return rc ? 1 : 0;
}
