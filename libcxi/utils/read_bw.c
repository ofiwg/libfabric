/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI RDMA read bandwidth benchmark */

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

#define MAX_BUF_SIZE_DFLT (1UL << 32) /* 4GiB */
#define BUF_ALIGN_DFLT 64
#define ITERS_DFLT 1000
#define SIZE_DFLT 65536
#define LIST_SIZE_DFLT 256

static const char *name = "cxi_read_bw";
static const char *version = "2.4.0";

/* Allocate RDMA Initiator resources */
int read_bw_alloc_ini(struct util_context *util)
{
	int rc;
	struct cxi_ctx_ini_opts ini_opts = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;

	/* Config */
	ini_opts.alloc_ct = true;

	ini_opts.eq_attr.queue_len = (opts->list_size + 1) * 64;
	ini_opts.eq_attr.queue_len =
		NEXT_MULTIPLE(ini_opts.eq_attr.queue_len, s_page_size);
	ini_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	ini_opts.buf_opts.length = opts->buf_size;
	ini_opts.buf_opts.pattern = CTX_BUF_PAT_A5;
	ini_opts.buf_opts.hp = opts->hugepages;
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

	/* Nomatch DMA Get command setup */
	util->dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
	util->dma_cmd.nomatch_dma.command.opcode = C_CMD_NOMATCH_GET;
	util->dma_cmd.nomatch_dma.index_ext = cxi->index_ext;
	util->dma_cmd.nomatch_dma.lac = cxi->ini_buf->md->lac;
	util->dma_cmd.nomatch_dma.event_send_disable = 1;
	util->dma_cmd.nomatch_dma.event_success_disable = 1;
	util->dma_cmd.nomatch_dma.event_ct_reply = 1;
	if (!opts->unrestricted)
		util->dma_cmd.nomatch_dma.restricted = 1;
	util->dma_cmd.nomatch_dma.dfa = cxi->dfa;
	util->dma_cmd.nomatch_dma.eq = cxi->ini_eq->eqn;
	util->dma_cmd.nomatch_dma.ct = cxi->ini_ct->ctn;

	/* IDC command setup - invalid for Gets */

	/* CT Event command setup */
	util->ct_cmd.ct.eq = cxi->ini_eq->eqn;
	util->ct_cmd.ct.trig_ct = cxi->ini_ct->ctn;

	return 0;
}

/* Allocate RDMA Target resources */
int read_bw_alloc_tgt(struct util_context *util)
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

/* Send list_size reads and wait for their REPLY events */
int do_single_iteration(struct util_context *util)
{
	int rc = 0;
	int i;
	uint64_t rmt_offset;
	uint64_t local_addr;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;

	rc = inc_ct(cxi->ini_trig_cq, &util->ct_cmd.ct, opts->list_size);
	if (rc)
		return rc;

	rmt_offset = 0;
	local_addr = (uintptr_t)cxi->ini_buf->buf;

	/* Enqueue TX command and ring doorbell */
	for (i = 0; i < opts->list_size; i++) {
		util->dma_cmd.nomatch_dma.request_len = util->size;
		util->dma_cmd.nomatch_dma.remote_offset = rmt_offset;
		util->dma_cmd.nomatch_dma.local_addr =
			CXI_VA_TO_IOVA(cxi->ini_buf->md, local_addr);
		rc = cxi_cq_emit_nomatch_dma(cxi->ini_cq,
					     &util->dma_cmd.nomatch_dma);
		if (rc) {
			fprintf(stderr, "Failed to issue DMA command: %s\n",
				strerror(-rc));
			return rc;
		}

		inc_tx_buf_offsets(util, &rmt_offset, &local_addr);
	}
	cxi_cq_ring(cxi->ini_cq);

	/* Wait for REPLY Event(s) */
	rc = wait_for_ct(cxi->ini_eq, NO_TIMEOUT, "initiator ACK");
	if (rc)
		return rc;

	return rc;
}

void usage(void)
{
	printf("Usage:\n");
	printf("  cxi_read_bw [-d DEV] [-p PORT]\n");
	printf("                         Start a server and wait for connection\n");
	printf("  cxi_read_bw ADDR [OPTIONS]\n");
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
	printf("  -s, --size=MIN[:MAX]   Read size or range to use (default: %u)\n",
	       SIZE_DFLT);
	printf("                         Ranges must be powers of 2 (e.g. \"1:8192\")\n");
	printf("                         The maximum size is %lu\n",
	       MAX_MSG_SIZE);
	printf("  -l, --list-size=SIZE   Number of reads per iteration, all pushed to\n");
	printf("                         the Tx CQ prior to initiating xfer (default: %u)\n",
	       LIST_SIZE_DFLT);
	printf("  -b, --bidirectional    Measure bidirectional bandwidth\n");
	printf("      --unrestricted     Use unrestricted reads\n");
	printf("      --buf-sz=SIZE      The max TX/RDMA buffer size, specified in bytes\n");
	printf("                         If (size * list_size) > buf_sz, reads will overlap\n");
	printf("                         (default: %lu)\n", MAX_BUF_SIZE_DFLT);
	printf("      --buf_align=ALIGN  Byte-alignment of reads in the buffer (default: %u)\n",
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
	bool using_tx;
	bool using_rx;

	opts->loc_opts.svc_id = CXI_DEFAULT_SVC_ID;
	opts->loc_opts.port = DFLT_PORT;
	opts->iters = ITERS_DFLT;
	opts->min_size = SIZE_DFLT;
	opts->max_size = SIZE_DFLT;
	opts->list_size = LIST_SIZE_DFLT;
	opts->max_buf_size = MAX_BUF_SIZE_DFLT;
	opts->buf_align = BUF_ALIGN_DFLT;

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
		{ "help", no_argument, NULL, 'h' },
		{ "version", no_argument, NULL, 'V' },
		{ NULL, 0, NULL, 0 }
	};

	while (1) {
		c = getopt_long(argc, argv, "d:v:p:t:r:g:n:D:s:l:bhV", longopts,
				NULL);
		if (c == -1)
			break;
		parse_common_opt(c, opts, name, version, usage);
	}
	if (opts->max_size > opts->max_buf_size)
		errx(1, "Max RDMA buffer size (%lu) < max read size (%lu)!",
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
	using_rx = ctrl->is_server || opts->bidirectional;
	using_tx = !using_rx || opts->bidirectional;

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
		 "RDMA Size[B]", COUNT_W, "Reads", BW_W, "BW[MB/s]", MRATE_W,
		 "PktRate[Mpkt/s]");
	print_separator(strlen(util.header));
	printf("    CXI RDMA Read Bandwidth Test\n");
	print_loc_opts(opts, ctrl->is_server);
	if (opts->duration) {
		printf("Test Type        : Duration\n");
		printf("Duration         : %u seconds\n", opts->duration);
	} else {
		printf("Test Type        : Iteration\n");
		printf("Iterations       : %lu\n", opts->iters);
	}
	if (opts->min_size == opts->max_size) {
		printf("Read Size        : %lu\n", opts->max_size);
	} else {
		printf("Min Read Size    : %lu\n", opts->min_size);
		printf("Max Read Size    : %lu\n", opts->max_size);
	}
	printf("List Size        : %u\n", opts->list_size);
	printf("Restricted       : %s\n",
	       opts->unrestricted ? "Disabled" : "Enabled");
	printf("Bidirectional    : %s\n",
	       opts->bidirectional ? "Enabled" : "Disabled");
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
	if (using_tx) {
		rc = read_bw_alloc_ini(&util);
		if (rc < 0)
			goto cleanup;
	}
	if (using_rx) {
		rc = read_bw_alloc_tgt(&util);
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

	for (util.size = opts->min_size; util.size <= opts->max_size;
	     util.size *= 2) {
		util.buf_granularity =
			NEXT_MULTIPLE(util.size, opts->buf_align);
		if (using_tx)
			rc = run_bw_active(&util, do_single_iteration);
		else
			rc = run_bw_passive(&util);
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
