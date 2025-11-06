/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI / GPU loopback bandwidth benchmark */

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

#define ITERS_DFLT 25
#define SIZE_DFLT (1024 * 512)
#define LIST_SIZE_DFLT 8192

static const char *name = "cxi_gpu_loopback_bw";
static const char *version = "1.5.1";

/* Allocate TX resources */
int bw_alloc_tx(struct util_context *util)
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
	ini_opts.buf_opts.pattern = CTX_BUF_PAT_NONE;
	ini_opts.buf_opts.hp = opts->hugepages;
	ini_opts.use_gpu_buf = opts->loc_opts.use_tx_gpu;
	ini_opts.gpu_id = opts->loc_opts.tx_gpu;

	/* count granularity is 64B, and IDC commands can be up to 256B each
	 * with a preceding 64B CT State command. So worst case we need 4x + 1
	 */
	ini_opts.cq_opts.count = (opts->list_size * 4) + 1;
	ini_opts.cq_opts.flags = CXI_CQ_IS_TX;

	/* Allocate */
	cxi->rmt_addr.nic = cxi->loc_addr.nic;
	cxi->rmt_addr.pid = cxi->loc_addr.pid;
	rc = alloc_ini(cxi, &ini_opts);
	if (rc < 0)
		return rc;

	/* Nomatch DMA Put command setup */
	util->dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
	util->dma_cmd.nomatch_dma.command.opcode = C_CMD_NOMATCH_PUT;
	util->dma_cmd.nomatch_dma.index_ext = cxi->index_ext;
	util->dma_cmd.nomatch_dma.lac = cxi->ini_buf->md->lac;
	util->dma_cmd.nomatch_dma.event_send_disable = 1;
	util->dma_cmd.nomatch_dma.restricted = 1;
	util->dma_cmd.nomatch_dma.dfa = cxi->dfa;
	util->dma_cmd.nomatch_dma.local_addr =
		CXI_VA_TO_IOVA(cxi->ini_buf->md, cxi->ini_buf->buf);
	util->dma_cmd.nomatch_dma.eq = cxi->ini_eq->eqn;
	util->dma_cmd.nomatch_dma.event_success_disable = 1;
	util->dma_cmd.nomatch_dma.event_ct_ack = 1;
	util->dma_cmd.nomatch_dma.ct = cxi->ini_ct->ctn;

	/* CT Event command setup */
	util->ct_cmd.ct.eq = cxi->ini_eq->eqn;
	util->ct_cmd.ct.trig_ct = cxi->ini_ct->ctn;

	return 0;
}

/* Allocate RX resources */
int bw_alloc_rx(struct util_context *util)
{
	int rc;
	struct cxi_ctx_tgt_opts tgt_opts = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	uint32_t flags;
	uint64_t ignore_bits;

	/* Config */
	tgt_opts.eq_attr.queue_len = s_page_size;
	tgt_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	tgt_opts.buf_opts.length = opts->buf_size;
	tgt_opts.buf_opts.pattern = CTX_BUF_PAT_NONE;
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
		C_LE_EVENT_UNLINK_DISABLE | C_LE_OP_PUT;
	/* enables RO=1 w/ restricted packets */
	ignore_bits = 1;
	rc = append_me(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0, flags,
		       cxi->tgt_pte->ptn, 0, 0, 0, ignore_bits, 0);
	if (rc)
		return rc;

	return 0;
}

/* Send cmds_per_iter writes and wait for their ACKs */
int do_single_iteration(struct util_context *util)
{
	int rc = 0;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	int i;

	rc = inc_ct(cxi->ini_trig_cq, &util->ct_cmd.ct, opts->list_size);
	if (rc)
		return rc;

	/* Enqueue TX command and ring doorbell */
	for (i = 0; i < opts->list_size; i++) {
		util->dma_cmd.nomatch_dma.request_len = util->size;
		util->dma_cmd.nomatch_dma.remote_offset = 0;
		rc = cxi_cq_emit_nomatch_dma(cxi->ini_cq,
					     &util->dma_cmd.nomatch_dma);
		if (rc) {
			fprintf(stderr, "Failed to issue DMA command: %s\n",
				strerror(-rc));
			return rc;
		}
	}
	cxi_cq_ring(cxi->ini_cq);

	/* Wait for ACK Event(s) */
	rc = wait_for_ct(cxi->ini_eq, NO_TIMEOUT, "initiator ACK");
	if (rc)
		return rc;

	return rc;
}

void usage(void)
{
	printf("Usage:\n");
	printf("  cxi_loopback_bw [OPTIONS]\n");
	printf("\n");
	printf("Options:\n");
	printf("  -d, --device=DEV       Cassini device (default: \"cxi0\")\n");
	printf("  -v, --svc-id=SVC_ID    Service ID (default: 1)\n");
	printf("  -t, --tx-gpu=GPU       GPU index for allocating TX buf (default: no GPU)\n");
	printf("  -r, --rx-gpu=GPU       GPU index for allocating RX buf (default: no GPU)\n");
	printf("  -g, --gpu-type=TYPE    GPU type (AMD or NVIDIA or INTEL). Default type determined by\n");
	printf("                         discovered GPU files on system.\n");
	printf("  -D, --duration=SEC     Run for the specified number of seconds\n");
	printf("  -i, --iters=ITERS      Number of iterations to run if duration not\n");
	printf("                         specified (default: %u)\n", ITERS_DFLT);
	printf("  -n, --num-xfers=XFERS  Number of transfers per iteration, all pushed to\n");
	printf("                         the Tx CQ prior to initiating xfer (default: %u)\n",
	       LIST_SIZE_DFLT);
	printf("  -s, --size=SIZE        Transfer Size in Bytes (default: %u)\n",
	       SIZE_DFLT);
	printf("                         The maximum size is %lu bytes\n",
	       MAX_MSG_SIZE);
	printf("  --use-hp=SIZE          Attempt to use huge pages when allocating\n");
	printf("                         Size may be \"2M\" or \"1G\"\n");
	printf("                         system memory\n");
	printf("  -h, --help             Print this help text and exit\n");
	printf("  -V, --version          Print the version and exit\n");
}

int main(int argc, char **argv)
{
	int c;
	int rc;
	char *endptr;
	struct util_context util = { 0 };
	struct cxi_context *cxi = &util.cxi;
	struct util_opts *opts = &util.opts;
	int count;
	int num_hp;

	opts->loc_opts.svc_id = CXI_DEFAULT_SVC_ID;
	opts->iters = ITERS_DFLT;
	opts->max_size = SIZE_DFLT;
	opts->list_size = LIST_SIZE_DFLT;
	opts->print_gbits = 1;

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
		{ "use-hp", required_argument, NULL, VAL_USE_HP },
		{ "device", required_argument, NULL, 'd' },
		{ "svc-id", required_argument, NULL, 'v' },
		{ "tx-gpu", required_argument, NULL, 't' },
		{ "rx-gpu", required_argument, NULL, 'r' },
		{ "gpu-type", required_argument, NULL, 'g' },
		{ "iters", required_argument, NULL, 'i' },
		{ "duration", required_argument, NULL, 'D' },
		{ "size", required_argument, NULL, 's' },
		{ "num-xfers", required_argument, NULL, 'n' },
		{ "help", no_argument, NULL, 'h' },
		{ "version", no_argument, NULL, 'V' },
		{ NULL, 0, NULL, 0 }
	};

	while (1) {
		c = getopt_long(argc, argv, "d:v:t:r:g:i:D:s:n:hV", longopts,
				NULL);
		if (c == -1)
			break;

		switch (c) {
		case 'i':
			/* Other utils use n for this */
			parse_common_opt('n', opts, name, version, usage);
			break;
		case 's':
			/* Not a range like other utils */
			errno = 0;
			endptr = NULL;
			opts->max_size = strtoul(optarg, &endptr, 0);
			if (errno != 0 || *endptr != 0 ||
			    opts->max_size > MAX_MSG_SIZE)
				errx(1, "Invalid transfer size: %s", optarg);
			opts->min_size = opts->max_size;
			break;
		case 'n':
			/* Other utils use l for this */
			parse_common_opt('l', opts, name, version, usage);
			break;
		default:
			parse_common_opt(c, opts, name, version, usage);
			break;
		}
	}
	if ((opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu) &&
	    opts->loc_opts.gpu_type < 0)
		errx(1, "Invalid GPU type or unable to find GPU libraries");

	/* Determine hugepages needed and check if available */
	opts->buf_size = opts->max_size;
	num_hp = get_hugepages_needed(opts, true, true);

	/* Print out the configuration */
	snprintf(util.header, MAX_HDR_LEN, "%*s  %*s  %*s  %*s", SIZE_W,
		 "RDMA Size[B]", COUNT_W, "Writes", BW_W, "BW[Gb/s]", MRATE_W,
		 "PktRate[Mpkt/s]");
	print_separator(strlen(util.header));
	printf("    CXI Loopback Bandwidth Test\n");
	printf("Device           : cxi%u\n", opts->loc_opts.dev_id);
	printf("Service ID       : %u\n", opts->loc_opts.svc_id);
	if (opts->loc_opts.use_tx_gpu)
		printf("TX Mem           : GPU %d\n",
		       opts->loc_opts.tx_gpu);
	else
		printf("TX Mem           : System\n");
	if (opts->loc_opts.use_rx_gpu)
		printf("RX Mem           : GPU %d\n",
		       opts->loc_opts.rx_gpu);
	else
		printf("RX Mem           : System\n");
	if (opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu)
		printf("GPU Type         : %s\n",
		       gpu_names[opts->loc_opts.gpu_type]);
	if (opts->duration) {
		printf("Test Type        : Duration\n");
		printf("Duration         : %u seconds\n", opts->duration);
	} else {
		printf("Test Type        : Iteration\n");
		printf("Iterations       : %lu\n", opts->iters);
	}
	printf("Write Size (B)   : %lu\n", opts->max_size);
	printf("Cmds per iter    : %u\n", opts->list_size);
	print_hugepage_opts(opts, num_hp);

	/* Allocate CXI context */
	rc = ctx_alloc(cxi, opts->loc_opts.dev_id, opts->loc_opts.svc_id);
	if (rc < 0)
		goto cleanup;

	/* Initialize GPU library if we are using GPU memory */
	if (opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu) {
		rc = gpu_lib_init(opts->loc_opts.gpu_type);
		if (rc < 0)
			goto cleanup;
		count = get_gpu_device_count();
		printf("Found %d GPU(s)\n", count);
	}

	/* Allocate resources */
	rc = bw_alloc_tx(&util);
	if (rc < 0)
		goto cleanup;
	rc = bw_alloc_rx(&util);
	if (rc < 0)
		goto cleanup;

	print_separator(strlen(util.header));
	printf("%s\n", util.header);

	/* Run test */
	util.size = opts->max_size;
	rc = run_bw_active(&util, do_single_iteration);

	print_separator(strlen(util.header));

cleanup:
	/* Clean up CXI resources */
	ctx_destroy(cxi);

	if (opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu)
		gpu_lib_fini(opts->loc_opts.gpu_type);
	return rc ? 1 : 0;
}
