/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI RDMA send latency benchmark */

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
#include "get_clock.h"

// clang-format off
#define ITERS_DFLT  100
#define WARMUP_DFLT 10
#define SIZE_DFLT   8
#define DELAY_DFLT  1000

#define MATCH_BITS       0x123456789ABCDEF0
#define RDZV_IGNORE_BITS 0xF00 /* Ignore rdzv transaction type */
#define SRVR_RDZV_ID     0xAA
#define CLNT_RDZV_ID     0xCC
#define HEADER_DATA      0x77
#define USER_PTR         0x88
// clang-format on

static const char *name = "cxi_send_lat";
static const char *version = "2.3.0";

/* Allocate resources */
int send_lat_alloc(struct util_context *util)
{
	int rc;
	struct cxi_ctx_ini_opts ini_opts = { 0 };
	struct cxi_ctx_tgt_opts tgt_opts = { 0 };
	union c_cmdu c_st_cmd = { 0 };
	struct cxi_context *cxi = &util->cxi;
	struct ctrl_connection *ctrl = &util->ctrl;
	struct util_opts *opts = &util->opts;
	uint64_t rdzv_match_id;
	uint64_t rdzv_match_bits;
	uint64_t dma_initiator;
	uint64_t me_initiator;
	uint32_t flags;

	if (opts->clock == CYCLES) {
		util->clock_ghz =
			get_cpu_mhz(opts->ignore_cpu_freq_mismatch) / 1000;
	}

	/* Config */
	ini_opts.alloc_rdzv = opts->use_rdzv;

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

	tgt_opts.eq_attr.queue_len = s_page_size;
	tgt_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	tgt_opts.buf_opts.length = opts->buf_size;
	tgt_opts.buf_opts.pattern = CTX_BUF_PAT_ZERO;
	tgt_opts.buf_opts.hp = opts->hugepages;
	tgt_opts.use_gpu_buf = opts->loc_opts.use_rx_gpu;
	tgt_opts.gpu_id = opts->loc_opts.rx_gpu;

	tgt_opts.cq_opts.count = 1;

	if (opts->use_rdzv)
		tgt_opts.pt_opts.is_matching = 1;
	tgt_opts.use_final_lat = true;

	/* Allocate */
	rc = alloc_ini(cxi, &ini_opts);
	if (rc < 0)
		return rc;
	rc = alloc_tgt(cxi, &tgt_opts);
	if (rc < 0)
		return rc;

	/* Physical matching (use_logical is not set) */
	dma_initiator = CXI_MATCH_ID(cxi->dev->info.pid_bits, cxi->dom->pid,
				     cxi->dev->info.nid);

	/* Match DMA Put command setup */
	if (!opts->use_idc || opts->max_size > MAX_IDC_UNRESTRICTED_SIZE) {
		util->dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
		if (opts->use_rdzv) {
			util->dma_cmd.full_dma.command.opcode =
				C_CMD_RENDEZVOUS_PUT;
			util->dma_cmd.full_dma.match_bits = MATCH_BITS;
			if (!ctrl->is_server)
				util->dma_cmd.full_dma.rendezvous_id =
					CLNT_RDZV_ID;
			else
				util->dma_cmd.full_dma.rendezvous_id =
					SRVR_RDZV_ID;
			util->dma_cmd.full_dma.initiator = dma_initiator;
			util->dma_cmd.full_dma.header_data = HEADER_DATA;
			util->dma_cmd.full_dma.user_ptr = USER_PTR;
		} else {
			util->dma_cmd.full_dma.command.opcode = C_CMD_PUT;
		}
		util->dma_cmd.full_dma.index_ext = cxi->index_ext;
		util->dma_cmd.full_dma.lac = cxi->ini_buf->md->lac;
		util->dma_cmd.full_dma.event_send_disable = 1;
		util->dma_cmd.full_dma.dfa = cxi->dfa;
		util->dma_cmd.full_dma.local_addr =
			CXI_VA_TO_IOVA(cxi->ini_buf->md, cxi->ini_buf->buf);
		util->dma_cmd.full_dma.eq = cxi->ini_eq->eqn;
	}

	/* IDC command setup */
	if (opts->use_idc && opts->min_size <= MAX_IDC_UNRESTRICTED_SIZE) {
		/* C State */
		c_st_cmd.c_state.command.opcode = C_CMD_CSTATE;
		c_st_cmd.c_state.index_ext = cxi->index_ext;
		c_st_cmd.c_state.event_send_disable = 1;
		c_st_cmd.c_state.event_success_disable = 1;
		c_st_cmd.c_state.eq = cxi->ini_eq->eqn;

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
		flags = C_LE_EVENT_UNLINK_DISABLE | C_LE_OP_GET;
		rc = append_me(cxi->ini_rdzv_pte_cq, cxi->ini_rdzv_eq,
			       cxi->ini_buf, 0, flags, cxi->ini_rdzv_pte->ptn,
			       0, rdzv_match_id, rdzv_match_bits,
			       RDZV_IGNORE_BITS, 0);
		if (rc)
			return rc;
	}

	/* Physical matching (use_logical is not set) */
	me_initiator = CXI_MATCH_ID(cxi->dev->info.pid_bits, cxi->rmt_addr.pid,
				    cxi->rmt_addr.nic);

	/* Append Persistent RX ME */
	flags = C_LE_EVENT_UNLINK_DISABLE | C_LE_OP_PUT;
	if (opts->use_rdzv) {
		rc = append_me(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0,
			       flags, cxi->tgt_pte->ptn, 0, me_initiator,
			       MATCH_BITS, 0, 0);
		rc = append_me(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0,
			       flags, cxi->tgt_final_lat_pte->ptn, 0,
			       me_initiator, MATCH_BITS, 0, 1);
	} else {
		flags |= C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO;
		rc = append_le(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0,
			       flags, cxi->tgt_pte->ptn, 0, 0);
		rc = append_le(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf, 0,
			       flags, cxi->tgt_final_lat_pte->ptn, 0, 1);
	}
	if (rc)
		return rc;

	return 0;
}

/* Send and wait for ACKs */
static int do_single_send(struct util_context *util)
{
	int rc;
	int ev_rc;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	const union c_event *event;
	bool dma_used = !opts->use_idc ||
		util->size > MAX_IDC_UNRESTRICTED_SIZE;

	/* Enqueue TX command and ring doorbell */
	if (dma_used) {
		rc = cxi_cq_emit_dma(cxi->ini_cq, &util->dma_cmd.full_dma);
		if (rc) {
			fprintf(stderr, "Failed to issue DMA command: %s\n",
				strerror(-rc));
			return rc;
		}
	} else {
		rc = cxi_cq_emit_idc_put(cxi->ini_cq, &util->idc_cmd.idc_put,
					 cxi->ini_buf->buf, util->size);
		if (rc) {
			fprintf(stderr, "Failed to issue IDC command: %s\n",
				strerror(-rc));
			return rc;
		}
	}
	if (opts->use_ll)
		cxi_cq_ll_ring(cxi->ini_cq);
	else
		cxi_cq_ring(cxi->ini_cq);

	if (dma_used) {
		/* Wait for ACK */
		do {
			event = cxi_eq_get_event(cxi->ini_eq);
		} while (!event);

		if (event->hdr.event_type != C_EVENT_ACK) {
			fprintf(stderr, "Did not receive C_EVENT_ACK: %s\n",
				cxi_event_type_to_str(event->hdr.event_type));
			return -ENOMSG;
		}

		ev_rc = cxi_event_rc(event);
		if (ev_rc != C_RC_OK) {
			fprintf(stderr, "Event not RC_OK: %s\n",
				cxi_rc_to_str(ev_rc));
			return -ENOMSG;
		}

		if (opts->use_rdzv) {
			/* Wait for GET */
			do {
				event = cxi_eq_get_event(cxi->ini_rdzv_eq);
			} while (!event);

			if (event->hdr.event_type != C_EVENT_GET) {
				fprintf(stderr,
					"Did not receive C_EVENT_GET: %s\n",
					cxi_event_type_to_str(event->hdr.event_type));
				return -ENOMSG;
			}

			ev_rc = cxi_event_rc(event);
			if (ev_rc != C_RC_OK) {
				fprintf(stderr, "Event not RC_OK: %s\n",
					cxi_rc_to_str(ev_rc));
				return -ENOMSG;
			}
		}
	}

	return 0;
}

/* Receive PUT */
static int do_single_recv(struct util_context *util)
{
	int rc;
	int ev_rc;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	int cur_ev_cnt = 0;
	int max_ev_cnt;
	const union c_event *event;

	if (opts->use_rdzv)
		max_ev_cnt = 3;
	else
		max_ev_cnt = 1;

	while (cur_ev_cnt < max_ev_cnt) {
		do {
			event = cxi_eq_get_event(cxi->tgt_eq);
		} while (!event);

		ev_rc = cxi_event_rc(event);
		if (ev_rc != C_RC_OK) {
			fprintf(stderr, "Event not RC_OK: %s\n",
				cxi_rc_to_str(ev_rc));
			return -ENOMSG;
		}

		switch (event->hdr.event_type) {
		case C_EVENT_RENDEZVOUS:
			if (!event->tgt_long.get_issued) {
				/* Manually issue Get */
				rc = sw_rdzv_get(util, event->tgt_long);
				if (rc)
					return rc;
			}
			break;

		case C_EVENT_PUT:
			util->final_lat_recv = event->tgt_long.buffer_id;
		case C_EVENT_REPLY:
			break;

		default:
			fprintf(stderr, "Unexpected event: %s\n",
				cxi_event_type_to_str(event->hdr.event_type));
			return -ENOMSG;
		}

		cur_ev_cnt++;
	}

	return 0;
}

static int do_client(struct util_context *util)
{
	int rc;

	rc = do_single_send(util);
	if (rc)
		return rc;

	return do_single_recv(util);
}

static int do_server(struct util_context *util)
{
	int rc;

	rc = do_single_recv(util);
	if (rc)
		return rc;

	return do_single_send(util);
}

/* Do a single send and measure the time until its PUT response arrives */
int do_single_iteration(struct util_context *util)
{
	int rc;
	int test_rc;
	struct ctrl_connection *ctrl = &util->ctrl;
	struct cxi_context *cxi = &util->cxi;
	struct util_opts *opts = &util->opts;
	struct timespec lat_ts_0 = {};
	cycles_t start = 0;
	struct timespec lat_ts_1 = {};
	cycles_t end = 0;

	util->dma_cmd.full_dma.request_len = util->size;

	rc = ctrl_barrier(ctrl, NO_TIMEOUT, "Sync");
	if (rc)
		return rc;

	if (opts->clock == CYCLES) {
		start = get_cycles();
	} else {
		rc = clock_gettime_or_counters(CLOCK_MONOTONIC_RAW, &lat_ts_0,
					       cxi->dev);
		if (rc < 0) {
			rc = -errno;
			fprintf(stderr, "clock_gettime() failed: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	if (ctrl->is_server)
		test_rc = do_server(util);
	else
		test_rc = do_client(util);

	if (opts->clock == CYCLES) {
		end = get_cycles();
	} else {
		rc = clock_gettime_or_counters(CLOCK_MONOTONIC_RAW, &lat_ts_1,
					       cxi->dev);
		if (rc < 0) {
			rc = -errno;
			fprintf(stderr, "clock_gettime() failed: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	if (test_rc)
		return test_rc;

	/* Only need to ACK EQs on test completion. */
	cxi_eq_ack_events(cxi->tgt_eq);
	cxi_eq_ack_events(cxi->ini_eq);

	if (opts->use_rdzv)
		cxi_eq_ack_events(cxi->ini_rdzv_eq);

	/* Get end time */
	if (!ctrl->is_server) {
		if (opts->clock == CYCLES)
			util->last_lat = (end - start) / util->clock_ghz;
		else
			util->last_lat = TV_NSEC_DIFF(lat_ts_0, lat_ts_1);

		util->last_lat /= 2; /* account for c->s and s->c */
	}

	return 0;
}

void usage(void)
{
	printf("Usage:\n");
	printf("  cxi_send_lat [-d DEV] [-p PORT]\n");
	printf("                          Start a server and wait for connection\n");
	printf("  cxi_send_lat ADDR [OPTIONS]\n");
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
	printf("  -s, --size=MIN[:MAX]    Send size or range to use (default: %u)\n",
	       SIZE_DFLT);
	printf("                          Ranges must be powers of 2 (e.g. \"1:8192\")\n");
	printf("                          The maximum size is %lu\n",
	       MAX_MSG_SIZE);
	printf("      --no-idc            Do not use Immediate Data Cmds for sizes <= %u bytes\n",
	       MAX_IDC_UNRESTRICTED_SIZE);
	printf("      --no-ll             Do not use Low-Latency command issuing\n");
	printf("  -R, --rdzv              Use rendezvous PUTs\n");
	printf("      --report-all        Report all latency measurements individually\n");
	printf("                          This option is ignored when using --duration\n");
	printf("      --use-hp=SIZE       Attempt to use huge pages when allocating\n");
	printf("                          Size may be \"2M\" or \"1G\"\n");
	printf("  -c, --clock=TYPE        Clock type used to calculate latency.\n");
	printf("                          Valid options are: cycles or clock_gettime.\n");
	printf("                          Default is clock_gettime.\n");
	printf("      --ignore-cpu-freq-mismatch\n");
	printf("                          Used when clock type is cycles. Ignore CPU frequency\n");
	printf("                          mismatch. Mismatch can occur when ondemand CPU frequency\n");
	printf("                          is enabled such as cpufreq_ondemand governor.\n");
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
	opts->use_idc = 1;
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
		{ "no-idc", no_argument, NULL, VAL_NO_IDC },
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
		{ "rdzv", no_argument, NULL, 'R' },
		{ "clock", required_argument, NULL, 'c'},
		{ "ignore-cpu-freq-mismatch", no_argument, NULL, VAL_IGNORE_CPU_FREQ_MISMATCH},
		{ "help", no_argument, NULL, 'h' },
		{ "version", no_argument, NULL, 'V' },
		{ NULL, 0, NULL, 0 }
	};

	while (1) {
		c = getopt_long(argc, argv, "c:d:v:p:t:r:g:n:D:s:hVR", longopts,
				NULL);
		if (c == -1)
			break;
		parse_common_opt(c, opts, name, version, usage);
	}
	if (opts->use_rdzv)
		opts->use_idc = 0;
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
	num_hp = get_hugepages_needed(opts, true, true);

	/* Print out the configuration */
	snprintf(util.header, MAX_HDR_LEN, "%*s  %*s  %*s  %*s  %*s  %*s",
		 SIZE_W, "Send Size[B]", COUNT_W, "Sends", LAT_W, "Min[us]",
		 LAT_W, "Max[us]", LAT_W, "Mean[us]", LAT_W, "StdDev[us]");
	print_separator(strlen(util.header));
	printf("    CXI RDMA Send Latency Test\n");
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
		printf("Send Size        : %lu\n", opts->max_size);
	} else {
		printf("Min Send Size    : %lu\n", opts->min_size);
		printf("Max Send Size    : %lu\n", opts->max_size);
	}
	printf("IDC              : %s\n",
	       opts->use_idc ? "Enabled" : "Disabled");
	printf("LL Cmd Launch    : %s\n",
	       opts->use_ll ? "Enabled" : "Disabled");
	printf("Rendezvous PUTs  : %s\n",
	       opts->use_rdzv ? "Enabled" : "Disabled");
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
	rc = send_lat_alloc(&util);
	if (rc < 0)
		goto cleanup;

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
		util.istate = DONE;
		rc = run_lat_active(&util, do_single_iteration);
	}
	print_separator(strlen(util.header));

cleanup:
	ctx_destroy(cxi);
	ctrl_close(ctrl);
	if (opts->loc_opts.use_tx_gpu || opts->loc_opts.use_rx_gpu)
		gpu_lib_fini(opts->loc_opts.gpu_type);
	return rc ? 1 : 0;
}
