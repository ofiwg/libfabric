/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * EFA completion-with-signal integration test.
 *
 * Exercises the EFA "completion with signal" API (see the OFI Extended
 * Completion Signaling design) end to end over an efa-direct RDMA write:
 *
 *   1. Both sides enable signal support on the endpoint before it is enabled
 *      (fi_setopt FI_OPT_EFA_COMP_SIGNAL).
 *   2. A MEMSET completion memory operation is created over a "semaphore"
 *      buffer, then wrapped in a signal (create_comp_mem_op +
 *      register_signal via the FI_EFA_SIGNAL_OPS ops table).
 *   3. The receiver's remote signal_id is exchanged out-of-band to the sender.
 *   4. The initiator issues RDMA writes with FI_EFA_EXTENDED_MSG, attaching a
 *      local and/or remote signal *with data* (the value the device MEMSETs
 *      into the semaphore on completion).
 *   5. Each side verifies its semaphore was set to the expected value.
 *
 * The test focuses on "signal with data" and supports attaching the local
 * signal only, the remote signal only, or both (--signal-mode).
 *
 * MEMSET semantics: the device writes (SET, not accumulate) the per-WR signal
 * data value into the target on each signaled completion, so after a run the
 * semaphore holds the value carried by the last signaled write.
 *
 * This is a single-EP, single-pair test driven over the fabtests OOB socket.
 *
 * Usage:
 *   Server: fi_efa_rma_comp_signal [-e rdm]
 *   Client: fi_efa_rma_comp_signal [-e rdm] <server_addr>
 *
 * Options:
 *   --signal-mode local|remote|both  Which signal(s) to attach (default: both)
 *   --signal-width 8|16|32           MEMSET value width (default: 32)
 *   -I <iters>                       Number of signaled writes (default: 8)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <getopt.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_ext_efa.h>

#include <shared.h>
#include "efa_shared.h"

enum signal_mode {
	SIGNAL_MODE_LOCAL,
	SIGNAL_MODE_REMOTE,
	SIGNAL_MODE_BOTH,
};

enum {
	LONG_OPT_SIGNAL_MODE = 512,
	LONG_OPT_SIGNAL_WIDTH,
};

static enum signal_mode signal_mode = SIGNAL_MODE_BOTH;
static int signal_width = 32;
static int iters = 8;

/* The value the device MEMSETs into the semaphore, per-WR. Chosen to fit any
 * width and to be visibly non-trivial. */
#define SIGNAL_DATA_LOCAL  0x5aU
#define SIGNAL_DATA_REMOTE 0xa5U

/* Local (initiator) signal state: fires on TX completion. */
static struct fid_ep *cs_ep;
static struct fi_efa_ops_signal *sig_ops;

/* Signal handles (fids), and the wire signal ids used in the WR. */
static struct fid_efa_comp_mem_op *local_mem_op;
static struct fid_efa_comp_mem_op *remote_mem_op;
static struct fid_efa_comp_signal *local_signal;
static struct fid_efa_comp_signal *remote_signal;
static uint32_t local_signal_id;
static uint32_t remote_signal_id;

/* Semaphore targets for MEMSET signals (host VA memory). */
static volatile uint32_t *local_sem;
static volatile uint32_t *remote_sem;

static bool want_local(void)
{
	return signal_mode == SIGNAL_MODE_LOCAL ||
	       signal_mode == SIGNAL_MODE_BOTH;
}

static bool want_remote(void)
{
	return signal_mode == SIGNAL_MODE_REMOTE ||
	       signal_mode == SIGNAL_MODE_BOTH;
}

static enum fi_efa_comp_mem_op width_to_op(int width)
{
	switch (width) {
	case 8:
		return FI_EFA_COMP_MEM_OP_SET_SIGNAL_VAL_8;
	case 16:
		return FI_EFA_COMP_MEM_OP_SET_SIGNAL_VAL_16;
	default:
		return FI_EFA_COMP_MEM_OP_SET_SIGNAL_VAL_32;
	}
}

/*
 * Create a MEMSET completion memory op over @sem and wrap it in a signal.
 * On success sets *mem_op / *signal handles and *signal_id (the WR id).
 */
static int register_memset_signal(volatile uint32_t *sem,
				   struct fid_efa_comp_mem_op **mem_op,
				   struct fid_efa_comp_signal **signal,
				   uint32_t *signal_id)
{
	struct fi_efa_comp_mem_op_attr mem_attr = {0};
	struct fi_efa_comp_signal_attr sig_attr = {0};
	int ret;

	mem_attr.op = width_to_op(signal_width);
	mem_attr.flags = FI_EFA_COMP_MEM_OP_WITH_COMP_EXTERNAL_MEM;
	mem_attr.location.type = FI_EFA_MEMORY_LOCATION_VA;
	mem_attr.location.ptr = (uint8_t *) sem;
	mem_attr.length = sizeof(*sem);

	ret = sig_ops->create_comp_mem_op(domain, &mem_attr, mem_op);
	if (ret) {
		FT_PRINTERR("create_comp_mem_op", ret);
		return ret;
	}

	sig_attr.type = FI_EFA_COMP_SIGNAL_MEM_OP;
	sig_attr.mem_op = *mem_op;

	ret = sig_ops->register_signal(domain, &sig_attr, signal);
	if (ret) {
		FT_PRINTERR("register_signal", ret);
		fi_close(&(*mem_op)->fid);
		*mem_op = NULL;
		return ret;
	}

	*signal_id = fid_efa_comp_signal_get_id(*signal);
	return FI_SUCCESS;
}

/*
 * Enable signal support on the endpoint (must be before fi_enable) and open the
 * signal ops table. Called from a custom init that mirrors ft_init_fabric() but
 * defers fi_enable until after fi_setopt.
 */
static int enable_signal_support(void)
{
	bool enable = true;
	int ret;

	ret = fi_open_ops(&domain->fid, FI_EFA_SIGNAL_OPS, 0,
			  (void **) &sig_ops, NULL);
	if (ret) {
		FT_PRINTERR("fi_open_ops(FI_EFA_SIGNAL_OPS)", ret);
		return ret;
	}

	ret = fi_setopt(&cs_ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_COMP_SIGNAL,
			&enable, sizeof(enable));
	if (ret) {
		if (ret == -FI_EOPNOTSUPP) {
			/*
			 * Device/driver does not advertise completion-with-signal
			 * support. Report as ENODATA so fabtests treats it as a
			 * skip rather than a failure.
			 */
			FT_WARN("Completion with signal not supported by device, "
				"skipping test\n");
			return -FI_ENODATA;
		}
		FT_PRINTERR("fi_setopt(FI_OPT_EFA_COMP_SIGNAL)", ret);
		return ret;
	}

	return FI_SUCCESS;
}

/*
 * Custom fabric init: mirrors ft_init_fabric() but inserts fi_setopt for signal
 * support between endpoint creation and enable.
 */
static int init_fabric_with_signal(void)
{
	char buf[FT_MAX_CTRL_MSG];
	int ret;

	ret = ft_init();
	if (ret)
		return ret;

	ret = ft_init_oob();
	if (ret)
		return ret;

	ret = ft_getinfo(hints, &fi);
	if (ret)
		return ret;

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	cs_ep = ep;

	ret = enable_signal_support();
	if (ret)
		return ret;

	ret = ft_enable_ep_recv();
	if (ret)
		return ret;

	ret = ft_init_av();
	if (ret)
		return ret;

	(void) buf;
	return 0;
}

/*
 * Exchange a 32-bit value over the OOB socket. Order is asymmetric so the two
 * sides do not both block on recv.
 */
static int oob_exchange_u32(uint32_t send_val, uint32_t *recv_val)
{
	int ret;

	if (opts.dst_addr) {
		ret = ft_sock_send(oob_sock, &send_val, sizeof(send_val));
		if (ret)
			return ret;
		return ft_sock_recv(oob_sock, recv_val, sizeof(*recv_val));
	}

	ret = ft_sock_recv(oob_sock, recv_val, sizeof(*recv_val));
	if (ret)
		return ret;
	return ft_sock_send(oob_sock, &send_val, sizeof(send_val));
}

/*
 * Post a single RDMA write carrying the requested completion signal(s) with
 * data, using the FI_EFA_EXTENDED_MSG op flag.
 */
static ssize_t post_signaled_write(struct fi_rma_iov *remote, void *context)
{
	struct fi_efa_msg_rma emsg = {0};
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	iov.iov_base = tx_buf;
	iov.iov_len = opts.transfer_size;

	rma_iov.addr = remote->addr;
	rma_iov.len = opts.transfer_size;
	rma_iov.key = remote->key;

	emsg.msg.msg_iov = &iov;
	emsg.msg.desc = &mr_desc;
	emsg.msg.iov_count = 1;
	emsg.msg.addr = remote_fi_addr;
	emsg.msg.rma_iov = &rma_iov;
	emsg.msg.rma_iov_count = 1;
	emsg.msg.context = context;

	if (want_local()) {
		emsg.feature_bits |= FI_EFA_LOCAL_SIGNAL_ID |
				     FI_EFA_LOCAL_SIGNAL_DATA;
		emsg.local_signal_id = local_signal_id;
		emsg.local_signal_data = SIGNAL_DATA_LOCAL;
	}

	if (want_remote()) {
		emsg.feature_bits |= FI_EFA_REMOTE_SIGNAL_ID |
				     FI_EFA_REMOTE_SIGNAL_DATA;
		emsg.remote_signal_id = remote_signal_id;
		emsg.remote_signal_data = SIGNAL_DATA_REMOTE;
	}

	return fi_writemsg(cs_ep, (struct fi_msg_rma *) &emsg,
			   FI_EFA_EXTENDED_MSG);
}

static uint32_t sem_mask(int width)
{
	switch (width) {
	case 8:
		return 0xffU;
	case 16:
		return 0xffffU;
	default:
		return 0xffffffffU;
	}
}

/*
 * Spin on a semaphore until it holds @expected (masked to the signal width),
 * with a timeout. Progresses the endpoint so the provider can drain the CQ.
 */
static int wait_sem(volatile uint32_t *sem, uint32_t expected)
{
	struct timespec a, b;
	uint32_t mask = sem_mask(signal_width);
	int wait_s = (timeout >= 0) ? timeout : 10;

	clock_gettime(CLOCK_MONOTONIC, &a);
	while ((*sem & mask) != (expected & mask)) {
		ft_force_progress();
		clock_gettime(CLOCK_MONOTONIC, &b);
		if ((b.tv_sec - a.tv_sec) > wait_s) {
			FT_ERR("semaphore timeout: got 0x%x, expected 0x%x",
			       *sem & mask, expected & mask);
			return -FI_ETIMEDOUT;
		}
	}
	return 0;
}

static int run_initiator(struct fi_rma_iov *remote)
{
	int i, ret;

	for (i = 0; i < iters; i++) {
		do {
			ret = post_signaled_write(remote, &tx_ctx);
			if (ret == -FI_EAGAIN)
				(void) fi_cq_read(txcq, NULL, 0);
		} while (ret == -FI_EAGAIN);
		if (ret) {
			FT_PRINTERR("fi_writemsg", ret);
			return ret;
		}
		tx_seq++;

		ret = ft_get_tx_comp(tx_seq);
		if (ret)
			return ret;
	}

	/* Local signal fires on local TX completion: verify our semaphore. */
	if (want_local()) {
		ret = wait_sem(local_sem, SIGNAL_DATA_LOCAL);
		if (ret)
			return ret;
		printf("Local signal verified: semaphore = 0x%x\n",
		       *local_sem & sem_mask(signal_width));
	}

	return 0;
}

static int run_target(void)
{
	int ret;

	/* Remote signal fires on the target when the write lands. */
	if (want_remote()) {
		ret = wait_sem(remote_sem, SIGNAL_DATA_REMOTE);
		if (ret)
			return ret;
		printf("Remote signal verified: semaphore = 0x%x\n",
		       *remote_sem & sem_mask(signal_width));
	}

	return 0;
}

static int run(void)
{
	struct fi_rma_iov remote = {0};
	uint32_t peer_signal_id = 0;
	int ret;

	ret = init_fabric_with_signal();
	if (ret)
		return ret;

	/*
	 * Register signals. Both sides register a MEMSET signal:
	 *   - initiator registers the LOCAL signal (fires on its TX completion)
	 *   - target registers the REMOTE signal (fires when the write lands)
	 * The target's signal_id is sent to the initiator out-of-band.
	 */
	if (opts.dst_addr) {
		if (want_local()) {
			local_sem = calloc(1, sizeof(*local_sem));
			if (!local_sem)
				return -FI_ENOMEM;
			ret = register_memset_signal(local_sem,
						     &local_mem_op,
						     &local_signal,
						     &local_signal_id);
			if (ret)
				return ret;
		}
	} else {
		if (want_remote()) {
			remote_sem = calloc(1, sizeof(*remote_sem));
			if (!remote_sem)
				return -FI_ENOMEM;
			ret = register_memset_signal(remote_sem,
						     &remote_mem_op,
						     &remote_signal,
						     &remote_signal_id);
			if (ret)
				return ret;
		}
	}

	/* Exchange the target's remote signal_id to the initiator. */
	ret = oob_exchange_u32(opts.dst_addr ? 0 : remote_signal_id,
			       &peer_signal_id);
	if (ret)
		return ret;
	if (opts.dst_addr && want_remote())
		remote_signal_id = peer_signal_id;

	/* Exchange RMA keys so the initiator can write into the target buffer. */
	ret = ft_exchange_keys(&remote);
	if (ret)
		return ret;

	if (opts.dst_addr)
		ret = run_initiator(&remote);
	else
		ret = run_target();
	if (ret)
		goto out;

	ret = ft_finalize();
out:
	/* Signals reference their mem-op, so close signals first. */
	if (local_signal)
		fi_close(&local_signal->fid);
	if (local_mem_op)
		fi_close(&local_mem_op->fid);
	if (remote_signal)
		fi_close(&remote_signal->fid);
	if (remote_mem_op)
		fi_close(&remote_mem_op->fid);
	free((void *) local_sem);
	free((void *) remote_sem);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;
	int lopt_idx = 0;
	struct option long_opts[] = {
		{"signal-mode", required_argument, NULL, LONG_OPT_SIGNAL_MODE},
		{"signal-width", required_argument, NULL, LONG_OPT_SIGNAL_WIDTH},
		{0, 0, 0, 0}
	};

	opts = INIT_OPTS;
	opts.rma_op = FT_RMA_WRITE;
	opts.transfer_size = 64;
	opts.comp_method = FT_COMP_SPIN;
	opts.iterations = 8;
	/* OOB sync so our own signal-id exchange does not race the tx/rx seqs. */
	opts.options |= FT_OPT_OOB_SYNC;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "h" CS_OPTS INFO_OPTS API_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		case LONG_OPT_SIGNAL_MODE:
			if (!strcasecmp(optarg, "local"))
				signal_mode = SIGNAL_MODE_LOCAL;
			else if (!strcasecmp(optarg, "remote"))
				signal_mode = SIGNAL_MODE_REMOTE;
			else if (!strcasecmp(optarg, "both"))
				signal_mode = SIGNAL_MODE_BOTH;
			else {
				fprintf(stderr,
					"signal-mode must be local|remote|both\n");
				return EXIT_FAILURE;
			}
			break;
		case LONG_OPT_SIGNAL_WIDTH:
			signal_width = atoi(optarg);
			if (signal_width != 8 && signal_width != 16 &&
			    signal_width != 32) {
				fprintf(stderr,
					"signal-width must be 8|16|32\n");
				return EXIT_FAILURE;
			}
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "EFA completion-with-signal integration test.");
			FT_PRINT_OPTS_USAGE("--signal-mode <m>",
				"which signal(s) to attach: local|remote|both "
				"(default: both)");
			FT_PRINT_OPTS_USAGE("--signal-width <w>",
				"MEMSET value width in bits: 8|16|32 "
				"(default: 32)");
			FT_PRINT_OPTS_USAGE("-I <iters>",
				"number of signaled writes (default: 8)");
			return EXIT_FAILURE;
		default:
			ft_parse_api_opts(op, optarg, hints, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	iters = opts.iterations;

	/* Completion with signal is an efa-direct feature. */
	if (!hints->fabric_attr->name)
		hints->fabric_attr->name = strdup(EFA_DIRECT_FABRIC_NAME);
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_RMA;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	hints->addr_format = opts.address_format;
	hints->mode |= FI_CONTEXT2;

	printf("Signal mode: %s\n",
	       signal_mode == SIGNAL_MODE_LOCAL ? "LOCAL" :
	       signal_mode == SIGNAL_MODE_REMOTE ? "REMOTE" : "BOTH");
	printf("Signal width: %d bits\n", signal_width);
	printf("Iterations: %d\n", iters);

	ret = run();

	ft_free_res();
	return ft_exit_code(ret);
}
