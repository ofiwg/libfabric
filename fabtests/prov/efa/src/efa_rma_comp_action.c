/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * EFA completion-action integration test.
 *
 * Exercises the EFA "completion action" API end to end over an efa-direct RDMA
 * write:
 *
 *   1. Both sides enable action support on the endpoint before it is enabled
 *      (fi_setopt FI_OPT_EFA_COMP_ACTION).
 *   2. A memory completion action is created over a one-entry "semaphore"
 *      vector (create_mem_comp_action via the FI_EFA_MEM_COMP_ACTION_OPS ops
 *      table).
 *   3. The receiver's remote action_id is exchanged out-of-band to the sender.
 *   4. The initiator issues RDMA writes with FI_EFA_EXTENDED_MSG, attaching a
 *      local and/or remote action *with a value* (the value the device writes
 *      into the semaphore on completion).
 *   5. Each side verifies its semaphore was set to the expected value.
 *
 * The test focuses on "action with data" and supports attaching the local
 * action only, the remote action only, or both (--action-mode).
 *
 * FI_EFA_MEM_COMP_ACTION_SET_INITIATOR_VAL semantics: the device writes (sets,
 * does not accumulate) the per-WR value into the entry the WR names, on each
 * completion that carries the action, so after a run the semaphore holds the
 * value carried by the last such write.
 *
 * This is a single-EP, single-pair test driven over the fabtests OOB socket.
 *
 * Usage:
 *   Server: fi_efa_rma_comp_action [-e rdm]
 *   Client: fi_efa_rma_comp_action [-e rdm] <server_addr>
 *
 * Options:
 *   --action-mode local|remote|both  Which action(s) to attach (default: both)
 *   --action-width 8|16|32           Entry width in bits (default: 32)
 *   -I <iters>                       Number of writes that carry an action (default: 8)
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

enum action_mode {
	ACTION_MODE_LOCAL,
	ACTION_MODE_REMOTE,
	ACTION_MODE_BOTH,
};

enum {
	LONG_OPT_ACTION_MODE = 512,
	LONG_OPT_ACTION_WIDTH,
};

static enum action_mode action_mode = ACTION_MODE_BOTH;
static int action_width = 32;
static int iters = 8;

/* The value the device writes into the semaphore, per-WR. Chosen to fit any
 * entry width and to be visibly non-trivial. */
#define ACTION_VALUE_LOCAL  0x5aU
#define ACTION_VALUE_REMOTE 0xa5U

/* Local (initiator) action state: fires on TX completion. */
static struct fid_ep *cs_ep;
static struct fi_efa_ops_mem_comp_action *action_ops;

/* Action handles (fids), and the wire action ids used in the WR. */
static struct fid_efa_comp_action *local_action;
static struct fid_efa_comp_action *remote_action;
static uint32_t local_action_id;
static uint32_t remote_action_id;

/* Single-entry target vectors for the memory actions (host VA memory). */
static volatile uint32_t *local_sem;
static volatile uint32_t *remote_sem;

static bool want_local(void)
{
	return action_mode == ACTION_MODE_LOCAL ||
	       action_mode == ACTION_MODE_BOTH;
}

static bool want_remote(void)
{
	return action_mode == ACTION_MODE_REMOTE ||
	       action_mode == ACTION_MODE_BOTH;
}

/*
 * Create a memory completion action over @sem, a one-entry vector of
 * action_width bits. On success sets *action and *action_id (the id a WR names
 * the action by).
 */
static int register_mem_action(volatile uint32_t *sem,
			       struct fid_efa_comp_action **action,
			       uint32_t *action_id)
{
	struct fi_efa_mem_comp_action_attr attr = {0};
	int ret;

	attr.op = FI_EFA_MEM_COMP_ACTION_SET_INITIATOR_VAL;
	attr.location.type = FI_EFA_MEMORY_LOCATION_VA;
	attr.location.ptr = (uint8_t *) sem;
	attr.num_entries = 1;
	attr.entry_size = action_width / 8;

	ret = action_ops->create_mem_comp_action(domain, &attr, action);
	if (ret) {
		FT_PRINTERR("create_mem_comp_action", ret);
		return ret;
	}

	*action_id = fid_efa_comp_action_get_id(*action);
	return FI_SUCCESS;
}

/*
 * Enable action support on the endpoint (must be before fi_enable) and open the
 * action ops table. Called from a custom init that mirrors ft_init_fabric() but
 * defers fi_enable until after fi_setopt.
 */
static int enable_action_support(void)
{
	bool enable = true;
	int ret;

	ret = fi_open_ops(&domain->fid, FI_EFA_MEM_COMP_ACTION_OPS, 0,
			  (void **) &action_ops, NULL);
	if (ret) {
		FT_PRINTERR("fi_open_ops(FI_EFA_MEM_COMP_ACTION_OPS)", ret);
		return ret;
	}

	ret = fi_setopt(&cs_ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_COMP_ACTION,
			&enable, sizeof(enable));
	if (ret) {
		if (ret == -FI_EOPNOTSUPP) {
			/*
			 * Device/driver does not advertise completion-action
			 * support. Report as ENODATA so fabtests treats it as a
			 * skip rather than a failure.
			 */
			FT_WARN("Completion action not supported by device, "
				"skipping test\n");
			return -FI_ENODATA;
		}
		FT_PRINTERR("fi_setopt(FI_OPT_EFA_COMP_ACTION)", ret);
		return ret;
	}

	return FI_SUCCESS;
}

/*
 * Custom fabric init: mirrors ft_init_fabric() but inserts fi_setopt for action
 * support between endpoint creation and enable.
 */
static int init_fabric_with_action(void)
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

	ret = enable_action_support();
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
 * Post a single RDMA write carrying the requested completion action(s) with
 * data, using the FI_EFA_EXTENDED_MSG op flag.
 */
static ssize_t post_action_write(struct fi_rma_iov *remote, void *context)
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

	/*
	 * The action's vector holds a single entry, so entry 0 is the only one a
	 * WR can name and the INDEX feature bits are left unset (an unset index
	 * is 0).
	 */
	if (want_local()) {
		emsg.feature_bits |= FI_EFA_LOCAL_ACTION_ID |
				     FI_EFA_LOCAL_ACTION_VALUE;
		emsg.local.id = local_action_id;
		emsg.local.value = ACTION_VALUE_LOCAL;
	}

	if (want_remote()) {
		emsg.feature_bits |= FI_EFA_REMOTE_ACTION_ID |
				     FI_EFA_REMOTE_ACTION_VALUE;
		emsg.remote.id = remote_action_id;
		emsg.remote.value = ACTION_VALUE_REMOTE;
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
 * Spin on a semaphore until it holds @expected (masked to the action width),
 * with a timeout. Progresses the endpoint so the provider can drain the CQ.
 */
static int wait_sem(volatile uint32_t *sem, uint32_t expected)
{
	struct timespec a, b;
	uint32_t mask = sem_mask(action_width);
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
			ret = post_action_write(remote, &tx_ctx);
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

	/* Local action fires on local TX completion: verify our semaphore. */
	if (want_local()) {
		ret = wait_sem(local_sem, ACTION_VALUE_LOCAL);
		if (ret)
			return ret;
		printf("Local action verified: semaphore = 0x%x\n",
		       *local_sem & sem_mask(action_width));
	}

	return 0;
}

static int run_target(void)
{
	int ret;

	/* Remote action fires on the target when the write lands. */
	if (want_remote()) {
		ret = wait_sem(remote_sem, ACTION_VALUE_REMOTE);
		if (ret)
			return ret;
		printf("Remote action verified: semaphore = 0x%x\n",
		       *remote_sem & sem_mask(action_width));
	}

	return 0;
}

static int run(void)
{
	struct fi_rma_iov remote = {0};
	uint32_t peer_action_id = 0;
	int ret;

	ret = init_fabric_with_action();
	if (ret)
		return ret;

	/*
	 * Register actions. Both sides register a memory action:
	 *   - initiator registers the LOCAL action (fires on its TX completion)
	 *   - target registers the REMOTE action (fires when the write lands)
	 * The target's action_id is sent to the initiator out-of-band.
	 */
	if (opts.dst_addr) {
		if (want_local()) {
			local_sem = calloc(1, sizeof(*local_sem));
			if (!local_sem)
				return -FI_ENOMEM;
			ret = register_mem_action(local_sem, &local_action,
						  &local_action_id);
			if (ret)
				return ret;
		}
	} else {
		if (want_remote()) {
			remote_sem = calloc(1, sizeof(*remote_sem));
			if (!remote_sem)
				return -FI_ENOMEM;
			ret = register_mem_action(remote_sem, &remote_action,
						  &remote_action_id);
			if (ret)
				return ret;
		}
	}

	/* Exchange the target's remote action_id to the initiator. */
	ret = oob_exchange_u32(opts.dst_addr ? 0 : remote_action_id,
			       &peer_action_id);
	if (ret)
		return ret;
	if (opts.dst_addr && want_remote())
		remote_action_id = peer_action_id;

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
	if (local_action)
		fi_close(&local_action->fid);
	if (remote_action)
		fi_close(&remote_action->fid);
	free((void *) local_sem);
	free((void *) remote_sem);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;
	int lopt_idx = 0;
	struct option long_opts[] = {
		{"action-mode", required_argument, NULL, LONG_OPT_ACTION_MODE},
		{"action-width", required_argument, NULL, LONG_OPT_ACTION_WIDTH},
		{0, 0, 0, 0}
	};

	opts = INIT_OPTS;
	opts.rma_op = FT_RMA_WRITE;
	opts.transfer_size = 64;
	opts.comp_method = FT_COMP_SPIN;
	opts.iterations = 8;
	/* OOB sync so our own action-id exchange does not race the tx/rx seqs. */
	opts.options |= FT_OPT_OOB_SYNC;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "h" CS_OPTS INFO_OPTS API_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		case LONG_OPT_ACTION_MODE:
			if (!strcasecmp(optarg, "local"))
				action_mode = ACTION_MODE_LOCAL;
			else if (!strcasecmp(optarg, "remote"))
				action_mode = ACTION_MODE_REMOTE;
			else if (!strcasecmp(optarg, "both"))
				action_mode = ACTION_MODE_BOTH;
			else {
				fprintf(stderr,
					"action-mode must be local|remote|both\n");
				return EXIT_FAILURE;
			}
			break;
		case LONG_OPT_ACTION_WIDTH:
			action_width = atoi(optarg);
			if (action_width != 8 && action_width != 16 &&
			    action_width != 32) {
				fprintf(stderr,
					"action-width must be 8|16|32\n");
				return EXIT_FAILURE;
			}
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "EFA completion-action integration test.");
			FT_PRINT_OPTS_USAGE("--action-mode <m>",
				"which action(s) to attach: local|remote|both "
				"(default: both)");
			FT_PRINT_OPTS_USAGE("--action-width <w>",
				"action vector entry width in bits: 8|16|32 "
				"(default: 32)");
			FT_PRINT_OPTS_USAGE("-I <iters>",
				"number of writes that carry an action (default: 8)");
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

	/* Completion action is an efa-direct feature. */
	if (!hints->fabric_attr->name)
		hints->fabric_attr->name = strdup(EFA_DIRECT_FABRIC_NAME);
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_RMA;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	hints->addr_format = opts.address_format;
	hints->mode |= FI_CONTEXT2;

	printf("Action mode: %s\n",
	       action_mode == ACTION_MODE_LOCAL ? "LOCAL" :
	       action_mode == ACTION_MODE_REMOTE ? "REMOTE" : "BOTH");
	printf("Action width: %d bits\n", action_width);
	printf("Iterations: %d\n", iters);

	ret = run();

	ft_free_res();
	return ft_exit_code(ret);
}
