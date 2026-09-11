/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * EFA-specific RMA bandwidth test.
 *
 * This test measures RMA bandwidth with support for EFA-specific features
 * such as the FI_EFA_WR_HIGH_PPS flag and completion actions. It currently
 * supports write, writedata, and read operations.
 *
 * Unlike fi_rma_bw, this test uses a nonblocking benchmark loop that
 * interleaves posting and completion polling to keep the pipeline full,
 * similar to the approach used by rdma-core/perftest. This avoids blocking
 * at window boundaries and maximizes throughput.
 *
 * Multi-EP support:
 *   Multiple endpoints (--num-eps / -q) share a single CQ pair (txcq/rxcq)
 *   and AV. Each EP independently tracks its own posted and completed
 *   operation counts (per_ep_posted[] / per_ep_completed[]), enabling
 *   per-EP flow control: an EP may only have up to window_size operations
 *   in flight at any time.
 *
 * Per-EP completion attribution:
 *   Each posted operation carries an efa_rma_bw_ctx containing the EP index.
 *   On completion, the CQ entry's op_context is used (via container_of) to
 *   recover the efa_rma_bw_ctx and attribute the completion to the correct EP.
 *
 *   Context pool slots are partitioned per-EP:
 *     slot = ep_idx * window_size + (per_ep_posted[ep_idx] % window_size)
 *   This ensures an EP's in-flight contexts are never overwritten by another
 *   EP's posts.
 *
 *   For the writedata receiver (unsolicited write-with-imm without posted
 *   receives, i.e. !FI_RX_CQ_DATA), completions have no user context, so
 *   per_ep_completed is passed as NULL and only global counting is performed.
 *
 * FI_MORE batching (--post-list):
 *   When post_list > 1, FI_MORE is set on consecutive posts to the same EP
 *   to batch doorbell rings. FI_MORE is cleared (doorbell fires) when:
 *     - The post count hits a post_list boundary (per_ep_posted % post_list == 0)
 *     - The EP's per-iteration quota is reached (last post for that EP)
 *     - The EP's window is full (next post would exceed window_size outstanding)
 *   This guarantees the provider always receives a final non-FI_MORE post to
 *   flush the batch, preventing hangs from un-rung doorbells.
 *
 * Loop structure (initiator / tx side):
 *   Follows the perftest pattern: the outer while loop runs until all posts
 *   are issued and completed. Each pass iterates over ALL EPs (for-loop),
 *   posting as many operations as each EP's window allows. This prevents
 *   starvation — a full EP doesn't block others from making progress. After
 *   the posting pass, the shared CQ is polled once to harvest completions.
 *
 * Loop structure (writedata server / rx side):
 *   Pre-posts window_size receives per EP. Then polls rxcq and reposts to
 *   EPs that have room, using the same per-EP window logic. FI_MORE is
 *   applied to consecutive reposts on the same EP.
 *
 * Completion actions (--action-mode):
 *   Each write carries an EFA completion action, so the same loop that measures
 *   bandwidth also exercises the action path end to end:
 *
 *     1. Both sides enable action support on every endpoint before it is
 *        enabled (fi_setopt FI_OPT_EFA_COMP_ACTION).
 *     2. A memory completion action is created over a one-entry "semaphore"
 *        vector (create_mem_comp_action via the FI_EFA_MEM_COMP_ACTION_OPS ops
 *        table).
 *     3. The action id a peer's writes have to name is exchanged out of band.
 *     4. Writes go out with FI_EFA_MSG_ACTION_V1, attaching a local and/or
 *        remote action *with a value* (the value the device writes into the
 *        semaphore on completion).
 *     5. Each side verifies its semaphore was set to the expected value once
 *        the transfers are done.
 *
 *   A local action fires on the posting side's completion and a remote action
 *   where the write lands, so which side registers which follows the direction
 *   of the chosen op: write and read run both ways, writedata only from the
 *   client. Under FI_EFA_MEM_COMP_ACTION_SET_INITIATOR_VAL the device sets
 *   rather than accumulates, and every write carries the same value, so a
 *   semaphore holds that value once any write carrying the action has landed.
 *
 *   With -o writedata the writes also carry remote CQ data (write with
 *   immediate), so one work request exercises both device features and the
 *   receiver checks the data each completion carries. The device only carries a
 *   local action on such a write, not a remote one, so --action-mode remote and
 *   both are rejected with -o writedata.
 *
 *   Completion actions need an efa-direct endpoint, which the test asks for
 *   when no fabric was named. The test reports -FI_ENODATA, which fabtests
 *   treats as a skip, when the libfabric it is linked against has no action
 *   support or the device does not advertise it.
 *
 * Usage:
 *   Server: fi_efa_rma_bw
 *   Client: fi_efa_rma_bw -H <server_addr>
 *
 * Options:
 *   --high-pps        Enable FI_EFA_WR_HIGH_PPS flag on writes.
 *   -o write|writedata|read  Select RMA operation (default: write).
 *   --post-list <n>   Batch n posts per doorbell using FI_MORE (default: 1).
 *   -q <n>, --num-eps <n>  Number of endpoints/QPs (default: 1).
 *   --mr-relaxed-ordering  Register MRs with FI_EFA_MR_RELAXED_ORDERING.
 *   --action-mode local|remote|both  Attach a completion action to each write.
 *   --action-width 8|16|32  Action vector entry width in bits (default: 32).
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
#include <hmem.h>
#include "efa_shared.h"
#include "benchmarks/benchmark_shared.h"


#define EFA_RMA_BW_CQ_POLL_BATCH 16
#define EFA_RMA_BW_MAX_EPS 64

/*
 * Per-operation context that embeds the EP index so CQ completions
 * can be attributed to specific EPs.
 * ep_idx is placed before fi_context2 to avoid provider clobbering it.
 */
struct efa_rma_bw_ctx {
	int ep_idx;
	int pad;
	struct fi_context2 context;
};

#define EFA_RMA_BW_CTX_FROM_OP_CONTEXT(ptr) \
	container_of(ptr, struct efa_rma_bw_ctx, context)

static struct efa_rma_bw_ctx *tx_ctx_pool;
static struct efa_rma_bw_ctx *rx_ctx_pool;

static int use_high_pps;
static int post_list = 1;
static int num_eps = 1;
static int use_mr_relaxed_ordering;
static struct fid_ep *eps[EFA_RMA_BW_MAX_EPS];
static fi_addr_t remote_addrs[EFA_RMA_BW_MAX_EPS];

/*
 * Set on the side that receives writes with immediate while a completion action
 * rides along, where the data a completion carries is checked against what the
 * initiator sends.
 */
static bool check_cq_data;

#ifdef FI_EFA_MSG_ACTION_V1

/*
 * The value the device writes into a semaphore, per work request. Chosen to fit
 * any entry width and to be visibly non-trivial.
 */
#define ACTION_VALUE_LOCAL  0x5aU
#define ACTION_VALUE_REMOTE 0xa5U

enum action_mode {
	ACTION_MODE_OFF,
	ACTION_MODE_LOCAL,
	ACTION_MODE_REMOTE,
	ACTION_MODE_BOTH,
};

static enum action_mode action_mode = ACTION_MODE_OFF;
static int action_width = 32;

/*
 * One action target. It lives in whatever memory type -D selects, so on device
 * memory the NIC reaches it through a dmabuf fd instead of a host VA and the
 * value is not host readable: the location and the handle travel together.
 */
struct action_sem {
	volatile uint32_t *ptr;
	int dmabuf_fd;
	uint64_t dmabuf_offset;
	size_t size;
};

static struct fi_efa_ops_mem_comp_action *action_ops;

/* Action handles (fids), and the wire action ids used in the WR. */
static struct fid_efa_comp_action *local_action;
static struct fid_efa_comp_action *remote_action;
static uint32_t local_action_id;
static uint32_t remote_action_id;

/* Single-entry target vectors for the memory actions. */
static struct action_sem local_sem = { .dmabuf_fd = -1 };
static struct action_sem remote_sem = { .dmabuf_fd = -1 };

static bool action_enabled(void)
{
	return action_mode != ACTION_MODE_OFF;
}

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
 * Which side posts: write and read run both ways, writedata only from the
 * client. A local action fires on the posting side's completion, a remote action
 * on the side the writes land on, so this decides which actions a side needs.
 */
static bool i_post(void)
{
	return opts.rma_op != FT_RMA_WRITEDATA || opts.dst_addr;
}

static bool peer_posts(void)
{
	return opts.rma_op != FT_RMA_WRITEDATA || !opts.dst_addr;
}

/* The action target shares the memory type of the tx/rx buffers. */
#define ACTION_MEM_IS_HOST (opts.iface == FI_HMEM_SYSTEM)

/*
 * Device allocators round up anyway and the action only touches the first
 * entry, so ask for a page rather than four bytes.
 */
#define ACTION_SEM_DEVICE_SIZE 4096

static int alloc_action_sem(struct action_sem *sem)
{
	int ret;

	if (ACTION_MEM_IS_HOST) {
		sem->size = sizeof(*sem->ptr);
		sem->ptr = calloc(1, sem->size);
		return sem->ptr ? FI_SUCCESS : -FI_ENOMEM;
	}

	sem->size = ACTION_SEM_DEVICE_SIZE;
	ret = ft_hmem_alloc(opts.iface, opts.device, (void **) &sem->ptr,
			    sem->size);
	if (ret) {
		FT_PRINTERR("ft_hmem_alloc", ret);
		return ret;
	}

	ret = ft_hmem_memset(opts.iface, opts.device, (void *) sem->ptr, 0,
			     sem->size);
	if (ret) {
		FT_PRINTERR("ft_hmem_memset", ret);
		return ret;
	}

	ret = ft_hmem_get_dmabuf_fd(opts.iface, (void *) sem->ptr, sem->size,
				    &sem->dmabuf_fd, &sem->dmabuf_offset);
	if (ret) {
		FT_PRINTERR("ft_hmem_get_dmabuf_fd", ret);
		return ret;
	}

	FT_INFO("Action target on device memory: va %p, dmabuf fd %d, "
		"offset %lu", (void *) sem->ptr, sem->dmabuf_fd,
		(unsigned long) sem->dmabuf_offset);
	return FI_SUCCESS;
}

static void free_action_sem(struct action_sem *sem)
{
	if (!sem->ptr)
		return;

	if (ACTION_MEM_IS_HOST) {
		free((void *) sem->ptr);
	} else {
		if (sem->dmabuf_fd >= 0)
			ft_hmem_put_dmabuf_fd(opts.iface, sem->dmabuf_fd);
		ft_hmem_free(opts.iface, (void *) sem->ptr);
	}
	sem->ptr = NULL;
	sem->dmabuf_fd = -1;
}

/* Device memory is not host readable, so the value comes back through a copy. */
static int read_action_sem(struct action_sem *sem, uint32_t *val)
{
	if (ACTION_MEM_IS_HOST) {
		*val = *sem->ptr;
		return FI_SUCCESS;
	}

	return ft_hmem_copy_from(opts.iface, opts.device, val,
				 (void *) sem->ptr, sizeof(*val));
}

/*
 * Create a memory completion action over @sem, a one-entry vector of
 * action_width bits. On success sets *action and *action_id (the id a WR names
 * the action by).
 */
static int register_mem_action(struct action_sem *sem,
			       struct fid_efa_comp_action **action,
			       uint32_t *action_id)
{
	struct fi_efa_mem_comp_action_attr attr = {0};
	int ret;

	attr.op = FI_EFA_MEM_COMP_ACTION_SET_INITIATOR_VAL;
	if (ACTION_MEM_IS_HOST) {
		attr.location.type = FI_EFA_MEMORY_LOCATION_VA;
		attr.location.ptr = (uint8_t *) sem->ptr;
	} else {
		attr.location.type = FI_EFA_MEMORY_LOCATION_DMABUF;
		attr.location.dmabuf.fd = sem->dmabuf_fd;
		attr.location.dmabuf.offset = sem->dmabuf_offset;
	}
	attr.num_entries = 1;
	attr.entry_size = action_width / 8;

	ret = action_ops->create_mem_comp_action(domain, &attr, action);
	if (ret) {
		FT_PRINTERR("create_mem_comp_action", ret);
		return ret;
	}

	*action_id = fid_efa_comp_action_get_id(*action);
	FT_INFO("Created memory action: op SET_INITIATOR_VAL, target %p (%s), "
		"%u entry of %u bytes, id %u",
		(void *) sem->ptr,
		ACTION_MEM_IS_HOST ? "host VA" : "device dmabuf",
		attr.num_entries, attr.entry_size, *action_id);
	return FI_SUCCESS;
}

/*
 * An endpoint with actions enabled carries a wider send queue entry, which costs
 * both send-queue depth and inline room, so report what the endpoint says it has
 * now that fi_setopt() has gone through.
 */
static int report_action_ep_limits(struct fid_ep *target_ep)
{
	size_t tx_size, inject_msg_size, inject_rma_size;
	size_t len;
	int ret;

	len = sizeof(tx_size);
	ret = fi_getopt(&target_ep->fid, FI_OPT_ENDPOINT, FI_OPT_TX_SIZE,
			&tx_size, &len);
	if (ret) {
		FT_PRINTERR("fi_getopt(FI_OPT_TX_SIZE)", ret);
		return ret;
	}

	len = sizeof(inject_msg_size);
	ret = fi_getopt(&target_ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_MSG_SIZE,
			&inject_msg_size, &len);
	if (ret) {
		FT_PRINTERR("fi_getopt(FI_OPT_INJECT_MSG_SIZE)", ret);
		return ret;
	}

	len = sizeof(inject_rma_size);
	ret = fi_getopt(&target_ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE,
			&inject_rma_size, &len);
	if (ret) {
		FT_PRINTERR("fi_getopt(FI_OPT_INJECT_RMA_SIZE)", ret);
		return ret;
	}

	FT_INFO("Endpoint with actions enabled: tx size %zu, inject msg size "
		"%zu, inject rma size %zu", tx_size, inject_msg_size,
		inject_rma_size);

	/*
	 * ft_enable_ep() sets both inject options from this single number, and
	 * the endpoint refuses more inline room than it has, so take the
	 * smaller of the two it just reported. A zero rma size means this
	 * endpoint has no inline rma room at all (the -j size stayed within the
	 * device's regular inline buffer, so the send queue entry is the narrow
	 * one), and ft_enable_ep() then leaves both options alone.
	 */
	if (opts.inject_size > MIN(inject_msg_size, inject_rma_size))
		opts.inject_size = MIN(inject_msg_size, inject_rma_size);

	return FI_SUCCESS;
}

/* Open the action ops table on the domain. */
static int open_action_ops(void)
{
	uint32_t max_mem_comp_actions;
	int ret;

	ret = fi_open_ops(&domain->fid, FI_EFA_MEM_COMP_ACTION_OPS, 0,
			  (void **) &action_ops, NULL);
	if (ret) {
		FT_PRINTERR("fi_open_ops(FI_EFA_MEM_COMP_ACTION_OPS)", ret);
		return ret;
	}

	/*
	 * Informational only. fi_setopt below is what decides whether the test
	 * can run, so report whatever the device says and carry on either way.
	 */
	ret = action_ops->query_max_mem_comp_actions(domain,
						     &max_mem_comp_actions);
	if (ret)
		FT_WARN("query_max_mem_comp_actions failed: %d (%s)", ret,
			fi_strerror(-ret));
	else
		FT_INFO("Device reports max_mem_comp_actions %u",
			max_mem_comp_actions);

	return FI_SUCCESS;
}

/* Enable action support on an endpoint. Must be before the endpoint is enabled. */
static int enable_action_on_ep(struct fid_ep *target_ep)
{
	bool enable = true;
	int ret;

	ret = fi_setopt(&target_ep->fid, FI_OPT_ENDPOINT,
			FI_OPT_EFA_COMP_ACTION, &enable, sizeof(enable));
	if (ret) {
		if (ret == -FI_EOPNOTSUPP) {
			/*
			 * Device/driver does not advertise completion-action
			 * support. Report as ENODATA so fabtests treats it as a
			 * skip rather than a failure.
			 */
			FT_WARN("Completion action not supported by device, "
				"skipping test");
			return -FI_ENODATA;
		}
		FT_PRINTERR("fi_setopt(FI_OPT_EFA_COMP_ACTION)", ret);
		return ret;
	}

	return FI_SUCCESS;
}

/*
 * The extra endpoints --num-eps asks for are created after the fabric is up, so
 * they need the same fi_setopt between fi_endpoint() and ft_enable_ep().
 */
static int enable_action_on_new_ep(struct fid_ep *target_ep)
{
	if (!action_enabled())
		return FI_SUCCESS;

	return enable_action_on_ep(target_ep);
}

/*
 * ft_init_fabric() with fi_setopt for action support between endpoint creation
 * and enable, which is the one point the option can be set.
 */
static int init_fabric_with_action(void)
{
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

	ret = open_action_ops();
	if (ret)
		return ret;

	ret = enable_action_on_ep(ep);
	if (ret)
		return ret;

	ret = report_action_ep_limits(ep);
	if (ret)
		return ret;

	ret = ft_enable_ep_recv();
	if (ret)
		return ret;

	return ft_init_av();
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
 * Register the actions this side needs and swap the id a peer's writes have to
 * name. A side that posts registers the local action; a side the peer writes to
 * registers the remote action, and sends that id across.
 */
static int setup_actions(void)
{
	uint32_t peer_action_id = 0;
	int ret;

	if (!action_enabled())
		return FI_SUCCESS;

	if (want_local() && i_post()) {
		ret = alloc_action_sem(&local_sem);
		if (ret)
			return ret;
		ret = register_mem_action(&local_sem, &local_action,
					  &local_action_id);
		if (ret)
			return ret;
	}

	if (want_remote() && peer_posts()) {
		ret = alloc_action_sem(&remote_sem);
		if (ret)
			return ret;
		ret = register_mem_action(&remote_sem, &remote_action,
					  &remote_action_id);
		if (ret)
			return ret;
	}

	ret = oob_exchange_u32(remote_action_id, &peer_action_id);
	if (ret)
		return ret;

	if (want_remote() && i_post())
		remote_action_id = peer_action_id;

	return FI_SUCCESS;
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
static int wait_sem(struct action_sem *sem, uint32_t expected)
{
	struct timespec a, b;
	uint32_t mask = sem_mask(action_width);
	int wait_s = (timeout >= 0) ? timeout : 10;
	uint32_t val = 0;
	int ret;

	clock_gettime(CLOCK_MONOTONIC, &a);
	for (;;) {
		ret = read_action_sem(sem, &val);
		if (ret)
			return ret;
		if ((val & mask) == (expected & mask))
			return 0;
		ft_force_progress();
		clock_gettime(CLOCK_MONOTONIC, &b);
		if ((b.tv_sec - a.tv_sec) > wait_s) {
			FT_ERR("semaphore timeout: got 0x%x, expected 0x%x",
			       val & mask, expected & mask);
			return -FI_ETIMEDOUT;
		}
	}
}

/* Every write carries the same value, so one check per action covers the run. */
static int verify_actions(void)
{
	uint32_t val = 0;
	int ret;

	if (!action_enabled())
		return FI_SUCCESS;

	if (local_action) {
		ret = wait_sem(&local_sem, ACTION_VALUE_LOCAL);
		if (ret)
			return ret;
		(void) read_action_sem(&local_sem, &val);
		FT_INFO("Local action verified: %s semaphore = 0x%x",
			ACTION_MEM_IS_HOST ? "host" : "device",
			val & sem_mask(action_width));
	}

	if (remote_action) {
		ret = wait_sem(&remote_sem, ACTION_VALUE_REMOTE);
		if (ret)
			return ret;
		(void) read_action_sem(&remote_sem, &val);
		FT_INFO("Remote action verified: %s semaphore = 0x%x",
			ACTION_MEM_IS_HOST ? "host" : "device",
			val & sem_mask(action_width));
	}

	return FI_SUCCESS;
}

static void free_actions(void)
{
	if (local_action)
		fi_close(&local_action->fid);
	if (remote_action)
		fi_close(&remote_action->fid);
	free_action_sem(&local_sem);
	free_action_sem(&remote_sem);
}

#else /* FI_EFA_MSG_ACTION_V1 */

#define action_enabled() false

static int enable_action_on_new_ep(struct fid_ep *target_ep)
{
	(void) target_ep;
	return FI_SUCCESS;
}

static int setup_actions(void)
{
	return FI_SUCCESS;
}

static int verify_actions(void)
{
	return FI_SUCCESS;
}

static void free_actions(void)
{
}

#endif /* FI_EFA_MSG_ACTION_V1 */

static ssize_t post_rma(struct fid_ep *target_ep, fi_addr_t target_addr,
			char *buf, size_t size,
			struct fi_rma_iov *remote, void *context,
			uint64_t base_flags)
{
#ifdef FI_EFA_MSG_ACTION_V1
	struct fi_efa_msg_rma_action_v1 emsg;
	struct fi_msg_rma *msg = &emsg.msg;
#else
	struct fi_msg_rma msg_storage;
	struct fi_msg_rma *msg = &msg_storage;
#endif
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;
	uint64_t flags = base_flags;
	ssize_t ret;

#ifdef FI_EFA_MSG_ACTION_V1
	/*
	 * The fields past the plain work request only mean anything with
	 * FI_EFA_MSG_ACTION_V1 set, so a run without actions does not pay for
	 * clearing them.
	 */
	if (action_enabled())
		memset(&emsg, 0, sizeof(emsg));
#endif

	msg_iov.iov_base = buf;
	msg_iov.iov_len = size;
	msg->msg_iov = &msg_iov;
	msg->desc = &mr_desc;
	msg->iov_count = 1;
	rma_iov.addr = remote->addr + (buf - (opts.rma_op == FT_RMA_READ ? rx_buf : tx_buf));
	rma_iov.len = size;
	rma_iov.key = remote->key;
	msg->rma_iov = &rma_iov;
	msg->rma_iov_count = 1;
	msg->addr = target_addr;
	msg->context = context;

	if (opts.rma_op == FT_RMA_READ) {
		msg->data = 0;
		return fi_readmsg(target_ep, msg, flags);
	}

	if (use_high_pps)
		flags |= FI_EFA_WR_HIGH_PPS;
	if (opts.rma_op == FT_RMA_WRITEDATA) {
		flags |= FI_REMOTE_CQ_DATA;
		msg->data = remote_cq_data;
	} else {
		msg->data = 0;
	}

#ifdef FI_EFA_MSG_ACTION_V1
	/*
	 * The action's vector holds a single entry, so entry 0 is the only one a
	 * WR can name and the INDEX feature bits are left unset (an unset index
	 * is 0). The action block and the immediate data occupy different parts
	 * of the WQE, so a writedata WR carries both.
	 */
	if (action_enabled()) {
		if (want_local()) {
			emsg.feature_bits |= FI_EFA_LOCAL_ACTION_ID |
					     FI_EFA_LOCAL_ACTION_VALUE;
			emsg.local_id = local_action_id;
			emsg.local_value = ACTION_VALUE_LOCAL;
		}
		if (want_remote()) {
			emsg.feature_bits |= FI_EFA_REMOTE_ACTION_ID |
					     FI_EFA_REMOTE_ACTION_VALUE;
			emsg.remote_id = remote_action_id;
			emsg.remote_value = ACTION_VALUE_REMOTE;
		}
		flags |= FI_EFA_MSG_ACTION_V1;
	}
#endif

	ret = fi_writemsg(target_ep, msg, flags);

	return ret;
}

/*
 * Poll CQ for completions in a nonblocking manner.
 * If per_ep_completed is non-NULL, each completion's op_context is used to
 * recover the efa_rma_bw_ctx and credit the correct EP. For the target side
 * of unsolicited write recv, there is no rx buffer post and no per-EP completion
 * needed (because it is used as the credit to repost rx buffer), the per_ep_completed
 * is not needed and will be passed as NULL.
 * Returns the number of completions harvested, or negative on error.
 */
static int bw_comp_nonblocking(struct fid_cq *cq, uint64_t *cq_cntr,
			       int *completed_cnt,
			       int *per_ep_completed)
{
	int ret, cnt = 0, i;
	struct fi_cq_data_entry comp[EFA_RMA_BW_CQ_POLL_BATCH];
	struct efa_rma_bw_ctx *ctx;

	while ((ret = fi_cq_read(cq, comp, EFA_RMA_BW_CQ_POLL_BATCH)) > 0) {
		if (per_ep_completed) {
			for (i = 0; i < ret; i++) {
				ctx = EFA_RMA_BW_CTX_FROM_OP_CONTEXT(comp[i].op_context);
				per_ep_completed[ctx->ep_idx]++;
			}
		}

		/*
		 * The immediate data is the only part of the write the target
		 * sees directly, so where an action rides along it is worth
		 * checking that both arrived.
		 */
		if (check_cq_data) {
			for (i = 0; i < ret; i++) {
				if (comp[i].data == remote_cq_data)
					continue;
				FT_ERR("cq data mismatch: got 0x%lx, expected "
				       "0x%lx", (unsigned long) comp[i].data,
				       (unsigned long) remote_cq_data);
				return -FI_EIO;
			}
		}

		(*completed_cnt) += ret;
		(*cq_cntr) += ret;
		cnt += ret;
	}

	if (ret == -FI_EAVAIL) {
		ret = ft_cq_readerr(cq);
		return ret;
	}

	if (ret < 0 && ret != -FI_EAGAIN) {
		FT_PRINTERR("fi_cq_read", ret);
		return ret;
	}

	return cnt;
}

/*
 * Post a receive buffer on the given EP. The context slot is determined
 * per-EP (ep_idx * window_size + posted % window_size) to avoid cross-EP
 * slot reuse. The ep_idx is stamped into the context so completions can
 * be attributed back to this EP.
 */
static int post_rx(int ep_idx, int *per_ep_posted, int *per_ep_completed,
		   int *posted_cnt, uint64_t rx_flags)
{
	int slot = ep_idx * opts.window_size +
		   (per_ep_posted[ep_idx] % opts.window_size);
	struct iovec iov = {
		.iov_base = rx_buf,
		.iov_len = FT_MAX_CTRL_MSG + ft_rx_prefix_size(),
	};
	struct fi_msg msg = {
		.msg_iov = &iov,
		.desc = &mr_desc,
		.iov_count = 1,
		.addr = FI_ADDR_UNSPEC,
		.context = &rx_ctx_pool[slot].context,
	};
	int ret;

	rx_ctx_pool[slot].ep_idx = ep_idx;
	per_ep_posted[ep_idx]++;

	ret = fi_recvmsg(eps[ep_idx], &msg, rx_flags);
	if (ret) {
		per_ep_posted[ep_idx]--;
		return ret;
	}
	(*posted_cnt)++;
	return 0;
}

/*
 * Unified post/poll loop for both TX (initiator) and RX (writedata server).
 *
 * TX side: posts RMA ops round-robin across EPs, polls txcq.
 * RX side: pre-posts recvs, polls rxcq, reposts on completion.
 *
 * iters_per_ep: number of operations per EP to complete.
 * do_measure: if true, calls ft_start()/ft_stop() around the loop.
 */
static int run_loop(struct fi_rma_iov *remote, size_t rma_start_offset,
		    int iters_per_ep, bool do_measure)
{
	int ret, posted_cnt = 0, completed_cnt = 0;
	int per_ep_posted[EFA_RMA_BW_MAX_EPS] = {0};
	int per_ep_completed[EFA_RMA_BW_MAX_EPS] = {0};
	int total_posts = iters_per_ep * num_eps;
	size_t offset;
	char *buf;
	uint64_t flags;

	if (opts.rma_op == FT_RMA_WRITEDATA && !opts.dst_addr) {
		/* Server side for writedata: pre-post rx buffers round-robin. */
		if (fi->rx_attr->mode & FI_RX_CQ_DATA) {
			int pre_post_limit = MIN(opts.window_size, iters_per_ep);
			for (int ep_idx = 0; ep_idx < num_eps; ep_idx++) {
				while (per_ep_posted[ep_idx] < pre_post_limit) {
					ret = post_rx(ep_idx, per_ep_posted,
						      per_ep_completed,
						      &posted_cnt, 0);
					if (ret == -FI_EAGAIN)
						break;
					if (ret)
						return ret;
				}
			}
		}

		if (do_measure)
			ft_start();

		/* Poll rxcq for completions, reposting to same EP. */
		while (completed_cnt < total_posts) {
			ret = bw_comp_nonblocking(rxcq, &rx_cq_cntr,
						  &completed_cnt,
						  (fi->rx_attr->mode & FI_RX_CQ_DATA) ?
						  per_ep_completed : NULL);
			if (ret < 0)
				return ret;

			if ((fi->rx_attr->mode & FI_RX_CQ_DATA) && ret > 0) {
				/* Repost to EPs that have room in their window */
				for (int ep_idx = 0; ep_idx < num_eps; ep_idx++) {
					while (per_ep_posted[ep_idx] <
					       iters_per_ep &&
					       (per_ep_posted[ep_idx] -
					        per_ep_completed[ep_idx]) <
					       opts.window_size) {
						/*
						 * FI_MORE for rx: peek ahead to
						 * check if the next post on this
						 * EP will also proceed (not at
						 * post_list boundary, iteration
						 * limit, or window limit).
						 */
						uint64_t rx_flags = 0;

						if (post_list > 1 &&
						    (per_ep_posted[ep_idx] + 1) % post_list &&
						    per_ep_posted[ep_idx] + 1 <
						    iters_per_ep &&
						    (per_ep_posted[ep_idx] + 1 -
						     per_ep_completed[ep_idx]) <
						    opts.window_size)
							rx_flags = FI_MORE;

						ret = post_rx(ep_idx,
							      per_ep_posted,
							      per_ep_completed,
							      &posted_cnt,
							      rx_flags);
						if (ret == -FI_EAGAIN)
							break;
						if (ret)
							return ret;
					}
				}
			}
		}
	} else {
		if (do_measure)
			ft_start();

		/* Initiator side: post RMA ops, try all EPs each pass (perftest style) */
		while (posted_cnt < total_posts ||
		       completed_cnt < total_posts) {
			for (int ep_idx = 0; ep_idx < num_eps; ep_idx++) {
				while (per_ep_posted[ep_idx] < iters_per_ep &&
				       (per_ep_posted[ep_idx] -
				        per_ep_completed[ep_idx]) <
				       opts.window_size) {
					int slot = ep_idx * opts.window_size +
						   (per_ep_posted[ep_idx] %
						    opts.window_size);

					offset = rma_start_offset +
						 (per_ep_posted[ep_idx] %
						  opts.window_size) *
							opts.transfer_size;

					buf = (opts.rma_op == FT_RMA_READ) ?
					      rx_buf + offset : tx_buf + offset;

					tx_ctx_pool[slot].ep_idx = ep_idx;
					per_ep_posted[ep_idx]++;

					/*
					 * FI_MORE: batch posts within the same EP.
					 * Set only when all three conditions hold:
					 * 1. Not at a post_list boundary
					 * 2. Not at the EP's iteration limit
					 * 3. Window won't be full after this post
					 * This ensures the next iteration of the
					 * inner while will also execute, so a
					 * non-FI_MORE post always follows to ring
					 * the doorbell.
					 */
					flags = 0;
					if (post_list > 1 &&
					    per_ep_posted[ep_idx] % post_list &&
					    per_ep_posted[ep_idx] <
					    iters_per_ep &&
					    (per_ep_posted[ep_idx] -
					     per_ep_completed[ep_idx]) <
					    opts.window_size)
						flags = FI_MORE;

					ret = post_rma(eps[ep_idx],
							remote_addrs[ep_idx],
							buf, opts.transfer_size,
							remote,
							&tx_ctx_pool[slot].context,
							flags);
					if (ret == -FI_EAGAIN) {
						per_ep_posted[ep_idx]--;
						break;
					}
					if (ret)
						return ret;
					posted_cnt++;
				}
			}

			ret = bw_comp_nonblocking(txcq, &tx_cq_cntr,
						  &completed_cnt,
						  per_ep_completed);
			if (ret < 0)
				return ret;
		}
	}

	if (do_measure)
		ft_stop();

	return 0;
}

static int bandwidth_rma_efa(struct fi_rma_iov *remote)
{
	int ret;
	size_t rma_start_offset;
	int pool_size = opts.window_size * num_eps;

	tx_ctx_pool = calloc(pool_size, sizeof(*tx_ctx_pool));
	rx_ctx_pool = calloc(pool_size, sizeof(*rx_ctx_pool));
	if (!tx_ctx_pool || !rx_ctx_pool) {
		ret = -FI_ENOMEM;
		goto out_free;
	}

	rma_start_offset = FT_RMA_SYNC_MSG_BYTES +
			   MAX(ft_tx_prefix_size(), ft_rx_prefix_size());

	/* Warmup */
	ret = ft_sync();
	if (ret)
		goto out_free;

	ret = run_loop(remote, rma_start_offset,
		       opts.warmup_iterations, false);
	if (ret)
		goto out_free;

	/* Measurement */
	ret = ft_sync();
	if (ret)
		goto out_free;

	ret = run_loop(remote, rma_start_offset,
		       opts.iterations, true);
	if (ret)
		goto out_free;

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations * num_eps,
			     &start, &end, 1, opts.argc, opts.argv);
	else
		show_perf(NULL, opts.transfer_size, opts.iterations * num_eps,
			  &start, &end, 1);

	ret = 0;
out_free:
	free(tx_ctx_pool);
	free(rx_ctx_pool);
	tx_ctx_pool = NULL;
	rx_ctx_pool = NULL;
	return ret;
}

static int init_fabric(void)
{
#ifdef FI_EFA_MSG_ACTION_V1
	if (action_enabled())
		return init_fabric_with_action();
#endif
	return ft_init_fabric();
}

static int run(void)
{
	int i, ret;

	/*
	 * Use FI_CQ_FORMAT_DATA so the CQ entry type matches the
	 * fi_cq_data_entry buffer in bw_comp_nonblocking. Without this,
	 * ft_init_fabric defaults to FI_CQ_FORMAT_CONTEXT (no FI_TAGGED cap),
	 * causing a mismatch with our fi_cq_read calls.
	 */
	cq_attr.format = FI_CQ_FORMAT_DATA;

	ret = init_fabric();
	if (ret)
		return ret;

	/* eps[0] is the default ep created by ft_init_fabric */
	eps[0] = ep;
	remote_addrs[0] = remote_fi_addr;

	/* Create additional EPs, all bound to the same CQs and AV */
	for (i = 1; i < num_eps; i++) {
		ret = fi_endpoint(domain, fi, &eps[i], NULL);
		if (ret) {
			FT_PRINTERR("fi_endpoint", ret);
			return ret;
		}
		ret = enable_action_on_new_ep(eps[i]);
		if (ret)
			return ret;
		ret = ft_enable_ep(eps[i], eq, av, txcq, rxcq,
				   txcntr, rxcntr, rma_cntr);
		if (ret)
			return ret;
		ret = ft_init_av_addr(av, eps[i], &remote_addrs[i]);
		if (ret)
			return ret;
	}

	ret = ft_exchange_keys(&remote);
	if (ret)
		return ret;

	/*
	 * ft_exchange_keys() leaves a pre-posted receive (context = &rx_ctx)
	 * on eps[0]. Consume it with a dummy message so it doesn't get
	 * matched by writedata completions producing a bogus op_context.
	 */
	if (opts.dst_addr) {
		ret = ft_post_tx(ep, remote_fi_addr, 1, NO_CQ_DATA, &tx_ctx);
		if (ret)
			return ret;
		ret = ft_get_tx_comp(tx_seq);
	} else {
		ret = ft_get_rx_comp(rx_seq);
	}
	if (ret)
		return ret;

	/* Actions have to be in place before the first write that names one. */
	ret = setup_actions();
	if (ret)
		goto out;

	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			opts.transfer_size = test_size[i].size;
			init_test(&opts, test_name, sizeof(test_name));
			ret = bandwidth_rma_efa(&remote);
			if (ret)
				goto out;
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = bandwidth_rma_efa(&remote);
		if (ret)
			goto out;
	}

	ret = verify_actions();
	if (ret)
		goto out;

	ret = ft_finalize();
out:
	free_actions();
	for (i = 1; i < num_eps; i++)
		FT_CLOSE_FID(eps[i]);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_BW;
	opts.rma_op = FT_RMA_WRITE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	hints->caps = FI_MSG | FI_RMA;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	hints->addr_format = opts.address_format;

	build_efa_long_opts();

	while ((op = getopt_long(argc, argv, "hq:" CS_OPTS INFO_OPTS API_OPTS
			    BENCHMARK_OPTS, efa_long_opts,
			    &lopt_idx)) != -1) {
		switch (op) {
		case OPT_HIGH_PPS:
			use_high_pps = 1;
			break;
		case OPT_POST_LIST:
			post_list = atoi(optarg);
			break;
		case OPT_NUM_EPS:
		case 'q':
			num_eps = atoi(optarg);
			if (num_eps < 1 || num_eps > EFA_RMA_BW_MAX_EPS) {
				fprintf(stderr, "num-eps must be 1-%d\n",
					EFA_RMA_BW_MAX_EPS);
				return EXIT_FAILURE;
			}
			break;
		case OPT_MR_RELAXED_ORDERING:
			use_mr_relaxed_ordering = 1;
			break;
#ifdef FI_EFA_MSG_ACTION_V1
		case OPT_ACTION_MODE:
			if (!strcasecmp(optarg, "local"))
				action_mode = ACTION_MODE_LOCAL;
			else if (!strcasecmp(optarg, "remote"))
				action_mode = ACTION_MODE_REMOTE;
			else if (!strcasecmp(optarg, "both"))
				action_mode = ACTION_MODE_BOTH;
			else {
				FT_ERR("action-mode must be local|remote|both");
				return EXIT_FAILURE;
			}
			break;
		case OPT_ACTION_WIDTH:
			action_width = atoi(optarg);
			if (action_width != 8 && action_width != 16 &&
			    action_width != 32) {
				FT_ERR("action-width must be 8|16|32");
				return EXIT_FAILURE;
			}
			break;
#else
		case OPT_ACTION_MODE:
		case OPT_ACTION_WIDTH:
			/*
			 * Built against a libfabric without completion actions,
			 * so there is nothing to run: ENODATA is a skip.
			 */
			FT_WARN("libfabric has no FI_EFA_MSG_ACTION_V1, "
				"skipping test");
			return ft_exit_code(-FI_ENODATA);
#endif
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "EFA RMA bandwidth test.");
			ft_benchmark_usage();
			FT_PRINT_OPTS_USAGE("-o <op>",
				"RMA op type: write|writedata|read (default: write)");
			fprintf(stderr, "Note: read/write bw tests are bidirectional.\n"
					"      writedata bw test is unidirectional"
					" from the client side.\n");
			efa_longopts_usage();
			return EXIT_FAILURE;
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_benchmark_opts(op, optarg);
			ft_parse_api_opts(op, optarg, hints, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->tx_attr->tclass = FI_TC_BULK_DATA;
	/* Using OOB sync to not mess up with the tx/rx seq cntrs in fabtests common code */
	opts.options |= FT_OPT_OOB_SYNC;

	const char *op_str = "WRITE";
	if (opts.rma_op == FT_RMA_WRITEDATA)
		op_str = "WRITEDATA";
	else if (opts.rma_op == FT_RMA_READ)
		op_str = "READ";

	if (use_high_pps)
		printf("High PPS mode: ENABLED\n");
	else
		printf("High PPS mode: DISABLED\n");

	if (use_mr_relaxed_ordering) {
		ft_mr_reg_flags |= FI_EFA_MR_RELAXED_ORDERING;
		FT_INFO("MR relaxed ordering: ENABLED");
	} else {
		FT_INFO("MR relaxed ordering: DISABLED");
	}

	printf("RMA op: %s\n", op_str);
	printf("Num EPs: %d\n", num_eps);

#ifdef FI_EFA_MSG_ACTION_V1
	if (action_enabled()) {
		if (opts.rma_op == FT_RMA_READ) {
			FT_ERR("completion actions ride on writes, so "
			       "--action-mode needs -o write or -o writedata");
			return EXIT_FAILURE;
		}

		/*
		 * The device carries a remote action on a plain write only: a
		 * write with immediate names the action but the write is
		 * rejected at the target, so keep the combination out.
		 */
		if (opts.rma_op == FT_RMA_WRITEDATA && want_remote()) {
			FT_ERR("the device does not carry a remote completion "
			       "action on a write with immediate, so -o "
			       "writedata needs --action-mode local");
			return EXIT_FAILURE;
		}

		/* Completion action is an efa-direct feature. */
		if (!hints->fabric_attr->name) {
			hints->fabric_attr->name =
				strdup(EFA_DIRECT_FABRIC_NAME);
			if (!hints->fabric_attr->name)
				return EXIT_FAILURE;
		}

		/*
		 * The immediate data is delivered to the target and the CQ is
		 * already read in FI_CQ_FORMAT_DATA, so the receiving side can
		 * check that the data and the action arrived together.
		 */
		check_cq_data = opts.rma_op == FT_RMA_WRITEDATA &&
				!opts.dst_addr;

		printf("Action mode: %s, action width: %d bits, "
		       "action target: %s\n",
		       action_mode == ACTION_MODE_LOCAL  ? "LOCAL" :
		       action_mode == ACTION_MODE_REMOTE ? "REMOTE" : "BOTH",
		       action_width, ACTION_MEM_IS_HOST ? "HOST" : "DEVICE");
	}
#endif

	ret = run();

	cleanup_ret = ft_free_res();
	return -(ret ? ret : cleanup_ret);
}
