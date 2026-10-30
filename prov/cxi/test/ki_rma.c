/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2026 Hewlett Packard Enterprise Development LP
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"
#include "cxip_ki_test_common.h"

/* End-to-end KI data-path tests. A GPU is not required: get_gpu_ep_info()
 * publishes host pointers for the command buffer and doorbell, so these
 * tests build the same DMA commands a GPU kernel would, emit them from
 * the host, and target the endpoint's own address (loopback). Completion
 * is observed on the bound send counter (ep_info.ctn).
 */

#define KI_RMA_WIN_KEY   0x1f
#define KI_RMA_WIN_LEN   (4 * 4096)
#define KI_RMA_WAIT_SEC  2
#define KI_RMA_SETTLE_US 200000
#define KI_RMA_INIT_VAL  0xaa
#define KI_RMA_PUT_VAL   0x55

static struct fi_cxi_gpu_ep_info ki_ep_info;
static struct fi_cxi_gpu_target_info ki_self_target;
static struct fi_cxi_gpu_remote_mr ki_self_remote_mr;
static struct fi_cxi_gpu_mr_info ki_local_mr_info;
static struct mem_region ki_mem_window;
static uint32_t ki_initiator;

/* Insert the endpoint's own address, create an MR, and resolve the
 * GPU-side metadata a command needs. Called after one of the
 * cxit_setup_ki_full* fixtures has enabled the endpoint.
 */
static void ki_rma_init(void)
{
	int ret;
	struct cxip_ep *cep;
	fi_addr_t self_addr;
	uint64_t key = KI_RMA_WIN_KEY;

	ret = fi_av_insert(cxit_av, &cxit_ep_addr, 1, &self_addr, 0, NULL);
	cr_assert_eq(ret, 1, "fi_av_insert returned %d", ret);

	cxit_ki_create_remote_mr(KI_RMA_WIN_LEN,
				 FI_REMOTE_READ | FI_REMOTE_WRITE, &key,
				 &ki_mem_window, NULL, &ki_self_remote_mr);

	ret = cxit_ki_ops->get_gpu_ep_info(&cxit_ep->fid, &ki_ep_info);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_ep_info returned %d", ret);
	cr_assert_neq(ki_ep_info.ctn, C_CT_NONE, "no send counter bound");

	ret = cxit_ki_ops->get_gpu_target_info(&cxit_ep->fid, &self_addr, 1,
					       &ki_self_target);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_target_info returned %d",
		     ret);

	ret = cxit_ki_ops->get_gpu_mr_info(&cxit_ep->fid,
					   &ki_mem_window.mr->fid,
					   ki_mem_window.mem,
					   &ki_local_mr_info);
	cr_assert_eq(ret, FI_SUCCESS, "get_gpu_mr_info returned %d", ret);

	/* Same initiator match ID the provider's own RMA path uses */
	cep = container_of(cxit_ep, struct cxip_ep, ep);
	ki_initiator = cxip_msg_match_id(cep->ep_obj->txc);
}

static void ki_rma_setup(void)
{
	cxit_setup_ki_full();
	ki_rma_init();
}

static void ki_rma_prov_key_setup(void)
{
	cxit_setup_ki_full_prov_key();
	ki_rma_init();
}

static void ki_rma_teardown(void)
{
	mr_destroy(&ki_mem_window);
	cxit_teardown_ki_full();
}

TestSuite(ki_rma, .init = ki_rma_setup, .fini = ki_rma_teardown,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Same fixture on an FI_MR_PROV_KEY domain */
TestSuite(ki_rma_prov_key, .init = ki_rma_prov_key_setup,
	  .fini = ki_rma_teardown, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Build a Put or Get DMA command. Success events are suppressed (KI
 * completions are counter-only); eq/user_ptr are set so a failure still
 * reports through the endpoint's event queue.
 */
static void ki_build_dma(struct c_full_dma_cmd *cmd, bool write,
			 struct fi_cxi_gpu_remote_mr *remote_mr,
			 uint64_t local_off, uint64_t remote_off,
			 uint32_t len, uint32_t ctn)
{
	uint16_t ptl_idx;
	uint8_t idx_ext;

	memset(cmd, 0, sizeof(*cmd));

	ptl_idx = write ? remote_mr->write_ptl_idx : remote_mr->read_ptl_idx;

	cmd->command.opcode = write ? C_CMD_PUT : C_CMD_GET;
	cxi_build_dfa(ki_self_target.nid, ki_self_target.pid,
		      ki_ep_info.pid_bits, ptl_idx, &cmd->dfa, &idx_ext);
	cmd->index_ext = idx_ext;

	cmd->event_send_disable = 1;
	cmd->event_success_disable = 1;
	cmd->eq = ki_ep_info.eqn;
	cmd->user_ptr = ki_ep_info.user_ptr;

	cmd->initiator = ki_initiator;
	cmd->match_bits = remote_mr->match_bits;
	cmd->remote_offset = remote_mr->offset_adjust + remote_off;
	cmd->lac = ki_local_mr_info.lac;
	cmd->local_addr = ki_local_mr_info.iova + local_off;
	cmd->request_len = len;

	cmd->ct = ctn;
	if (write)
		cmd->event_ct_ack = 1;
	else
		cmd->event_ct_reply = 1;
}

/* Load a libcxi command queue from the GPU command queue state so the
 * tests can use libcxi's command builders. ki_cq_save() writes the
 * updated pointers back. Only the fields the emit and ring paths use
 * are populated.
 */
static void ki_cq_load(struct cxi_cq *ring,
		       const struct fi_cxi_gpu_cmdq_state *state)
{
	memset(ring, 0, sizeof(*ring));
	ring->cmds32 = state->cmds32;
	ring->wp_addr = state->wp_addr;
	ring->wp32 = state->wp32;
	ring->hw_wp32 = state->hw_wp32;
	ring->rp32 = state->rp32;
	ring->size32 = state->size32;
}

static void ki_cq_save(struct fi_cxi_gpu_cmdq_state *state,
		       const struct cxi_cq *ring)
{
	state->wp32 = ring->wp32;
	state->hw_wp32 = ring->hw_wp32;
	state->rp32 = ring->rp32;
}

/* Emit one DMA command and ring the doorbell */
static void ki_emit(struct fi_cxi_gpu_cmdq_state *state,
		    struct c_full_dma_cmd *cmd)
{
	struct cxi_cq ring;
	int ret;

	ki_cq_load(&ring, state);
	ret = cxi_cq_emit_dma(&ring, cmd);
	cr_assert_eq(ret, 0, "cxi_cq_emit_dma returned %d", ret);
	cxi_cq_ring(&ring);
	ki_cq_save(state, &ring);
}

/* Wait for a counter value (fi_cntr_read or fi_cntr_readerr) to reach
 * expected. Progress is driven through the TX CQ so error completions
 * are delivered.
 */
static void ki_cntr_wait_fn(struct fid_cntr *cntr,
			    uint64_t (*read)(struct fid_cntr *cntr),
			    uint64_t expected, const char *what)
{
	int ret;
	struct timespec end;
	struct timespec cur;
	struct fi_cq_tagged_entry cqe;
	uint64_t value;

	ret = clock_gettime(CLOCK_MONOTONIC, &end);
	cr_assert_eq(ret, 0);

	end.tv_sec += KI_RMA_WAIT_SEC;

	while (true) {
		ret = clock_gettime(CLOCK_MONOTONIC, &cur);
		cr_assert_eq(ret, 0);

		value = read(cntr);
		if (value >= expected)
			break;

		cr_assert(cur.tv_sec <= end.tv_sec,
			  "%s stuck at %lu, expected %lu\n", what, value,
			  expected);

		fi_cq_read(cxit_tx_cq, &cqe, 1);
		sched_yield();
	}
}

/* Wait for a counter's success count to reach a value */
static void ki_cntr_wait(struct fid_cntr *cntr, uint64_t expected)
{
	ki_cntr_wait_fn(cntr, fi_cntr_read, expected, "Counter");
}

/* Wait for a counter's error count to reach a value */
static void ki_cntr_wait_err(struct fid_cntr *cntr, uint64_t expected)
{
	ki_cntr_wait_fn(cntr, fi_cntr_readerr, expected, "Counter error");
}

/* Validate payload and the bytes on either side of it */
static void ki_validate(uint8_t *buf, size_t off, size_t len)
{
	int i;

	for (i = 0; i < len; i++)
		cr_assert_eq(buf[off + i], KI_RMA_PUT_VAL,
			     "data mismatch, element: (%d) %02x != %02x\n", i,
			     buf[off + i], KI_RMA_PUT_VAL);

	cr_assert_eq(buf[off - 1], KI_RMA_INIT_VAL, "undershoot at %zu",
		     off - 1);
	cr_assert_eq(buf[off + len], KI_RMA_INIT_VAL, "overshoot at %zu",
		     off + len);
}

/* PUT len bytes from ki_mem_window at src_off to remote_mr at dst_off,
 * wait for the send counter, and validate the landing zone in dst_buf
 * (the host view of remote_mr).
 */
static void ki_put_once(struct fi_cxi_gpu_remote_mr *remote_mr,
			uint8_t *dst_buf, size_t dst_len, size_t src_off,
			size_t dst_off, size_t len)
{
	struct c_full_dma_cmd cmd;
	uint8_t *buf = ki_mem_window.mem;
	uint64_t initial;

	initial = fi_cntr_read(cxit_send_cntr);

	memset(dst_buf, KI_RMA_INIT_VAL, dst_len);
	memset(buf, KI_RMA_INIT_VAL, KI_RMA_WIN_LEN);
	memset(buf + src_off, KI_RMA_PUT_VAL, len);

	ki_build_dma(&cmd, true, remote_mr, src_off, dst_off, len,
		     ki_ep_info.ctn);
	ki_emit(&ki_ep_info.tx_cmdq.cq, &cmd);

	ki_cntr_wait(cxit_send_cntr, initial + 1);
	ki_validate(dst_buf, dst_off, len);
}

/* Test a single PUT */
Test(ki_rma, put_verify)
{
	int ret;
	struct fi_cq_tagged_entry cqe;
	uint8_t *buf = ki_mem_window.mem;
	uint64_t initial;
	size_t src_off = 0;
	size_t dst_off = 256;
	size_t len = 64;

	initial = fi_cntr_read(cxit_send_cntr);

	ki_put_once(&ki_self_remote_mr, buf, KI_RMA_WIN_LEN, src_off, dst_off,
		    len);

	/* Exactly one completion, and nothing on the TX CQ: KI success
	 * events are suppressed.
	 */
	usleep(KI_RMA_SETTLE_US);
	cr_assert_eq(fi_cntr_read(cxit_send_cntr), initial + 1,
		     "send counter advanced past %lu", initial + 1);
	cr_assert_eq(fi_cntr_readerr(cxit_send_cntr), 0,
		     "send counter reported an error");
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read returned %d", ret);
}

/* Test a PUT that fails at the target. The command addresses an
 * optimized PTL index with no MR behind it, so the target drops it.
 * The failure must be counted on the send counter's error count and
 * reported through the TX CQ with the endpoint's context.
 */
Test(ki_rma, put_error)
{
	int ret;
	struct c_full_dma_cmd cmd;
	struct fi_cq_tagged_entry cqe;
	struct fi_cq_err_entry err = {};
	struct fi_cxi_gpu_remote_mr bad_remote_mr = ki_self_remote_mr;
	uint8_t *buf = ki_mem_window.mem;
	uint64_t initial;
	uint64_t initial_err;
	size_t src_off = 0;
	size_t dst_off = 256;
	size_t len = 64;

	/* No MR was registered with this optimized key */
	bad_remote_mr.write_ptl_idx = CXIP_PTL_IDX_WRITE_MR_OPT(7);

	initial = fi_cntr_read(cxit_send_cntr);
	initial_err = fi_cntr_readerr(cxit_send_cntr);

	memset(buf, KI_RMA_INIT_VAL, KI_RMA_WIN_LEN);
	memset(buf + src_off, KI_RMA_PUT_VAL, len);

	ki_build_dma(&cmd, true, &bad_remote_mr, src_off, dst_off, len,
		     ki_ep_info.ctn);
	ki_emit(&ki_ep_info.tx_cmdq.cq, &cmd);

	ki_cntr_wait_err(cxit_send_cntr, initial_err + 1);
	cr_assert_eq(fi_cntr_read(cxit_send_cntr), initial,
		     "send counter success count advanced on error");
	cr_assert_eq(buf[dst_off], KI_RMA_INIT_VAL,
		     "payload landed despite error");

	/* The error is on the TX CQ; user_ptr maps it back to the EP */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert_eq(ret, -FI_EAVAIL, "fi_cq_read returned %d", ret);

	ret = fi_cq_readerr(cxit_tx_cq, &err, 0);
	cr_assert_eq(ret, 1, "fi_cq_readerr returned %d", ret);
	cr_assert_eq(err.flags, 0, "err.flags is 0x%lx", err.flags);
	cr_assert_null(err.op_context, "err.op_context is %p",
		       err.op_context);
	cr_assert_neq(err.err, 0, "err.err is zero");

	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert_eq(ret, -FI_EAGAIN, "fi_cq_read returned %d", ret);
}

/* Test a PUT to an MR with a standard (non-optimized) key. The target
 * is a matching LE on the standard PTL index, so the command must carry
 * the match bits decoded from the peer blob.
 */
Test(ki_rma, put_std_key)
{
	struct mem_region std_window;
	struct fi_cxi_gpu_remote_mr std_remote_mr;
	uint64_t key = CXIP_PTL_IDX_MR_OPT_CNT + KI_RMA_WIN_KEY;
	size_t src_off = 0;
	size_t dst_off = 256;
	size_t len = 64;

	cxit_ki_create_remote_mr(s_page_size, FI_REMOTE_WRITE, &key,
				 &std_window, NULL, &std_remote_mr);
	cr_assert(std_remote_mr.unrestricted, "standard MR not unrestricted");
	cr_assert_eq(std_remote_mr.write_ptl_idx, CXIP_PTL_IDX_WRITE_MR_STD,
		     "write_ptl_idx is %u", std_remote_mr.write_ptl_idx);

	ki_put_once(&std_remote_mr, std_window.mem, s_page_size, src_off,
		    dst_off, len);

	mr_destroy(&std_window);
}

/* Test a PUT to an MR with a cached provider key. The LE spans the
 * whole LAC, so the command's remote offset must be the MR's IOVA
 * (carried in the key as lac_off and returned as offset_adjust) plus
 * the MR-relative offset.
 */
Test(ki_rma_prov_key, put_cached_key)
{
	struct cxip_mr_key key = { .raw = fi_mr_key(ki_mem_window.mr) };
	uint8_t *buf = ki_mem_window.mem;
	size_t src_off = 0;
	size_t dst_off = 256;
	size_t len = 64;

	if (!key.cached)
		cr_skip_test("MR key is not cached");

	cr_assert(key.is_prov, "cached key is not a provider key");
	cr_assert_eq(ki_self_remote_mr.offset_adjust, ki_self_remote_mr.iova,
		     "offset_adjust 0x%lx != iova 0x%lx",
		     ki_self_remote_mr.offset_adjust, ki_self_remote_mr.iova);
	cr_assert_neq(ki_self_remote_mr.offset_adjust, 0,
		      "offset_adjust is zero");
	cr_assert_eq(ki_self_remote_mr.match_bits,
		     CXIP_KEY_MATCH_BITS(key.raw), "match_bits 0x%lx",
		     ki_self_remote_mr.match_bits);

	ki_put_once(&ki_self_remote_mr, buf, KI_RMA_WIN_LEN, src_off, dst_off,
		    len);
}

/* Test a single GET. The source MR is FI_REMOTE_READ only, so the
 * transfer must use read_ptl_idx.
 */
Test(ki_rma, get_verify)
{
	struct c_full_dma_cmd cmd;
	struct mem_region ro_window;
	struct fi_cxi_gpu_remote_mr ro_remote_mr;
	uint8_t *buf = ki_mem_window.mem;
	uint64_t key = KI_RMA_WIN_KEY + 1;
	uint64_t initial;
	size_t src_off = 128;
	size_t dst_off = 256;
	size_t len = 64;

	cxit_ki_create_remote_mr(s_page_size, FI_REMOTE_READ, &key,
				 &ro_window, NULL, &ro_remote_mr);

	initial = fi_cntr_read(cxit_send_cntr);

	memset(ro_window.mem, KI_RMA_INIT_VAL, s_page_size);
	memset(ro_window.mem + src_off, KI_RMA_PUT_VAL, len);
	memset(buf, KI_RMA_INIT_VAL, KI_RMA_WIN_LEN);

	ki_build_dma(&cmd, false, &ro_remote_mr, dst_off, src_off, len,
		     ki_ep_info.ctn);
	ki_emit(&ki_ep_info.tx_cmdq.cq, &cmd);

	ki_cntr_wait(cxit_send_cntr, initial + 1);
	ki_validate(buf, dst_off, len);

	mr_destroy(&ro_window);
}

/* Test several PUTs emitted before a single doorbell ring */
Test(ki_rma, multi_put)
{
	int ret;
	int i;
	struct c_full_dma_cmd cmd;
	struct cxi_cq ring;
	uint8_t *buf = ki_mem_window.mem;
	uint64_t initial;
	size_t src_off = 0;
	size_t dst_base = 1024;
	size_t len = 64;
	int count = 8;

	initial = fi_cntr_read(cxit_send_cntr);

	memset(buf, KI_RMA_INIT_VAL, KI_RMA_WIN_LEN);
	memset(buf + src_off, KI_RMA_PUT_VAL, len);

	ki_cq_load(&ring, &ki_ep_info.tx_cmdq.cq);
	for (i = 0; i < count; i++) {
		ki_build_dma(&cmd, true, &ki_self_remote_mr, src_off,
			     dst_base + (i * 2 * len), len, ki_ep_info.ctn);
		ret = cxi_cq_emit_dma(&ring, &cmd);
		cr_assert_eq(ret, 0, "cxi_cq_emit_dma[%d] returned %d",
			     i, ret);
	}
	cxi_cq_ring(&ring);
	ki_cq_save(&ki_ep_info.tx_cmdq.cq, &ring);

	ki_cntr_wait(cxit_send_cntr, initial + count);

	for (i = 0; i < count; i++)
		ki_validate(buf, dst_base + (i * 2 * len), len);
}

/* Test a triggered PUT. It must stay pending until the trigger counter
 * reaches the threshold; an ordinary PUT routed to that counter fires it.
 */
Test(ki_rma, trig_put)
{
	int ret;
	struct c_full_dma_cmd dma;
	struct c_ct_cmd ct_cmd = {};
	struct cxi_cq ring;
	struct fid_cntr *trig_cntr;
	uint8_t *buf = ki_mem_window.mem;
	uint64_t trig_initial;
	uint64_t send_initial;
	uint32_t trig_ctn;
	size_t src_off = 0;
	size_t dst_off = 256;
	size_t fire_off = 2048;
	size_t len = 64;

	cr_assert_not_null(ki_ep_info.trig_cmdq.cq.wp_addr,
			   "no triggered cmdq published");

	ret = fi_cntr_open(cxit_domain, NULL, &trig_cntr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_cntr_open returned %d", ret);

	ret = cxit_ki_ops->get_cntr_number(&cxit_ep->fid, &trig_cntr->fid,
					   &trig_ctn);
	cr_assert_eq(ret, FI_SUCCESS, "get_cntr_number returned %d", ret);

	trig_initial = fi_cntr_read(trig_cntr);
	send_initial = fi_cntr_read(cxit_send_cntr);

	memset(buf, KI_RMA_INIT_VAL, KI_RMA_WIN_LEN);
	memset(buf + src_off, KI_RMA_PUT_VAL, len);

	/* Arm a triggered PUT that fires when trig_cntr reaches
	 * trig_initial + 1 and completes to the send counter.
	 */
	ki_build_dma(&dma, true, &ki_self_remote_mr, src_off, dst_off, len,
		     ki_ep_info.ctn);
	ct_cmd.trig_ct = trig_ctn;
	ct_cmd.threshold = trig_initial + 1;

	ki_cq_load(&ring, &ki_ep_info.trig_cmdq.cq);
	ret = cxi_cq_emit_trig_full_dma(&ring, &ct_cmd, &dma);
	cr_assert_eq(ret, 0, "cxi_cq_emit_trig_full_dma returned %d", ret);
	cxi_cq_ring(&ring);
	ki_cq_save(&ki_ep_info.trig_cmdq.cq, &ring);

	/* Not yet triggered */
	usleep(KI_RMA_SETTLE_US);
	cr_assert_eq(fi_cntr_read(cxit_send_cntr), send_initial,
		     "triggered PUT fired before threshold");
	cr_assert_eq(buf[dst_off], KI_RMA_INIT_VAL,
		     "payload landed before threshold");

	/* Fire: an ordinary PUT whose completion routes to trig_cntr */
	ki_build_dma(&dma, true, &ki_self_remote_mr, fire_off, fire_off, 32,
		     trig_ctn);
	ki_emit(&ki_ep_info.tx_cmdq.cq, &dma);

	ki_cntr_wait(trig_cntr, trig_initial + 1);
	ki_cntr_wait(cxit_send_cntr, send_initial + 1);
	ki_validate(buf, dst_off, len);

	fi_close(&trig_cntr->fid);
}
